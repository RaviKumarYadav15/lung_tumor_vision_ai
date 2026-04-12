import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv
import numpy as np

# Specific to Google Colab
from google.colab import drive

from dataset import LungSegmentationDataset
from unet import UNet
# UPGRADE 1: Import the new Combo Loss instead of standard DiceLoss
from metrics import BCEDiceLoss 

# --- 1. HYPERPARAMETERS & DRIVE SETUP ---
BATCH_SIZE = 16      
LEARNING_RATE = 1e-4
# UPGRADE 3: Increased epochs to 30 to let the scheduler work
EPOCHS = 30          

# Mount Google Drive
# drive.mount('/content/drive')

# Create a specific folder in your Drive to save everything permanently
DRIVE_SAVE_PATH = '/content/drive/MyDrive/Capstone_Lung_AI_Results'
os.makedirs(DRIVE_SAVE_PATH, exist_ok=True)
print(f"📂 All artifacts will be saved permanently to: {DRIVE_SAVE_PATH}")

def calculate_dice(pred_mask, true_mask, smooth=1e-6):
    pred_flat = pred_mask.view(-1)
    true_flat = true_mask.view(-1)
    intersection = (pred_flat * true_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)

def train_colab():
    print("🚀 Initializing Cloud Training Pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = LungSegmentationDataset('data/raw/train/')
    val_dataset = LungSegmentationDataset('data/raw/val/') 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # UPGRADE 1: Initialize the Combo Loss
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    # UPGRADE 2: Initialize the Learning Rate Scheduler
    # Cuts the learning rate in half if the Val Loss gets stuck for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_val_dice = 0.0

    print(f"🔥 Starting Training & Validation for {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                predictions = model(images)
                loss = criterion(predictions, masks)
                val_loss += loss.item()
                preds_binary = (torch.sigmoid(predictions) > 0.40).float()
                val_dice += calculate_dice(preds_binary, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_dice)

        print(f"🏁 Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice*100:.2f}%")
        
        # UPGRADE 2: Step the scheduler based on Validation Loss
        scheduler.step(avg_val_loss)
        
        # --- SAVE BEST MODEL DIRECTLY TO DRIVE ---
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            model_path = os.path.join(DRIVE_SAVE_PATH, 'lung_unet_BEST.pth')
            torch.save(model.state_dict(), model_path)
            print(f"   🌟 New Best Model Saved to Drive! (Dice: {best_val_dice*100:.2f}%)")
            
        # --- SAVE CHECKPOINT EVERY EPOCH ---
        checkpoint_path = os.path.join(DRIVE_SAVE_PATH, f'lung_unet_checkpoint_e{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

    # --- FINAL GLOBAL SAVE ---
    final_path = os.path.join(DRIVE_SAVE_PATH, 'lung_unet_global_FINAL.pth')
    torch.save(model.state_dict(), final_path)
    print(f"💾 Final Epoch Model saved to Drive: {final_path}")

    # --- REPORT GENERATION TO DRIVE ---
    print("\n📊 Generating Capstone Report Assets...")
    
    csv_path = os.path.join(DRIVE_SAVE_PATH, 'training_history.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Dice'])
        for i in range(EPOCHS):
            writer.writerow([i+1, history['train_loss'][i], history['val_loss'][i], history['val_dice'][i]])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS+1), history['train_loss'], label='Train Loss', color='blue')
    plt.plot(range(1, EPOCHS+1), history['val_loss'], label='Validation Loss', color='red')
    plt.title('Model Loss (Lower is Better)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (BCE + Dice)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), [d * 100 for d in history['val_dice']], label='Validation Dice', color='green')
    plt.title('Accuracy / Dice Score (Higher is Better)')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(DRIVE_SAVE_PATH, 'capstone_training_metrics.png')
    plt.savefig(plot_path)
    print(f"✅ Graph saved permanently to Drive: {plot_path}")

if __name__ == "__main__":
    train_colab()