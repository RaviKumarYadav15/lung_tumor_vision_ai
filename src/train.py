import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import LungSegmentationDataset
from unet import UNet
from metrics import DiceLoss

# --- 1. HYPERPARAMETERS (Fast & Safe Mode) ---
BATCH_SIZE = 16  
LEARNING_RATE = 1e-4
EPOCHS = 10        

def train_model():
    print("🚀 Initializing Global Training Pipeline...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    train_root_dir = 'data/raw/train/'

    print("📂 Crawling directories for Patient Data...")
    train_dataset = LungSegmentationDataset(train_root_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("🧠 Booting up the U-Net...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

    print(f"🔥 Starting Training for {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   Epoch [{epoch+1}/{EPOCHS}] | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"🏁 Epoch {epoch+1} Completed. Average Dice Loss: {avg_loss:.4f}")
        
        # --- NEW: SAVE AFTER EVERY EPOCH (Colab Safety) ---
        checkpoint_path = f'models/lung_unet_checkpoint_e{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}\n")

    # --- 6. FINAL SAVE ---
    print("💾 Training Complete! Saving final model weights...")
    torch.save(model.state_dict(), 'models/lung_unet_global_v2.pth')
    print("✅ Final Model successfully saved to: models/lung_unet_global_v2.pth")

if __name__ == "__main__":
    train_model()