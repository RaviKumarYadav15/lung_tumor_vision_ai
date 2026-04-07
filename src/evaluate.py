# src/evaluate.py
import os
import torch
import numpy as np
from unet import UNet

def calculate_dice(pred_mask, true_mask, smooth=1e-6):
    """Calculates the mathematical overlap between two masks."""
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    intersection = np.sum(pred_flat * true_flat)
    return (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(true_flat) + smooth)

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🧠 Loading Global AI Model...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('models/lung_unet_BEST.pth', map_location=device))
    model.eval()

    # --- 1. UNIVERSAL SANITY CHECK ---
    # Safely checks the first layer's weights to guarantee the model loaded
    weight_sum = next(model.parameters()).sum().item()
    print(f"🔍 System Check | Model Weight Signature: {weight_sum:.4f}")
    print("-" * 50)

    # 2. TEST A VALIDATION PATIENT
    patient_id = "57" 
    data_dir = f'data/raw/val/{patient_id}/data/'
    mask_dir = f'data/raw/val/{patient_id}/masks/'
    
    threshold = 0.40 # Keeping it at 0.40 to see all detections clearly
    dice_scores = []

    print(f"📊 Evaluating Validation Patient {patient_id} at Threshold {threshold}...")
    print("-" * 50)

    with torch.no_grad():
        for filename in os.listdir(mask_dir):
            if not filename.endswith('.npy'):
                continue
                
            mask_path = os.path.join(mask_dir, filename)
            img_path = os.path.join(data_dir, filename)
            
            # Only evaluate slices that actually have a tumor
            true_mask = np.load(mask_path)
            if np.max(true_mask) == 0:
                continue 

            true_mask = (true_mask > 0).astype(np.float32)
            
            # Load image safely
            raw_img = np.load(img_path).astype(np.float32)
            
            # --- 3. ADAPTIVE DATA PIPELINE ---
            # If the image is already normalized (values between 0 and ~1.0)
            if raw_img.min() >= 0.0 and raw_img.max() <= 2.0:
                img_processed = raw_img
            else:
                # If it is raw Hounsfield Units (-1000 to +400 range)
                img_processed = np.clip(raw_img, -1000, 400)
                img_processed = (img_processed - (-1000)) / (400 - (-1000))
            
            # Predict
            img_tensor = torch.from_numpy(img_processed).unsqueeze(0).unsqueeze(0).to(device)
            pred = torch.sigmoid(model(img_tensor))
            
            # Score
            max_conf = pred.max().item()
            pred_binary = (pred > threshold).float().cpu().numpy().squeeze()
            score = calculate_dice(pred_binary, true_mask)
            dice_scores.append(score)
            
            print(f"Slice {filename:>7} | Max Conf: {max_conf*100:>5.2f}% | Dice Score: {score*100:>5.2f}%")
            
    if len(dice_scores) == 0:
        print("⚠️ No slices with tumors found for this patient.")
        return

    final_score = np.mean(dice_scores)
    print("-" * 50)
    print(f"✅ Evaluation Complete for Patient {patient_id}!")
    print(f"🎯 Average Dice Score: {final_score * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()