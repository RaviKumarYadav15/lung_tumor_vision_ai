# src/predict.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet

def test_ai():
    print("🩺 Loading AI Doctor for inference...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('models/lung_unet_BEST.pth', map_location=device))
    model.eval()

    image_path = 'data/raw/val/57/data/210.npy'
    mask_path = 'data/raw/val/57/masks/210.npy'

    raw_img = np.load(image_path).astype(np.float32)
    ground_truth_mask = np.load(mask_path)

    if raw_img.min() >= 0.0 and raw_img.max() <= 2.0:
        img_processed = raw_img
    else:
        img_processed = np.clip(raw_img, -1000, 400)
        img_processed = (img_processed - (-1000)) / (400 - (-1000))
    
    img_tensor = torch.from_numpy(img_processed).float().unsqueeze(0).unsqueeze(0).to(device)

    print("🧠 AI is analyzing the scan...")
    with torch.no_grad():
        raw_prediction = model(img_tensor)
        probability_mask = torch.sigmoid(raw_prediction)
        threshold = 0.40 
        binary_prediction = (probability_mask > threshold).float()

    ai_mask = binary_prediction.squeeze().cpu().numpy()
    
    print(f"🎯 AI Max Confidence: {probability_mask.max().item() * 100:.2f}%")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(img_processed, cmap='gray')
    axes[0].set_title("Input")
    axes[0].axis('off')

    axes[1].imshow(img_processed, cmap='gray')
    masked_truth = np.ma.masked_where(ground_truth_mask == 0, ground_truth_mask)
    axes[1].imshow(masked_truth, cmap='autumn', alpha=0.9) 
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(img_processed, cmap='gray')
    masked_ai = np.ma.masked_where(ai_mask == 0, ai_mask)
    axes[2].imshow(masked_ai, cmap='winter', alpha=0.9)
    axes[2].set_title(f"Prediction (T={threshold})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_ai()