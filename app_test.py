import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.unet import UNet  # Imports your architecture from the src folder

# --- PAGE SETUP ---
st.set_page_config(page_title="Lung Tumor AI", layout="wide")
st.title("🩺 AI-Powered Lung Tumor Detection System")
st.markdown("**Developed by Ravi | NIT Jamshedpur Capstone Project**")
st.markdown("---")

# --- CACHE THE MODEL ---
# This prevents the app from reloading the heavy AI model every time you move a slider
@st.cache_resource
def load_ai():
    device = torch.device('cpu') # Using CPU ensures the web app doesn't crash on standard laptops
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Pointing to your highly accurate 100-epoch model!
    model_path = 'models/lung_unet_v1_new.pth' 
    
    if not os.path.exists(model_path):
        st.error(f"Cannot find model at {model_path}. Please check your folder structure.")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_ai()

# --- UI CONTROLS (Sidebar) ---
st.sidebar.header("Control Panel")
patient_id = st.sidebar.selectbox("Select Patient", ["0", "1", "2"])

# Defaulting to slice 340 since we know it's a great example for your presentation
slice_id = st.sidebar.slider("Select CT Slice", 10, 400, 340)

# Defaulting to 0.75, the clinical sweet spot we proved mathematically
confidence = st.sidebar.slider("AI Confidence Threshold", 0.10, 0.99, 0.75, 0.01)

st.sidebar.markdown("---")
analyze_button = st.sidebar.button("🔍 Analyze Scan", use_container_width=True)

# --- RUN INFERENCE ---
if analyze_button and model is not None:
    img_path = f'data/raw/train/{patient_id}/data/{slice_id}.npy'
    
    if os.path.exists(img_path):
        with st.spinner("AI is analyzing the scan..."):
            # 1. Load Data
            raw_img = np.load(img_path)
            
            # 2. DIP Processing (Hounsfield Windowing)
            img_processed = np.clip(raw_img, -1000, 400)
            img_processed = (img_processed - (-1000)) / (400 - (-1000))
            
            # Convert to PyTorch Tensor
            img_tensor = torch.from_numpy(img_processed).float().unsqueeze(0).unsqueeze(0)
            
            # 3. AI Prediction
            with torch.no_grad():
                prob_mask = torch.sigmoid(model(img_tensor))
                # Apply the slider's threshold
                ai_mask = (prob_mask > confidence).float().squeeze().numpy()
                
            # 4. Draw Results Side-by-Side
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            
            # Left Image: Original Scan
            axes[0].imshow(img_processed, cmap='gray')
            axes[0].set_title(f"Patient {patient_id} | Slice {slice_id}", fontsize=14)
            axes[0].axis('off')
            
            # Right Image: AI Overlay
            axes[1].imshow(img_processed, cmap='gray')
            # Only color the pixels where the AI found a tumor
            visible_ai = np.ma.masked_where(ai_mask == 0, ai_mask)
            axes[1].imshow(visible_ai, cmap='winter', alpha=0.9)
            axes[1].set_title(f"AI Detection (Confidence: {confidence * 100:.0f}%)", fontsize=14)
            axes[1].axis('off')
            
            # Render the plot in Streamlit
            st.pyplot(fig)
            
            # 5. Clinical Alert System
            if np.max(ai_mask) > 0:
                st.error("⚠️ **ALERT: Potential Malignant Nodule Detected.** Medical review required.")
            else:
                st.success("✅ **Clean Scan:** No nodules detected at this confidence level.")
    else:
        st.warning(f"Slice {slice_id} does not exist for Patient {patient_id}. Please select a different slice.")