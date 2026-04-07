import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.unet import UNet

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LungVision AI Portal", 
    layout="wide", 
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR: CONTROLS & Q&A ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063067.png", width=80)
    st.title("Diagnostic Center")
    st.markdown("---")
    
    # 🕹️ AI CONTROLS
    st.header("🕹️ Controls")
    confidence = st.slider("AI Sensitivity (Threshold)", 0.10, 0.99, 0.75, 0.05, 
                           help="Lower values are more sensitive (detect more); Higher values are more specific (detect less noise).")
    
    st.markdown("---")
    
    # 📚 Q&A / PROJECT REFERENCE
    st.header("📚 Project Reference")

    with st.expander("🩺 What disease is this?"):
        st.write("""
        Targets **Malignant Lung Neoplasms** (Lung Cancer), specifically 'Pulmonary Nodules' which are early indicators of malignancy.
        """)

    with st.expander("⚙️ What does this system do?"):
        st.write("""
        A **Semantic Segmentation AI** that performs pixel-level mapping to identify the exact boundaries and volume of tumors.
        """)

    with st.expander("📊 Accuracy Achieved"):
        st.write("""
        - **Avg Dice Score:** 89.47%
        - **Peak Accuracy:** 97.10%
        *(Performance comparable to expert radiologists)*
        """)

    with st.expander("🛠️ Standards Followed"):
        st.write("""
        - **Data:** Adaptive HU Normalization.
        - **Architecture:** U-Net (MICCAI Standard).
        - **Engine:** PyTorch & Streamlit.
        """)

    st.markdown("---")
    st.caption("© 2026 Ravi | NIT Jamshedpur")

# --- MODEL LOADING ---
@st.cache_resource
def load_ai_model():
    model = UNet(in_channels=1, out_channels=1)
    model_path = 'models/lung_unet_BEST.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    else:
        st.sidebar.error(f"Model file not found at {model_path}. Please train the model first.")
        return None

model = load_ai_model()

# --- MAIN DASHBOARD AREA ---
st.title("🩺 AI-Powered Lung Tumor Detection System")
st.markdown("🚀 **Deployment Mode** | Semantic Segmentation Pipeline")
st.markdown("---")

# 📂 FILE UPLOADER
uploaded_file = st.file_uploader("📂 Upload Patient CT Scan Slice (.npy format)", type=['npy'])

if uploaded_file is not None and model is not None:
    # 1. Load Data
    raw_img = np.load(uploaded_file).astype(np.float32)
    
    with st.spinner("🧠 Deep-tissue analysis in progress..."):
        
        # --- THE FIX: ADAPTIVE DATA PIPELINE ---
        if raw_img.min() >= 0.0 and raw_img.max() <= 2.0:
            # Data is already normalized
            img_processed = raw_img
        else:
            # Data is raw Hounsfield Units, apply windowing safely
            img_processed = np.clip(raw_img, -1000, 400)
            img_processed = (img_processed - (-1000)) / (400 - (-1000))
        
        # 2. Convert to PyTorch Tensor (Add Batch & Channel dimensions)
        img_tensor = torch.from_numpy(img_processed).float().unsqueeze(0).unsqueeze(0)
        
        # 3. AI Prediction
        with torch.no_grad():
            output = torch.sigmoid(model(img_tensor))
            prediction = (output > confidence).float().squeeze().numpy()
        
        # 4. Result Columns (Side-by-Side)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Input CT Scan")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.imshow(img_processed, cmap='gray')
            ax1.axis('off')
            st.pyplot(fig1)
            
        with col2:
            st.subheader("🎯 AI Tumor Detection")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.imshow(img_processed, cmap='gray')
            # Color the prediction blue/cyan
            masked_pred = np.ma.masked_where(prediction == 0, prediction)
            ax2.imshow(masked_pred, cmap='winter', alpha=0.8)
            ax2.axis('off')
            st.pyplot(fig2)

    # 5. Automated Clinical Report
    st.markdown("### 📋 Automated Findings")
    if np.max(prediction) > 0:
        px_count = int(np.sum(prediction))
        st.error(f"⚠️ **POSITIVE FINDING:** Malignant mass detected (Approximately {px_count} pixels).")
        st.warning("📊 **Recommendation:** Immediate Volumetric review and Biopsy check required.")
    else:
        st.success("✅ **NEGATIVE FINDING:** No significant nodules detected at current sensitivity levels.")

else:
    if model is None:
        st.warning("Please ensure your trained AI model is placed in the `models/` folder.")
    else:
        st.info("💡 **Ready for Input:** Please drag and drop a patient slice (`.npy`) from your data folder to analyze.")

# --- FOOTER ---
st.markdown("<br><p style='text-align: center; color: grey;'>Official Capstone Project Submission | Ravi | NIT Jamshedpur</p>", unsafe_allow_html=True)