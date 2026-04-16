import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time 
import datetime
import pydicom  
from src.unet import UNet

from modules.pdf_generator import generate_hospital_report
from modules.xai_visualizer import generate_confidence_heatmap

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LungVision AI", 
    layout="wide", 
    page_icon="🫁",
    initial_sidebar_state="expanded"
)

# Initialize Session States
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = int(time.time())
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "🩺 Single Slice Review"
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'applied_confidence' not in st.session_state:
    st.session_state.applied_confidence = 0.75 

# --- CUSTOM CSS ---
st.markdown("""
<style>
.led-indicator { width: 45px; height: 45px; border-radius: 50%; display: inline-block; border: 2px solid #2a2a2a; transition: all 0.3s ease; }
.led-red { background-color: #ff4757; box-shadow: 0 0 15px #ff4757, inset 0 0 5px rgba(255,255,255,0.5); animation: pulse-red 0.8s 5 ease-in-out; }
.led-green { background-color: #2ecc71; box-shadow: 0 0 15px #2ecc71, inset 0 0 5px rgba(255,255,255,0.5); animation: pulse-green 0.8s 5 ease-in-out; }
.led-idle { background-color: #555; box-shadow: inset 0 0 5px rgba(0,0,0,0.8); }
@keyframes pulse-red { 0%, 100% { opacity: 1; box-shadow: 0 0 20px #ff4757; } 50% { opacity: 0.4; box-shadow: 0 0 2px #ff4757; } }
@keyframes pulse-green { 0%, 100% { opacity: 1; box-shadow: 0 0 20px #2ecc71; } 50% { opacity: 0.4; box-shadow: 0 0 2px #2ecc71; } }

div[data-testid="stFileUploaderDropzone"] ~ div {
    max-height: 125px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONTROLS & Q&A ---
with st.sidebar:
    st.title("Diagnostic Center")    
    st.header("🕹️ Controls")
    
    confidence = st.slider("AI Sensitivity (Threshold)", 0.10, 0.99, 0.75, 0.05, 
                           help="Lower values are more sensitive (detect more); Higher values are more specific (detect less noise).",
                           disabled=st.session_state.is_processing)
    
    st.divider()
    
    if st.session_state.current_mode == "🩺 Single Slice Review":
        show_heatmap = st.checkbox("🔍 Show AI Confidence Heatmap", value=False, help="Overlays the raw neural network probability scores.", disabled=st.session_state.is_processing)
    else:
        show_heatmap = False
        
    st.header("📚 Project Reference")
    with st.expander("🩺 What disease is this?"):
        st.write("Targets **Malignant Lung Neoplasms** (Lung Cancer), specifically 'Pulmonary Nodules'.")
    with st.expander("⚙️ What does this system do?"):
        st.write("A **Semantic Segmentation AI** that performs pixel-level mapping to identify the exact boundaries and volume of tumors.")
    with st.expander("📊 Accuracy Achieved"):
        st.write("- **Avg Dice Score:** 53.43% \n- **Peak Accuracy:** 97.10%")
    with st.expander("🛠️ Standards Followed"):
        st.write("- **Data:** Adaptive HU Normalization.\n- **Architecture:** U-Net (MICCAI Standard).\n- **Engine:** PyTorch & Streamlit.")
    with st.expander("🔒 Privacy & Security"):
        st.write("- **Zero-Retention:** Patient scans are processed locally in RAM and instantly destroyed. No PHI is stored.")

    st.header("📖 User Guide")
    with st.expander("How to use this app"):
        st.write("1. Choose **Single Slice** for individual `.npy` or `.dcm` files.\n2. Choose **Volumetric** to upload an entire folder of slices.\n3. Adjust **AI Sensitivity** if the AI is missing tumors or showing too much noise.\n4. Download the **PDF Report** for a clinical summary.")
        
    st.divider()
    st.caption("© 2026 Ravi | NIT Jamshedpur")

# --- MODEL LOADING ---
@st.cache_resource
def load_ai_model():
    model = UNet(in_channels=1, out_channels=1)
    model_path = 'models/lung_unet_BEST.pth'
    
    if not os.path.exists(model_path):
        status_box = st.empty() 
        status_box.info("⬇️ Downloading AI model weights from Google Drive...")
        os.makedirs('models', exist_ok=True)
        file_id = '1wj9Noii5LLgJehLWSoxfGXh0idggLgsf' 
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            import gdown
            gdown.download(url, model_path, quiet=False)
            status_box.success("✅ Model downloaded successfully!")
            time.sleep(2)
            status_box.empty() 
        except Exception as e:
            status_box.error(f"Failed to download model from Google Drive: {e}")
            return None

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    return None

model = load_ai_model()

def preprocess_scan(uploaded_file):
    try:
        if uploaded_file.name.endswith('.npy'):
            raw_img = np.load(uploaded_file).astype(np.float32)
            if raw_img.min() >= 0.0 and raw_img.max() <= 2.0:
                return raw_img
            else:
                img_processed = np.clip(raw_img, -1000, 400)
                return (img_processed - (-1000)) / (400 - (-1000))
                
        elif uploaded_file.name.endswith('.dcm'):
            dicom_data = pydicom.dcmread(uploaded_file)
            raw_img = dicom_data.pixel_array.astype(np.float32)
            intercept = getattr(dicom_data, 'RescaleIntercept', 0)
            slope = getattr(dicom_data, 'RescaleSlope', 1)
            hu_img = raw_img * slope + intercept
            img_processed = np.clip(hu_img, -1000, 400)
            return (img_processed - (-1000)) / (400 - (-1000))
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return None

# --- MAIN HEADER ---
st.markdown("<h1>🩺 AI-Powered Lung Tumor Detection</h1>", unsafe_allow_html=True)
st.divider()

indicator_status = "led-idle"
mode_col, led_col = st.columns([3, 1])

with mode_col:
    new_mode = st.radio("Select Diagnostic Mode:", ["🩺 Single Slice Review", "📂 Volumetric Review"], horizontal=True, disabled=st.session_state.is_processing)

led_placeholder = led_col.empty()

if new_mode != st.session_state.current_mode:
    for key in ['processed_batch', 'single_result', 'single_file_name', 'batch_length', 'summary', 'slider_labels', 'vol_pdf_bytes', 'vol_pdf_name']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.current_mode = new_mode
    st.session_state.uploader_key = int(time.time())
    st.rerun()

st.divider()

# ==========================================
# MODE 1: SINGLE SLICE REVIEW
# ==========================================
if st.session_state.current_mode == "🩺 Single Slice Review":
    left_col, mid_col, right_col = st.columns([1, 1.2, 1.2], gap="medium")
    
    with left_col:
        uploaded_file = st.file_uploader("📂 Upload Patient CT Scan (.npy, .dcm)", type=['npy', 'dcm'], key=f"single_{st.session_state.uploader_key}", disabled=st.session_state.is_processing)
        
        if uploaded_file is not None and model is not None:
            file_changed = ('single_file_name' not in st.session_state) or (st.session_state.single_file_name != uploaded_file.name)
            slider_changed = st.session_state.applied_confidence != confidence

            if file_changed or slider_changed:
                with st.spinner("🧠 Deep-tissue analysis in progress..."):
                    img_processed = preprocess_scan(uploaded_file)
                    
                    if img_processed is not None:
                        img_tensor = torch.from_numpy(img_processed).float().unsqueeze(0).unsqueeze(0)
                        with torch.no_grad():
                            raw_sigmoid_output = torch.sigmoid(model(img_tensor)).squeeze().numpy()
                            prediction = (raw_sigmoid_output > confidence).astype(float)
                
                        px_count = int(np.sum(prediction))
                        st.session_state.single_result = {
                            "img_processed": img_processed,
                            "raw_sigmoid_output": raw_sigmoid_output,
                            "prediction": prediction,
                            "px_count": px_count
                        }
                        st.session_state.single_file_name = uploaded_file.name
                        st.session_state.applied_confidence = confidence 

            if st.button("🗑️ Clear Image", type="secondary", use_container_width=True):
                for key in ['single_result', 'single_file_name']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.uploader_key = int(time.time())
                st.rerun()

    if 'single_result' in st.session_state:
        res = st.session_state.single_result
        risk_level = "HIGH" if res['px_count'] > 0 else "LOW"
        indicator_status = "led-red" if res['px_count'] > 0 else "led-green"
        
        tumor_area_mm2 = res['px_count'] * 1.0  
        estimated_diameter_mm = (np.sqrt(tumor_area_mm2 / np.pi) * 2) if res['px_count'] > 0 else 0
        
        with left_col:
            st.subheader("📋 Automated Findings")
            if res['px_count'] > 0:
                st.error(f"⚠️ **POSITIVE FINDING:** Malignant mass detected (Approx. {res['px_count']} px).")
                st.warning("📊 **Recommendation:** Immediate Volumetric review required.")
            else:
                st.success("✅ **NEGATIVE FINDING:** No significant nodules detected.")

            st.caption("💾 **Export Clinical Report**")
            
            # REDUCED FIGSIZE TO FIT 1 PDF PAGE
            fig_export, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 3.5), facecolor='#0B1215')
            ax_left.imshow(res['img_processed'], cmap='gray'); ax_left.axis('off')
            ax_right.imshow(res['img_processed'], cmap='gray')
            masked_export = np.ma.masked_where(res['prediction'] == 0, res['prediction'])
            ax_right.imshow(masked_export, cmap='winter', alpha=0.8); ax_right.axis('off')
            fig_export.tight_layout()
            
            temp_image_path = "temp_scan_export.png"
            fig_export.savefig(temp_image_path, format="png", bbox_inches='tight', facecolor='#0B1215')
            plt.close(fig_export)
            
            unique_patient_id = f"P-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # SECURE ZERO-RETENTION PDF GENERATION
            try:
                pdf_bytes = generate_hospital_report(unique_patient_id, tumor_area_mm2, estimated_diameter_mm, risk_level, temp_image_path)
                st.download_button(label="📄 Download Official PDF Report", data=pdf_bytes, file_name=f"Report_{unique_patient_id}.pdf", mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error("Please ensure fpdf2 is installed to generate reports.")
            finally:
                # Instantly destroy the temporary image from the disk
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

        with mid_col:
            st.subheader("⚕️Input CT Scan")
            fig1, ax1 = plt.subplots(figsize=(4, 4)) 
            ax1.imshow(res['img_processed'], cmap='gray'); ax1.axis('off')
            fig1.patch.set_alpha(0.0); fig1.tight_layout(pad=0)
            st.pyplot(fig1, width="stretch"); plt.close(fig1)
            
        with right_col:
            st.subheader("AI Detection")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(res['img_processed'], cmap='gray')
            masked_pred = np.ma.masked_where(res['prediction'] == 0, res['prediction'])
            ax2.imshow(masked_pred, cmap='winter', alpha=0.8); ax2.axis('off')
            fig2.patch.set_alpha(0.0); fig2.tight_layout(pad=0)
            st.pyplot(fig2, width="stretch"); plt.close(fig2)

        if show_heatmap:
            st.divider()
            st.subheader("🔍 Confidence Heatmap")
            xai_fig = generate_confidence_heatmap(res['img_processed'], res['raw_sigmoid_output'])
            heat_col1, heat_col2, heat_col3 = st.columns([1, 2, 1])
            with heat_col2: st.pyplot(xai_fig, width="stretch")

# ==========================================
# MODE 2: VOLUMETRIC REVIEW
# ==========================================
elif st.session_state.current_mode == "📂 Volumetric Review":
    
    if 'processed_batch' in st.session_state and len(st.session_state.processed_batch) > 0:
        if confidence != st.session_state.applied_confidence:
            st.warning(f"⚠️ AI Sensitivity changed to **{confidence}**. Click 'Analyse Again' below to update the Volumetric Folder.")

    # 1. FULL WIDTH STOP BUTTON (OUTSIDE COLUMNS)
    if st.session_state.is_processing:
        if st.button("🛑 Stop Analysis", type="secondary", use_container_width=True):
            st.session_state.is_processing = False
            st.rerun()

    # Define columns
    batch_col_left, batch_col_right = st.columns([1, 2], gap="large")
    
    # 2. CREATE A PLACEHOLDER FOR THE RIGHT COLUMN TO RENDER DURING PROCESSING
    with batch_col_right:
        right_ui_placeholder = st.empty()

    with batch_col_left:
        uploaded_folder = st.file_uploader(
            "📂 Upload CT Scan Folder (.npy, .dcm)", 
            type=['npy', 'dcm'], 
            accept_multiple_files=True, 
            key=f"batch_{st.session_state.uploader_key}",
            disabled=st.session_state.is_processing
        )
        
        if uploaded_folder and model is not None:
            # Render "Analyze" and "Clear" if NOT processing
            if not st.session_state.is_processing:
                btn_col1, btn_col2 = st.columns(2)
                btn_label = "🧠 Analyse Again" if 'processed_batch' in st.session_state and len(st.session_state.processed_batch) > 0 else "🧠 Analyze Folder"
                
                with btn_col1:
                    if st.button(btn_label, type="primary", use_container_width=True):
                        st.session_state.is_processing = True
                        st.rerun()
                with btn_col2:
                    if st.button("🗑️ Clear", type="secondary", use_container_width=True):
                        for key in ['processed_batch', 'summary', 'slider_labels', 'vol_pdf_bytes', 'vol_pdf_name']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state.uploader_key = int(time.time())
                        st.rerun()

            if st.session_state.is_processing:
                
                # POPULATE RIGHT COLUMN WHILE PROCESSING
                with right_ui_placeholder.container():
                    st.info("⏳ Processing volumetric scan... Please wait.")
                    st.markdown("""
                        <div style="display: flex; justify-content: center; align-items: center; height: 300px; border: 1px dashed rgba(255,255,255,0.2); border-radius: 10px; margin-top: 20px;">
                            <h3 style="color: grey;">AI Engine Running...</h3>
                        </div>
                    """, unsafe_allow_html=True)
                
                try:
                    st.session_state.processed_batch = []
                    st.session_state.slider_labels = [] 
                    st.session_state.summary = {"total": 0, "infected": 0, "total_px": 0, "max_px": 0, "worst_slice_idx": 0} 
                    st.session_state.applied_confidence = confidence
                    
                    my_bar = st.progress(0, text="Processing Volumetric Data...")
                    total_files = len(uploaded_folder)
                    
                    for i, file in enumerate(uploaded_folder):
                        if not st.session_state.is_processing:
                            break 

                        img_processed = preprocess_scan(file)
                        
                        if img_processed is not None:
                            img_tensor = torch.from_numpy(img_processed).float().unsqueeze(0).unsqueeze(0)
                            with torch.no_grad():
                                raw_sigmoid_output = torch.sigmoid(model(img_tensor)).squeeze().numpy()
                                prediction = (raw_sigmoid_output > confidence).astype(float)
                            
                            px_count = int(np.sum(prediction))
                            
                            if px_count > 0: 
                                st.session_state.summary["infected"] += 1
                                st.session_state.summary["total_px"] += px_count
                                if px_count > st.session_state.summary["max_px"]:
                                    st.session_state.summary["max_px"] = px_count
                                    st.session_state.summary["worst_slice_idx"] = i
                                st.session_state.slider_labels.append(f"{i+1} 🔴")
                            else:
                                st.session_state.slider_labels.append(f"{i+1} 🟢")
                            
                            st.session_state.processed_batch.append({
                                "filename": file.name,
                                "image": img_processed,
                                "prediction": prediction,
                                "has_tumor": px_count > 0,
                                "px_count": px_count
                            })
                            
                            st.session_state.summary["total"] = len(st.session_state.processed_batch)
                            
                        my_bar.progress((i + 1) / total_files, text=f"Analyzing slice {i+1} of {total_files}")
                    
                    my_bar.empty()
                    
                    # GENERATE NATIVE PDF IMMEDIATELY AFTER PROCESSING
                    if st.session_state.summary["total"] > 0:
                        try:
                            worst_idx = st.session_state.summary["worst_slice_idx"] if st.session_state.summary["worst_slice_idx"] < len(st.session_state.processed_batch) else 0
                            worst_data = st.session_state.processed_batch[worst_idx]
                            
                            risk_level = "HIGH" if st.session_state.summary["infected"] > 0 else "LOW"
                            vol_mm3 = st.session_state.summary["total_px"] * 1.0 
                            diam_mm = (np.sqrt(st.session_state.summary["max_px"] / np.pi) * 2) if st.session_state.summary["max_px"] > 0 else 0
                            
                            # REDUCED FIGSIZE FOR 1-PAGE PDF
                            fig_vol, (ax_vol_left, ax_vol_right) = plt.subplots(1, 2, figsize=(7, 3.5), facecolor='#0B1215')
                            ax_vol_left.imshow(worst_data["image"], cmap='gray'); ax_vol_left.axis('off')
                            ax_vol_right.imshow(worst_data["image"], cmap='gray')
                            m = np.ma.masked_where(worst_data["prediction"] == 0, worst_data["prediction"])
                            ax_vol_right.imshow(m, cmap='winter', alpha=0.8); ax_vol_right.axis('off')
                            fig_vol.tight_layout()
                            
                            temp_p = "temp_vol_export.png"
                            fig_vol.savefig(temp_p, bbox_inches='tight', facecolor='#0B1215')
                            plt.close(fig_vol)
                            
                            unique_patient_id = "VOL-" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                            pdf_bytes = generate_hospital_report(unique_patient_id, vol_mm3, diam_mm, risk_level, temp_p)
                            
                            st.session_state.vol_pdf_bytes = pdf_bytes
                            st.session_state.vol_pdf_name = f"Report_{unique_patient_id}.pdf"
                        except Exception as e:
                            pass 
                        finally:
                            if os.path.exists(temp_p):
                                os.remove(temp_p)

                finally:
                    st.session_state.is_processing = False
                    st.rerun()

            if 'processed_batch' in st.session_state and len(st.session_state.processed_batch) > 0 and not st.session_state.is_processing:
                
                if st.session_state.summary["total"] < len(uploaded_folder):
                    st.warning("⚠️ Analysis was stopped midway. Showing partial results.")

                st.subheader("📊 Volumetric Summary")
                st.write(f"- **Total Slices Scanned:** {st.session_state.summary['total']}")
                st.write(f"- **Slices with Malignancy:** {st.session_state.summary['infected']}")

                st.caption("💾 **Export Volumetric Clinical Report**")
                
                # DIRECT 1-CLICK DOWNLOAD BUTTON
                if 'vol_pdf_bytes' in st.session_state:
                    st.download_button(
                        label="📄 Download PDF Report", 
                        data=st.session_state.vol_pdf_bytes, 
                        file_name=st.session_state.vol_pdf_name, 
                        mime="application/pdf", 
                        use_container_width=True
                    )
                else:
                    st.error("Report generation failed.")

    # RENDER THE RIGHT COLUMN (when NOT processing)
    with right_ui_placeholder.container():
        if not st.session_state.is_processing:
            if not uploaded_folder:
                st.info("💡 **Clinical Tip: 3D Volumetric Math**")
                st.write("To calculate the true anatomical volume of a neoplasm, this system requires the **complete, contiguous series** of 2D CT slices. Uploading an incomplete folder will result in a fragmented 3D reconstruction and an underestimation of the total tumor mass.")
                st.caption("👈 Drag and drop your folder into the uploader on the left to begin.")
                
            elif uploaded_folder and 'processed_batch' in st.session_state and len(st.session_state.processed_batch) > 0:
                top_ctrl_left, top_ctrl_right = st.columns([1.5, 1], gap="small")
                
                with top_ctrl_right:
                    if len(st.session_state.slider_labels) > 1:
                        selected_label = st.select_slider(
                            "Scrub through slices:", 
                            options=st.session_state.slider_labels,
                            value=st.session_state.slider_labels[0],
                            disabled=st.session_state.is_processing
                        )
                        slice_index = int(selected_label.split()[0])
                    elif len(st.session_state.slider_labels) == 1:
                        st.markdown(f"**Viewing Slice:** {st.session_state.slider_labels[0]}")
                        slice_index = 1
                    else:
                        slice_index = 1
                    
                active_data = st.session_state.processed_batch[slice_index - 1]
                indicator_status = "led-red" if active_data['has_tumor'] else "led-green"
                
                with top_ctrl_left:
                    st.subheader(f"Viewing: {active_data['filename']}")
                    if active_data['has_tumor']:
                        st.error(f"⚠️ Tumor detected in this slice (~{active_data['px_count']} px)")
                    else:
                        st.success("✅ No significant nodules detected")
                        
                fig_batch, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='#0B1215')
                ax_b1.imshow(active_data['image'], cmap='gray'); ax_b1.axis('off'); ax_b1.set_title("Input Slice", color="white")
                
                ax_b2.imshow(active_data['image'], cmap='gray')
                masked_pred_batch = np.ma.masked_where(active_data['prediction'] == 0, active_data['prediction'])
                ax_b2.imshow(masked_pred_batch, cmap='winter', alpha=0.8); ax_b2.axis('off'); ax_b2.set_title("AI Mask", color="white")
                
                fig_batch.tight_layout()
                st.pyplot(fig_batch, width="stretch")
                plt.close(fig_batch)

led_placeholder.markdown(f"""
    <div style="display: flex; justify-content: flex-end; align-items: center; gap: 12px; margin-top: 25px; padding-right: 20px;">
        <div class="led-indicator {indicator_status}"></div>
    </div>
""", unsafe_allow_html=True)

st.divider()
st.warning("⚠️ **Medical AI Disclaimer:** This is an academic Capstone prototype intended for research and educational demonstrations only. It is not an FDA-approved medical device and must never be used as a substitute for professional radiological diagnosis or clinical decision-making.")
st.markdown("<br><p style='text-align: center; color: grey;'>Official Capstone Project Submission | Ravi | NIT Jamshedpur</p>", unsafe_allow_html=True)