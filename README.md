# LungVision AI 🫁

An end-to-end Medical Imaging pipeline and deployment dashboard using a **U-Net** deep learning architecture to segment malignant lung tumors from CT scans.

## 📌 Project Overview
This project uses Deep Learning to assist radiologists by automatically identifying, masking, and volumetrically estimating malignant nodules in lung CT scans. It features an **Adaptive Digital Image Processing (DIP) Pipeline** to handle raw Hounsfield Units (HU), utilizes the **Dice Coefficient** to conquer class imbalance in medical data, and includes a state-managed, clinical Streamlit dashboard for real-time triage.

**🏆 Current Baseline Performance:** Achieved a Peak Pixel Accuracy of **97.10%** and a Validation Dice Score of **53.43%** on unseen patient data.

### ✨ Key Clinical Features
* **Dual Diagnostic Modes:** Supports rapid Single Slice triage or full 3D Volumetric Folder analysis.
* **Explainable AI (XAI):** Features interactive Confidence Heatmaps and dynamic AI Mask Opacity sliders.
* **Worst-Case Isolation:** Automatically scans hundreds of slices to identify and present the single largest tumor cross-section.
* **Zero-Retention Reporting:** Generates 1-click clinical PDF reports (with dynamic 2D/3D math) and instantly destroys temporary patient data to maintain strict privacy.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch
* **Data Processing:** NumPy, pydicom
* **Visualization:** Matplotlib, tqdm (Progress tracking)
* **Reporting:** fpdf2
* **UI & Deployment:** Streamlit

**👁️ Preview of App**
----
![Screenshot](screenshot_UI.png)

## 📁 Project Structure
```text
lung-tumor-vision-ai/
├── data/               # Raw and Processed CT slices (Ignored in Git)
├── models/             # Trained .pth model weights (Ignored in Git)
├── notebooks/          # Data exploration & DIP experiments
├── modules/            # UI Support Modules
│   ├── pdf_generator.py   # Dynamic clinical report synthesizer
│   └── xai_visualizer.py  # Confidence heatmap generation
├── src/                # Modular Core Engine
│   ├── dataset.py      # Custom PyTorch Dataset & Normalization
│   ├── unet.py         # U-Net Model Architecture
│   ├── metrics.py      # Dice Loss & IoU implementation
│   ├── train.py        # Lightweight local training tester
│   ├── train_colab.py  # Heavy cloud training script
│   ├── evaluate.py     # Patient-level validation tester
│   └── predict.py      # Inference & Visualization script
├── app.py              # 🚀 Streamlit Clinical Dashboard
└── requirements.txt    # Optimized deployment dependencies

🚀 Getting Started
1. Clone & Setup Environment

```Bash
git clone [https://github.com/RaviKumarYadav15/lung-tumor-vision-ai.git](https://github.com/RaviKumarYadav15/lung-tumor-vision-ai.git)
```
```bash
cd lung-tumor-vision-ai
```
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```
(Note: You will need to add the data/raw/ and models/ folders locally, as heavy files are omitted from version control via .gitignore.)

2. Training the Model
To test the pipeline locally on a CPU/small GPU:

```Bash
python src/train.py
```
To train the full model on a cloud GPU (e.g., Google Colab), run:

```Bash
python src/train_colab.py
```

3. Evaluating & Visualizing
To evaluate the model's performance on an entire unseen validation patient:

```Bash
python src/evaluate.py
```
To generate matplotlib visual comparisons (Patient CT vs. True Mask vs. AI Prediction):

```Bash
python src/predict.py
```
4. Launching the Clinical Dashboard 🩺
To start the interactive Streamlit web application:

``` bash
streamlit run app.py
```
📊 Methodology (DIP & AI)
Adaptive Data Pipeline: Intelligently detects whether incoming data is raw Hounsfield Units (-1000 to 400 HU) or pre-normalized [0, 1] arrays, preventing visualization blackouts during inference.

Segmentation: A 4-level U-Net (MICCAI Standard) captures deep spatial features via skip-connections.

Evaluation: Uses Dice Loss to prioritize exact geometric overlap with the tumor rather than background accuracy, preventing the AI from lazily guessing "healthy tissue" on imbalanced scans.

Volumetric Aggregation: Accumulates 2D slice predictions across the Z-axis to estimate 3D tumor volume mathematically.

⚖️ License & Credits
This project was developed for educational purposes as part of a 6th-semester Capstone project at NIT Jamshedpur.

Author: Ravi Kumar Yadav

