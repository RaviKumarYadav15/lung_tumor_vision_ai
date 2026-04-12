# modules/pdf_generator.py
from fpdf import FPDF
import datetime

def generate_hospital_report(patient_id, tumor_area, tumor_diameter, risk_level, image_path):
    """
    Generates a professional clinical PDF report.
    Returns the PDF as a byte string so Streamlit can download it.
    """
    class PDF(FPDF):
        def header(self):
            # Hospital Letterhead
            self.set_font('helvetica', 'B', 15)
            self.set_text_color(2, 128, 144) # Custom Teal Color
            self.cell(0, 10, 'LUNGVISION AI - CLINICAL DIAGNOSTIC REPORT', border=False, ln=True, align='C')
            self.set_font('helvetica', 'I', 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, 'Automated Pulmonary Nodule Segmentation Pipeline', border=False, ln=True, align='C')
            self.line(10, 30, 200, 30)
            self.ln(10)

        def footer(self):
            # Medical Disclaimer at the bottom
            self.set_y(-15)
            self.set_font('helvetica', 'I', 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, 'DISCLAIMER: AI-generated report. Must be reviewed by a certified radiologist prior to clinical action.', align='C')

    # 1. Initialize PDF
    pdf = PDF()
    pdf.add_page()
    
    # 2. Add Patient Metadata
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    pdf.cell(100, 8, f"Patient ID: {patient_id}", ln=False)
    pdf.cell(0, 8, f"Date Generated: {current_date}", ln=True)
    pdf.cell(100, 8, f"Scan Type: Computed Tomography (CT)", ln=True)
    pdf.ln(5)

    # 3. Add AI Radiomics Metrics
    pdf.set_font('helvetica', 'B', 14)
    
    # FIX: Changed 'self' to 'pdf' here
    if risk_level == "HIGH":
        pdf.set_text_color(200, 0, 0) # Red for high risk
    else: 
        pdf.set_text_color(0, 150, 0) # Green for low risk
        
    pdf.cell(0, 10, f"MALIGNANCY RISK: {risk_level}", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 8, f"Estimated Surface Area: {tumor_area:.2f} mm2", ln=True)
    pdf.cell(0, 8, f"Estimated Max Diameter: {tumor_diameter:.2f} mm", ln=True)
    pdf.ln(10)

    # 4. Insert the Visual Scan
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, "AI Segmentation Analysis:", ln=True)
    
    # Try to insert the image
    try:
        pdf.image(image_path, x=15, w=180)
    except Exception as e:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"Error loading scan image: {e}", ln=True)

    # 5. Output as bytes for Streamlit compatibility
    return bytes(pdf.output())