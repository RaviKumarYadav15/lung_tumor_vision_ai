import os
import numpy as np

def inspect_dataset():
    train_dir = 'data/raw/train/'
    
    print("🕵️‍♂️ Starting Dataset Diagnostic Scan...")
    print("=" * 60)
    
    # Check if directory exists
    if not os.path.exists(train_dir):
        print(f"❌ Error: Cannot find directory '{train_dir}'")
        return

    patient_folders = [f.name for f in os.scandir(train_dir) if f.is_dir()]
    patient_folders.sort(key=int) # Sort folders numerically (0, 1, 2...)
    
    normalized_count = 0
    hounsfield_count = 0

    for patient_id in patient_folders:
        data_dir = os.path.join(train_dir, patient_id, 'data')
        
        if not os.path.exists(data_dir):
            continue
            
        # Get all numpy files
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        if len(files) == 0:
            continue
            
        # Just grab the very first slice for this patient
        sample_file = files[0]
        file_path = os.path.join(data_dir, sample_file)
        
        # Load the data
        img = np.load(file_path).astype(np.float32)
        min_val = img.min()
        max_val = img.max()
        
        # Determine Data Type
        if min_val >= -10.0 and max_val <= 255.0:
            data_type = "⚠️ ALREADY NORMALIZED (or generic image)"
            normalized_count += 1
        elif min_val <= -800:
            data_type = "✅ RAW HOUNSFIELD UNITS (HU)"
            hounsfield_count += 1
        else:
            data_type = "❓ UNKNOWN FORMAT"
            
        print(f"Patient {patient_id:>2} | Slice {sample_file:>9} | Min: {min_val:>8.2f} | Max: {max_val:>8.2f} | {data_type}")

    print("=" * 60)
    print("📊 FINAL DIAGNOSTIC REPORT:")
    print(f"Total Patients Scanned: {len(patient_folders)}")
    print(f"Patients with Raw HU (-1000): {hounsfield_count}")
    print(f"Patients Already Normalized:  {normalized_count}")
    
    if normalized_count > 0 and hounsfield_count > 0:
        print("\n🚨 CRITICAL WARNING: Your dataset is mixed! Some patients are HU, some are normalized.")
    elif normalized_count > 0:
        print("\n💡 CONCLUSION: Your training data is ALREADY NORMALIZED. You MUST remove the HU math before retraining!")
    elif hounsfield_count > 0:
        print("\n💡 CONCLUSION: Your training data is in raw HU. The HU math formula is correct and necessary.")

if __name__ == "__main__":
    inspect_dataset()