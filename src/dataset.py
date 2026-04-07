import os
import torch
import numpy as np
from torch.utils.data import Dataset

class LungSegmentationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.mask_paths = []

        patient_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]

        for patient_path in patient_folders:
            data_dir = os.path.join(patient_path, 'data')
            mask_dir = os.path.join(patient_path, 'masks')

            if os.path.exists(data_dir) and os.path.exists(mask_dir):
                for filename in os.listdir(data_dir):
                    if filename.endswith('.npy'):
                        self.image_paths.append(os.path.join(data_dir, filename))
                        self.mask_paths.append(os.path.join(mask_dir, filename))
                        
        print(f"📊 Dataset Loaded: Found {len(self.image_paths)} total CT slices.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load Data
        raw_img = np.load(self.image_paths[idx]).astype(np.float32)
        true_mask = np.load(self.mask_paths[idx])

        # 2. NO HOUNSFIELD MATH NEEDED! 
        # Just ensure the mask is strictly binary (0 and 1)
        true_mask = (true_mask > 0).astype(np.float32)

        # 3. Convert to Tensors (Add the Channel Dimension)
        img_tensor = torch.from_numpy(raw_img).unsqueeze(0)  
        mask_tensor = torch.from_numpy(true_mask).unsqueeze(0)     

        return img_tensor, mask_tensor