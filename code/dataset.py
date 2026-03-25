# --- IMPORTS ---
import os
import cv2
import numpy as np
import pandas as pd
import torch
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
from settings import IMG_SIZE, BATCH_SIZE

# --- DATASET CLASS ---
class RSNADataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=512, mode='train'):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.mode = mode
        self.patient_ids = self.df['patientId'].unique()
        
        # Data Augmentation Setup
        if mode == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-10, 10), p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                ToTensorV2() 
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        img_path = os.path.join(self.img_dir, f"{patient_id}.dcm")
        
        try:
            dcm_data = pydicom.dcmread(img_path)
            img = dcm_data.pixel_array
        except:
            img = np.zeros((self.img_size, self.img_size))
            
        if img.shape[0] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
            
        # Normalization [0, 1] FLOAT32
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        records = self.df[self.df['patientId'] == patient_id]
        target = records['Target'].values[0] if 'Target' in records.columns else 0
        
        if target == 1:
            row = records.iloc[0]
            scale = self.img_size / 1024.0 
            box = [row['x']*scale, row['y']*scale, row['width']*scale, row['height']*scale]
            class_label = 1
        else:
            box = [0.0, 0.0, 1.0, 1.0] 
            class_label = 0
            
        try:
            transformed = self.transform(image=img, bboxes=[box], class_labels=[class_label])
            img_tensor = transformed['image'] 
            aug_box = np.array(transformed['bboxes'][0]) if len(transformed['bboxes']) > 0 else np.array([0.0, 0.0, 0.0, 0.0])
            if len(transformed['bboxes']) == 0: target = 0
        except:
            img_tensor = torch.tensor(img).unsqueeze(0)
            aug_box = np.array([0.0, 0.0, 0.0, 0.0])
            target = 0
            
        final_box = aug_box / self.img_size if target == 1 else np.array([0.0, 0.0, 0.0, 0.0])
        return img_tensor, torch.tensor(target, dtype=torch.float32), torch.tensor(final_box, dtype=torch.float32)

# --- DATALOADER BUILDER ---
def get_dataloaders(csv_path, img_path):
    print(f"[INFO] Building Dataset {IMG_SIZE}x{IMG_SIZE}...")
    full_ds = RSNADataset(csv_path, img_path, img_size=IMG_SIZE, mode='train')
    indices = list(range(len(full_ds)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(RSNADataset(csv_path, img_path, img_size=IMG_SIZE, mode='val'), val_idx)
    
    print("[INFO] Computing class weights for balanced sampling...")
    train_targets = np.array([full_ds.df.iloc[i]['Target'] for i in train_idx])
    class_counts = np.bincount(train_targets)
    
    weights = 1. / class_counts
    samples_weight = torch.from_numpy(weights[train_targets]).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader