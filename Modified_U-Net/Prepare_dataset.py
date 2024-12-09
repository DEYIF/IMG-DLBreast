from collections import OrderedDict
import torch
import torch.nn as nn 
from torch.autograd import Variable 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

class BreastCancerDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))
        self.paired_files = [
            (img, lbl) for img, lbl in zip(image_files, label_files)
            if os.path.splitext(img)[0] == os.path.splitext(lbl)[0]
        ]
        if not self.paired_files:
            raise ValueError("No matching image-label pairs found.")

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.paired_files):
            raise IndexError(f"Index {idx} is out of range. Total items: {len(self.paired_files)}")
        
        img_name, lbl_name = self.paired_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, lbl_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path).convert("L"))

        label = label / 255.0 

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        label = torch.tensor(label, dtype=torch.float32)  

        return image, label

transform = A.Compose([
    A.Resize(256, 256), 
    A.HorizontalFlip(p=0.5), 
    A.RandomRotate90(p=0.5), 
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(), 
])

test_image_dir = './BUS_dataset/BUS_all_dataset_resize/test/images/'
test_label_dir = './BUS_dataset/BUS_all_dataset_resize/test/labels/'
train_image_dir = './BUS_dataset/BUS_all_dataset_resize/train/images/'
train_label_dir = './BUS_dataset/BUS_all_dataset_resize/train/labels/'
