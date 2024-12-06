# Specific libraries for attenuation UNet --> use pytorch: open-source ML library
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

        self.images = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
        )
        self.labels = sorted(
            [f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
        )

        self.paired_files = [
            (img, lbl) for img, lbl in zip(self.images, self.labels)
            if os.path.splitext(img)[0] == os.path.splitext(lbl)[0]
        ]

        if not self.paired_files:
            raise ValueError(
                f"No matching image-label pairs found. "
                f"Check the filenames in:\nImage Dir: {image_dir}\nLabel Dir: {label_dir}"
            )

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

        return image, label.float()
        
transform = A.Compose([
    A.Resize(256, 256), 
    A.HorizontalFlip(p=0.5), 
    A.RandomRotate90(p=0.5), 
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(), 
])

print("Please enter the paths for your image and label directories:")
train_image_dir = input("Path to train image directory (default: './Data/train/images/'): ") or './Data/train/images/'
train_label_dir = input("Path to train label directory (default: './Data/train/labels/'): ") or './Data/train/labels/'
test_image_dir = input("Path to test image directory (default: './Data/test/images/'): ") or './Data/test/images/'
test_label_dir = input("Path to test label directory (default: './Data/test/labels/'): ") or './Data/test/labels/'
