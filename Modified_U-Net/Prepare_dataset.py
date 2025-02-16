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
import cv2
class BreastCancerDataset(Dataset):
    # def __init__(self, image_dir, label_dir, transform=None, save_dir=None):
    #     self.image_dir = image_dir
    #     self.label_dir = label_dir
    #     self.transform = transform
    #     self.save_dir = save_dir

    #     image_files = sorted(os.listdir(image_dir))
    #     label_files = sorted(os.listdir(label_dir))
    #     self.paired_files = [
    #         (img, lbl) for img, lbl in zip(image_files, label_files)
    #         if os.path.splitext(img)[0] == os.path.splitext(lbl)[0]
    #     ]
    #     # if not self.paired_files:
    #     #     raise ValueError("No matching image-label pairs found.")
        
    #     # create save img
    #     if self.save_dir:
    #         os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
    #         os.makedirs(os.path.join(self.save_dir, "labels"), exist_ok=True)
    def __init__(self, image_dir, label_dir, transform=None, save_dir=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.save_dir = save_dir

        # 获取文件名（去除扩展名），并存入字典，保留完整文件路径
        image_dict = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(image_dir))}
        label_dict = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(label_dir))}

        # 仅匹配两边都存在的文件
        common_keys = set(image_dict.keys()) & set(label_dict.keys())

        self.paired_files = [(image_dict[k], label_dict[k]) for k in sorted(common_keys)]

        if not self.paired_files:
            raise ValueError("No matching image-label pairs found.")

        # 创建保存目录
        if self.save_dir:
            os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "labels"), exist_ok=True)


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

         # save images after augmentation
        if self.save_dir:
            save_image_path = os.path.join(self.save_dir, "images", img_name)
            save_label_path = os.path.join(self.save_dir, "labels", lbl_name)

            # transfer back to PIL format and save
            Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).convert("L").save(save_image_path)
            Image.fromarray((label.numpy() * 255).astype(np.uint8)).convert("L").save(save_label_path)

        return image, label, img_name

# Data Augmentation 数据增强 
transform = A.Compose([
    A.Resize(256, 256), 
    A.HorizontalFlip(p=0.5), 
    A.RandomRotate90(p=0.5), 
    A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.5), 
    #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    A.ToFloat(),
    ToTensorV2(), 
])

# only do normalization for test data
test_transform = A.Compose([
    A.Resize(256, 256),
    # A.LongestMaxSize(max_size=256),  # 先等比例缩放，使最长边变成 256
    # A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),  # 用0填充
    #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToFloat(),
    ToTensorV2(),
])

print("Please decide train or test")
is_train = input("[train/test](default: test):") or 'test'
if is_train == 'train':
    is_train = True
elif is_train == 'test':
    is_train = False
else:
    print("Please enter the correct input")
    exit()


print("Please enter the paths for your image and label directories:")
dataset_dir = input("Path to dataset parent directory (default: '/root/Dataset/BUS_adapter_demo'): ") or '/root/Dataset/BUS_adapter_demo'

if is_train:
    train_image_dir = os.path.join(dataset_dir, 'train', 'images')
    train_label_dir = os.path.join(dataset_dir, 'train', 'labels')
    train_augu_dir = os.path.join(dataset_dir, 'augmented', 'train')

    train_dataset = BreastCancerDataset(train_image_dir, train_label_dir, transform=transform, save_dir=None)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) 

else:
    test_image_dir = os.path.join(dataset_dir, 'test', 'images')
    test_label_dir = os.path.join(dataset_dir, 'test', 'labels')
    test_augu_dir = os.path.join(dataset_dir, 'augmented', 'test')

    test_dataset = BreastCancerDataset(test_image_dir, test_label_dir, transform=test_transform, save_dir=None)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) 


if __name__ == "__main__":

    # When directly run this file, it will save all the data to sava_dir
    if is_train:
        train_dataset = BreastCancerDataset(train_image_dir, train_label_dir, transform=transform, save_dir=train_augu_dir)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        for images, labels in train_loader:
            print("Batch of images shape:", images.shape)  # 输出形状，例如 [4, 3, 256, 256]（4 张图像）
            print("Batch of labels shape:", labels.shape)  # 输出标签形状，例如 [4, 1]（4 个标签）
    else:
        test_dataset = BreastCancerDataset(test_image_dir, test_label_dir, transform=test_transform, save_dir=test_augu_dir)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
        for images, labels in test_loader:
            print("Batch of images shape:", images.shape)  # 输出形状，例如 [4, 3, 256, 256]（4 张图像）
            print("Batch of labels shape:", labels.shape)  # 输出标签形状，例如 [4, 1]（4 个标签）

    