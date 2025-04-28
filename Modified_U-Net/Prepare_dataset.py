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
    def __init__(self, image_dir, label_dir, transform=None, save_dir=None, is_second_unet=False, mask_dir=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir  # SAM输出的掩码目录
        self.transform = transform
        self.save_dir = save_dir
        self.is_second_unet = is_second_unet

        # 获取文件名（去除扩展名），并存入字典，保留完整文件路径
        image_dict = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(image_dir))}
        label_dict = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(label_dir))}
        
        # 如果是第二道UNet且提供了mask_dir，则加载mask文件
        if self.is_second_unet and self.mask_dir and os.path.exists(self.mask_dir):
            mask_dict = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(mask_dir))}
            # 仅匹配三者都存在的文件
            common_keys = set(image_dict.keys()) & set(label_dict.keys()) & set(mask_dict.keys())
            self.paired_files = [(image_dict[k], label_dict[k], mask_dict[k]) for k in sorted(common_keys)]
        else:
            # 仅匹配两者都存在的文件
            common_keys = set(image_dict.keys()) & set(label_dict.keys())
            self.paired_files = [(image_dict[k], label_dict[k], None) for k in sorted(common_keys)]

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
        
        file_items = self.paired_files[idx]
        img_name, lbl_name = file_items[0], file_items[1]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, lbl_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path).convert("L"))

        label = label / 255.0 
        
        # 如果是第二道UNet且有mask文件
        mask = None
        if self.is_second_unet and self.mask_dir and len(file_items) > 2 and file_items[2]:
            mask_name = file_items[2]
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
                mask = mask / 255.0  # 标准化到[0,1]范围

        # 应用数据增强 - 需要同时变换图像、标签和掩码
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=label, mask1=mask)
                image = augmented['image']
                label = augmented['mask']
                mask = augmented['mask1']
            else:
                augmented = self.transform(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask']

        # 转换标签和掩码为torch tensor
        label = torch.tensor(label, dtype=torch.float32)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float32)

        # 针对第二道UNet，将图像和SAM掩码拼接作为输入
        if self.is_second_unet and mask is not None:
            # 确保mask是单通道，[H, W] -> [1, H, W]
            if len(mask.shape) == 2:
                mask_channel = mask.unsqueeze(0)
            else:
                mask_channel = mask
                
            # 将原图和SAM掩码拼接作为第二道UNet的输入
            # image: [3, H, W], mask_channel: [1, H, W] -> combined: [4, H, W]
            combined_input = torch.cat([image, mask_channel], dim=0)
            input_data = combined_input
        else:
            input_data = image

        # save images after augmentation
        if self.save_dir:
            save_image_path = os.path.join(self.save_dir, "images", img_name)
            save_label_path = os.path.join(self.save_dir, "labels", lbl_name)

            # transfer back to PIL format and save
            if self.is_second_unet and mask is not None:
                # 保存的是拼接后的图像，这里只保存RGB部分用于可视化
                Image.fromarray((input_data[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(save_image_path)
            else:
                Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(save_image_path)
                
            Image.fromarray((label.numpy() * 255).astype(np.uint8)).save(save_label_path)

        return input_data, label, img_name

class MixSegDataset(Dataset):
    def __init__(self, image_dir, label_dir, test_prefixes=None, is_train=True, transform=None, save_dir=None, is_second_unet=False, mask_dir=None):
        """
        MixSegDataset类用于处理BUS_Mix_Seg_resize_512数据集
        
        Args:
            image_dir: 图像目录路径
            label_dir: 标签目录路径
            test_prefixes: 用于测试的文件前缀列表，例如['BUSI', 'BUSC', 'STU']
            is_train: 如果为True，使用非test_prefixes的文件作为训练集；如果为False，使用test_prefixes的文件作为测试集
            transform: 数据增强转换
            save_dir: 增强后数据的保存目录
            is_second_unet: 是否为第二道UNet（如果是，则输入数据为image和mask的拼接）
            mask_dir: SAM输出的掩码目录路径
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.save_dir = save_dir
        self.test_prefixes = test_prefixes if test_prefixes else []
        self.is_train = is_train
        self.is_second_unet = is_second_unet
        
        # 创建保存目录
        if self.save_dir:
            os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "labels"), exist_ok=True)
        
        # 获取所有图像和标签文件
        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))
        
        # 创建文件名到文件的映射
        image_dict = {f: f for f in image_files}
        label_dict = {f: f for f in label_files}
        
        # 如果是第二道UNet且提供了mask_dir，则加载mask文件
        if self.is_second_unet and self.mask_dir and os.path.exists(self.mask_dir):
            mask_files = sorted(os.listdir(mask_dir))
            mask_dict = {f: f for f in mask_files}
            common_files = set(image_dict.keys()) & set(label_dict.keys()) & set(mask_dict.keys())
        else:
            mask_dict = {}
            common_files = set(image_dict.keys()) & set(label_dict.keys())
        
        # 根据前缀筛选训练/测试文件
        self.paired_files = []
        for filename in sorted(common_files):
            # 检查文件是否以测试前缀开头
            is_test_file = any(filename.startswith(prefix) for prefix in self.test_prefixes)
            
            # 如果是训练模式且不是测试文件，或者是测试模式且是测试文件，则添加到数据集
            if (self.is_train and not is_test_file) or (not self.is_train and is_test_file):
                if filename in mask_dict:
                    self.paired_files.append((image_dict[filename], label_dict[filename], mask_dict[filename]))
                else:
                    self.paired_files.append((image_dict[filename], label_dict[filename], None))
        
        if not self.paired_files:
            mode = "训练" if self.is_train else "测试"
            prefixes_str = ", ".join(self.test_prefixes) if self.test_prefixes else "空"
            raise ValueError(f"没有找到匹配的{mode}图像-标签对。测试前缀: {prefixes_str}")
        
        print(f"{'训练' if self.is_train else '测试'}集大小: {len(self.paired_files)}个文件")

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.paired_files):
            raise IndexError(f"索引{idx}超出范围。总项目数: {len(self.paired_files)}")
        
        file_items = self.paired_files[idx]
        img_name, lbl_name = file_items[0], file_items[1]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, lbl_name)

        # 读取图像和标签
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path).convert("L"))

        # 标准化标签到[0,1]范围
        label = label / 255.0 
        
        # 如果是第二道UNet且有mask文件
        mask = None
        if self.is_second_unet and self.mask_dir and len(file_items) > 2 and file_items[2]:
            mask_name = file_items[2]
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
                mask = mask / 255.0  # 标准化到[0,1]范围

        # 应用数据增强 - 需要同时变换图像、标签和掩码
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=label, mask1=mask)
                image = augmented['image']
                label = augmented['mask']
                mask = augmented['mask1']
            else:
                augmented = self.transform(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask']

        # 转换标签和掩码为torch tensor
        label = torch.tensor(label, dtype=torch.float32)  
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float32)

        # 针对第二道UNet，将图像和SAM掩码拼接作为输入
        if self.is_second_unet and mask is not None:
            # 确保mask是单通道，[H, W] -> [1, H, W]
            if len(mask.shape) == 2:
                mask_channel = mask.unsqueeze(0)
            else:
                mask_channel = mask
                
            # 将原图和SAM掩码拼接作为第二道UNet的输入
            # image: [3, H, W], mask_channel: [1, H, W] -> combined: [4, H, W]
            combined_input = torch.cat([image, mask_channel], dim=0)
            input_data = combined_input
        else:
            input_data = image

        # 保存增强后的图像（如果需要）
        if self.save_dir:
            save_image_path = os.path.join(self.save_dir, "images", img_name)
            save_label_path = os.path.join(self.save_dir, "labels", lbl_name)

            # 转换回PIL格式并保存
            if self.is_second_unet and mask is not None:
                # 保存的是拼接后的图像，这里只保存RGB部分用于可视化
                Image.fromarray((input_data[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(save_image_path)
            else:
                if isinstance(image, torch.Tensor):
                    Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(save_image_path)
                else:
                    Image.fromarray(image).save(save_image_path)
            
            Image.fromarray((label.numpy() * 255).astype(np.uint8)).save(save_label_path)

        return input_data, label, img_name

# 修改Albumentations转换以支持额外的掩码
transform = A.Compose([
    A.Resize(256, 256), 
    A.HorizontalFlip(p=0.5), 
    A.RandomRotate90(p=0.5), 
    A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.5), 
    A.ToFloat(),
    ToTensorV2(), 
], additional_targets={'mask1': 'mask'})  # 添加额外的mask目标

# only do normalization for test data
test_transform = A.Compose([
    A.Resize(256, 256),
    A.ToFloat(),
    ToTensorV2(),
], additional_targets={'mask1': 'mask'})  # 添加额外的mask目标

print("请选择UNet模型阶段")
unet_stage = input("UNet阶段 [first/second](默认: first): ") or 'first'
is_second_unet = unet_stage.lower() == 'second'

if is_second_unet:
    print("使用第二道UNet")
    mask_input_type = input("请选择mask输入来源 [label/sam](默认: label): ") or 'label'
    if mask_input_type == 'sam':
        print("将使用SAM生成的掩码作为额外输入")
    else:
        print("将使用第一道UNet的输出作为额外输入")
else:
    print("使用第一道UNet，输入仅为原始图像")
    mask_input_type = 'none'

print("请选择数据集类型")
dataset_type = input("数据集类型 [standard/mixseg](默认: standard): ") or 'standard'

if dataset_type == 'standard':
    print("使用标准数据集类型")
    print("请决定训练或测试")
    is_train = input("[train/test](默认: test):") or 'test'
    if is_train == 'train':
        is_train = True
    elif is_train == 'test':
        is_train = False
    else:
        print("请输入正确的选项")
        exit()

    print("请输入图像和标签目录路径:")
    dataset_dir = input("数据集父目录路径 (默认: '/root/Dataset/BUS_adapter_demo'): ") or '/root/Dataset/BUS_adapter_demo'

    if is_train:
        train_image_dir = os.path.join(dataset_dir, 'train', 'images')
        train_label_dir = os.path.join(dataset_dir, 'train', 'labels')
        train_augu_dir = os.path.join(dataset_dir, 'augmented', 'train')
        
        # 如果是第二道UNet且使用SAM掩码，设置mask_dir
        train_mask_dir = None
        if is_second_unet and mask_input_type == 'sam':
            train_mask_dir = os.path.join(dataset_dir, 'train', 'masks')
            if not os.path.exists(train_mask_dir):
                print(f"警告：掩码目录 {train_mask_dir} 不存在，将使用标签作为输入")
                train_mask_dir = None

        train_dataset = BreastCancerDataset(train_image_dir, train_label_dir, transform=transform, 
                                           save_dir=None, is_second_unet=is_second_unet, mask_dir=train_mask_dir)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) 

    else:
        test_image_dir = os.path.join(dataset_dir, 'test', 'images')
        test_label_dir = os.path.join(dataset_dir, 'test', 'labels')
        test_augu_dir = os.path.join(dataset_dir, 'augmented', 'test')

        # 如果是第二道UNet且使用SAM掩码，设置mask_dir
        test_mask_dir = None
        if is_second_unet and mask_input_type == 'sam':
            test_mask_dir = os.path.join(dataset_dir, 'test', 'masks')
            if not os.path.exists(test_mask_dir):
                print(f"警告：掩码目录 {test_mask_dir} 不存在，将使用标签作为输入")
                test_mask_dir = None

        test_dataset = BreastCancerDataset(test_image_dir, test_label_dir, transform=test_transform, 
                                          save_dir=None, is_second_unet=is_second_unet, mask_dir=test_mask_dir)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) 

elif dataset_type == 'mixseg':
    print("使用混合分割数据集类型")
    print("请决定训练或测试")
    is_train = input("[train/test](默认: test):") or 'test'
    if is_train == 'train':
        is_train = True
    elif is_train == 'test':
        is_train = False
    else:
        print("请输入正确的选项")
        exit()
    
    print("请输入数据集路径:")
    dataset_dir = input("数据集父目录路径 (默认: '/root/Dataset/BUS_Mix_Seg_resize_512'): ") or '/root/Dataset/BUS_Mix_Seg_resize_512'
    
    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')
    augu_dir = os.path.join(dataset_dir, 'augmented', 'train' if is_train else 'test')
    
    # 如果是第二道UNet且使用SAM掩码，设置mask_dir
    mask_dir = None
    if is_second_unet and mask_input_type == 'sam':
        mask_dir = os.path.join(dataset_dir, 'masks')
        if not os.path.exists(mask_dir):
            print(f"警告：掩码目录 {mask_dir} 不存在，将使用标签作为输入")
            mask_dir = None
    
    if is_train:
        # 对于训练模式，我们需要知道测试前缀以便排除这些文件
        test_prefixes_input = input("请输入测试集前缀(用逗号分隔，例如 BUSI,BUSC,STU): ")
        test_prefixes = [prefix.strip() for prefix in test_prefixes_input.split(',') if prefix.strip()]
        
        train_dataset = MixSegDataset(image_dir, label_dir, test_prefixes=test_prefixes, 
                                     is_train=True, transform=transform, save_dir=None,
                                     is_second_unet=is_second_unet, mask_dir=mask_dir)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    else:
        # 对于测试模式，我们需要指定哪些前缀用于测试
        test_prefixes_input = input("请输入测试集前缀(用逗号分隔，例如 BUSI,BUSC,STU): ")
        test_prefixes = [prefix.strip() for prefix in test_prefixes_input.split(',') if prefix.strip()]
        
        test_dataset = MixSegDataset(image_dir, label_dir, test_prefixes=test_prefixes, 
                                    is_train=False, transform=test_transform, save_dir=None,
                                    is_second_unet=is_second_unet, mask_dir=mask_dir)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
else:
    print("不支持的数据集类型，请选择 standard 或 mixseg")
    exit()


if __name__ == "__main__":

    # 直接运行此文件时，将所有数据保存到save_dir
    if dataset_type == 'standard':
        if is_train:
            # 如果是第二道UNet且使用SAM掩码，设置mask_dir
            train_mask_dir = None
            if is_second_unet and mask_input_type == 'sam':
                train_mask_dir = os.path.join(dataset_dir, 'train', 'masks')
                if not os.path.exists(train_mask_dir):
                    print(f"警告：掩码目录 {train_mask_dir} 不存在，将使用标签作为输入")
                    train_mask_dir = None
                    
            train_dataset = BreastCancerDataset(train_image_dir, train_label_dir, transform=transform, 
                                               save_dir=train_augu_dir, is_second_unet=is_second_unet,
                                               mask_dir=train_mask_dir)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            for inputs, labels, _ in train_loader:
                if is_second_unet:
                    print("批次输入形状:", inputs.shape)  # 第二道UNet: [B, 4, H, W]（4通道 = 3RGB + 1Mask）
                else:
                    print("批次图像形状:", inputs.shape)  # 第一道UNet: [B, 3, H, W]
                print("批次标签形状:", labels.shape)  # [B, H, W]
        else:
            # 如果是第二道UNet且使用SAM掩码，设置mask_dir
            test_mask_dir = None
            if is_second_unet and mask_input_type == 'sam':
                test_mask_dir = os.path.join(dataset_dir, 'test', 'masks')
                if not os.path.exists(test_mask_dir):
                    print(f"警告：掩码目录 {test_mask_dir} 不存在，将使用标签作为输入")
                    test_mask_dir = None
                    
            test_dataset = BreastCancerDataset(test_image_dir, test_label_dir, transform=test_transform, 
                                              save_dir=test_augu_dir, is_second_unet=is_second_unet,
                                              mask_dir=test_mask_dir)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
            for inputs, labels, _ in test_loader:
                if is_second_unet:
                    print("批次输入形状:", inputs.shape)  # 第二道UNet: [B, 4, H, W]
                else:
                    print("批次图像形状:", inputs.shape)  # 第一道UNet: [B, 3, H, W]
                print("批次标签形状:", labels.shape)  # [B, H, W]
    elif dataset_type == 'mixseg':
        # 如果是第二道UNet且使用SAM掩码，设置mask_dir
        mask_dir = None
        if is_second_unet and mask_input_type == 'sam':
            mask_dir = os.path.join(dataset_dir, 'masks')
            if not os.path.exists(mask_dir):
                print(f"警告：掩码目录 {mask_dir} 不存在，将使用标签作为输入")
                mask_dir = None
                
        if is_train:
            # 创建用于保存的训练数据集
            train_dataset = MixSegDataset(image_dir, label_dir, test_prefixes=test_prefixes, 
                                         is_train=True, transform=transform, save_dir=augu_dir,
                                         is_second_unet=is_second_unet, mask_dir=mask_dir)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            for inputs, labels, _ in train_loader:
                if is_second_unet:
                    print("批次输入形状:", inputs.shape)  # 第二道UNet: [B, 4, H, W]
                else:
                    print("批次图像形状:", inputs.shape)  # 第一道UNet: [B, 3, H, W]
                print("批次标签形状:", labels.shape)  # [B, H, W]
        else:
            # 创建用于保存的测试数据集
            test_dataset = MixSegDataset(image_dir, label_dir, test_prefixes=test_prefixes,
                                        is_train=False, transform=test_transform, save_dir=augu_dir,
                                        is_second_unet=is_second_unet, mask_dir=mask_dir)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            for inputs, labels, _ in test_loader:
                if is_second_unet:
                    print("批次输入形状:", inputs.shape)  # 第二道UNet: [B, 4, H, W]
                else:
                    print("批次图像形状:", inputs.shape)  # 第一道UNet: [B, 3, H, W]
                print("批次标签形状:", labels.shape)  # [B, H, W]

    