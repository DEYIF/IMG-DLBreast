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
from Prepare_dataset import test_loader, train_loader
from code_U-Net_model import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = UNet(in_ch=3, out_ch=1).to(device)  


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss() 

    def forward(self, logits, targets): 

        if len(targets.shape) == 3:  
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(logits)
        smooth = 1e-5  

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        bce_loss = self.bce(logits, targets)

        return bce_loss + dice_loss

criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 


def train_model(model, train_loader, criterion, optimizer, num_epochs=50): 
    model.train() 
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 
            optimizer.zero_grad() 
            outputs = model(images) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            epoch_loss += loss.item() 
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}") 


def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = image * std + mean  
    return image

print("Please enter the paths for your image and label directories:")
save_dir = input("Path to train image directory (default: './Data/prediction/'): ") or './Data/prediction/'
