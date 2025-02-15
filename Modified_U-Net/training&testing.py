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
# 修改
from Prepare_dataset import dataset_dir, is_train
if is_train:
    from Prepare_dataset import train_loader
else:
    from Prepare_dataset import test_loader
from code_UNet_model import UNet, Encoder
import evaluate
from datetime import datetime
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss() # Combines a sigmoid activation function and Binary Cross-Entropy loss. Works directly with logits (raw model outputs) instead of manually applied sigmoid.

    def forward(self, logits, targets): # Computes BCE + Dice loss for the given predictions (logits) and ground truth (targets).

        if len(targets.shape) == 3:  # If missing channel dimension add one
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(logits)  # Convert logits to probabilities with sigmoid function
        smooth = 1e-5  # To avoid division by zero

        # Compute Dice coefficient (dice) and Dice loss (dice_loss)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        # Compute BCE Loss
        bce_loss = self.bce(logits, targets)

        # Combine losses
        return bce_loss + dice_loss

# Training loop
def train_model(model, train_loader, criterion, optimizer, save_root, num_epochs=50): # criterions is the loss function
    model.train() # Activate training mode in the model

    # 生成唯一的文件夹名称（基于当前时间）
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_root, f"training_{timestamp}")
    os.makedirs(save_path, exist_ok=True)  # 确保目录存在
    N = 5  # 每 5 轮保存一次

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0 # Initialize a variable to accumulate the total loss for the current epoch.
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) # Transfer to appropriate device for processing
            optimizer.zero_grad() # Resets gradient of model's parameter as gradients persist in Pytorch by default and must be cleared before backpropagation
            outputs = model(images) # Passes the input images through the model to generate predictions (outputs)
            loss = criterion(outputs, labels) # Computes loss by comparing prediction (output) with ground truth (labels)
            loss.backward() # Performs backpropagation to compute gradients for the model's parameters with respect to the loss
            optimizer.step() # Update model's parameters
            epoch_loss += loss.item() # loss.item(): Converts the PyTorch tensor (loss) to a scalar value for accumulation.
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}") # Visualization


        if (epoch + 1) % N == 0:
          torch.save(model.state_dict(), f"{save_path}/model_sd_epoch_{epoch+1}.pth")
          torch.save(model, f"{save_path}/model_full_{epoch+1}.pth")

def evaluate_model(model, test_loader):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model.eval()
    test_image_dir = os.path.join(dataset_dir, 'test', 'images')
    original_filenames = sorted(os.listdir(test_image_dir))

    step = 5  # 每隔5张打印一次

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images))
            outputs = (outputs > 0.5).float()

            save_dir = os.path.join(dataset_dir, 'test', 'prompts')
            for i in range(len(outputs)):
                original_filename = original_filenames[idx * len(outputs) + i]
                name, ext = os.path.splitext(original_filename)
                new_filename = f"{name}.png"
                mask = outputs[i].squeeze().cpu().numpy() * 255
                mask = mask.astype('uint8')
                mask_path = os.path.join(save_dir, new_filename)
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                Image.fromarray(mask).save(mask_path)
            print(f"Predictions saved in: {save_dir}")

            # 选取间隔5的索引
            indices = list(range(0, len(images), step))
            num_selected = len(indices)

            if num_selected == 0:
                continue  # 如果没有足够的样本，跳过

            fig, axes = plt.subplots(num_selected, 3, figsize=(10, num_selected * 3), squeeze=False)

            # 设置每列的标题，只在第一行显示
            column_titles = ["Input Image", "True Mask", "Predicted Mask"]
            for col in range(3):
                axes[0, col].set_title(column_titles[col])

            for row, idx in enumerate(indices):  # 遍历被选中的索引
                # 直接使用张量，无需反归一化
                image = images[idx].cpu().permute(1, 2, 0).numpy()
                image = np.clip(image, 0, 1)  # 确保值在 [0,1]

                # 绘制输入图像
                axes[row, 0].imshow(image)
                axes[row, 0].axis('off')

                # 绘制真实标签
                axes[row, 1].imshow(labels[idx].cpu().numpy().squeeze(), cmap='gray')
                axes[row, 1].axis('off')

                # 绘制预测结果
                axes[row, 2].imshow(outputs[idx].cpu().numpy().squeeze(), cmap='gray')
                axes[row, 2].axis('off')

            plt.tight_layout()
            plt.show()
    
    # Input ground truth and result folders
    gt_folder = os.path.join(dataset_dir, 'test', 'labels')
    result_folder = save_dir
    csv_folder = os.path.join(dataset_dir, 'test', 'csv')
    # Ensure the CSV folder exists, create it if it doesn't
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
        print(f"Created folder: {csv_folder}")

    # Calculate metrics
    results = evaluate.calculate_dice_and_iou(gt_folder, result_folder)

    # Calculate statistics
    stats = evaluate.calculate_statistics(results)

    # Save individual metrics to CSV
    individual_file = os.path.join(csv_folder, "individual_metrics.csv")
    try:
        with open(individual_file, "w", newline='') as f:
            f.write("Filename,Dice,IoU\n")
            for filename, dice, iou in results:
                f.write(f"{filename},{dice:.4f},{iou:.4f}\n")
        print(f"Individual metrics saved to '{individual_file}'.")
    except Exception as e:
        print(f"Error saving individual metrics: {e}")

    # Save summary metrics to CSV
    summary_file = os.path.join(csv_folder, "summary_metrics.csv")
    try:
        with open(summary_file, "w", newline='') as f:
            f.write("Metric,Mean,Variance\n")
            f.write(f"Dice,{stats['Dice_mean']:.4f},{stats['Dice_variance']:.4f}\n")
            f.write(f"IoU,{stats['IoU_mean']:.4f},{stats['IoU_variance']:.4f}\n")
        print(f"Summary metrics saved to '{summary_file}'.")
    except Exception as e:
        print(f"Error saving summary metrics: {e}")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = UNet(in_ch=3, out_ch=1).to(device)  

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
    
    if is_train:
        train_model(model, train_loader, criterion, optimizer, num_epochs=8)
        torch.save(model.state_dict(), 'model_sd.pth')
        # torch.save(model, 'model_full.pth')
    else:
        # Load training
        model_path = input("Please enter the path to the model(default:/root/checkpoints/model_sd_epoch_10.pth):") or '/root/checkpoints/model_sd_epoch_10.pth'
        # my_model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))  
        evaluate_model(model, test_loader)

    