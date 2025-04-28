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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
# 修改
from Prepare_dataset import dataset_dir, is_train, dataset_type
if is_train:
    from Prepare_dataset import train_loader
else:
    from Prepare_dataset import test_loader, test_prefixes
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

    # Generate unique folder name (based on current time)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_root, f"training_{timestamp}")
    os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
    N = 10  # Save every 10 epochs

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0 # Initialize a variable to accumulate the total loss for the current epoch.
        for images, labels, _ in train_loader:
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
    model.eval()
    
    # Generate timestamp for folder names
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    
    # Create dataset suffix from test_prefixes
    if hasattr(test_loader.dataset, 'test_prefixes') and test_loader.dataset.test_prefixes:
        dataset_suffix = ','.join(test_loader.dataset.test_prefixes)
    elif 'test_prefixes' in globals() and test_prefixes:
        dataset_suffix = ','.join(test_prefixes)
    else:
        dataset_suffix = "test"
    
    # Folder suffix format: dataset_suffix + timestamp 
    folder_suffix = f"{dataset_suffix}_{timestamp}"
    
    # Determine image, label, and prediction paths based on dataset type
    if dataset_type == 'standard':
        # Standard dataset structure: with test subfolder
        images_dir = os.path.join(dataset_dir, 'test', 'images')
        labels_dir = os.path.join(dataset_dir, 'test', 'labels')
        save_dir = os.path.join(dataset_dir, 'test', f"prompts_{folder_suffix}")
        csv_folder = os.path.join(dataset_dir, 'test', f"csv_{folder_suffix}")
    else:  # mixseg
        # MixSeg dataset structure: directly in root directory
        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')
        save_dir = os.path.join(dataset_dir, f"prompts_{folder_suffix}")
        csv_folder = os.path.join(dataset_dir, f"csv_{folder_suffix}")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    
    print(f"Prediction results will be saved to: {save_dir}")
    print(f"Evaluation metrics will be saved to: {csv_folder}")

    with torch.no_grad():
        for idx, (images, labels, filenames) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images))
            outputs = (outputs > 0.5).float()

            for i in range(len(outputs)):
                original_filename = filenames[i]  # Get directly from DataLoader
                name, ext = os.path.splitext(original_filename)
                new_filename = f"{name}.png"
                mask = outputs[i].squeeze().cpu().numpy() * 255
                mask = mask.astype('uint8')
                mask_path = os.path.join(save_dir, new_filename)
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                Image.fromarray(mask).save(mask_path)
            print(f"Batch {idx+1} predictions saved.")

            step = min(5, len(images))  # Ensure step is within range
            # Select indices at step intervals
            indices = list(range(0, len(images), step))
            if not indices:
                continue

            fig, axes = plt.subplots(len(indices), 3, figsize=(10, len(indices) * 3), squeeze=False)

            # Set column titles, only shown in first row
            column_titles = ["Input Image", "True Mask", "Predicted Mask"]
            for col in range(3):
                axes[0, col].set_title(column_titles[col])

            for row, idx in enumerate(indices):  # Iterate through selected indices
                # Use tensors directly, no need for denormalization

                image = images[idx].cpu().permute(1, 2, 0).numpy()
                image = np.clip(image, 0, 1)  # Ensure values in [0,1]

                # Draw input image
                axes[row, 0].imshow(image)
                axes[row, 0].axis('off')

                # Draw true label
                axes[row, 1].imshow(labels[idx].cpu().numpy().squeeze(), cmap='gray')
                axes[row, 1].axis('off')

                # Draw prediction
                axes[row, 2].imshow(outputs[idx].cpu().numpy().squeeze(), cmap='gray')
                axes[row, 2].axis('off')

            plt.tight_layout()
            plt.show()
    
    print(f"All predictions saved in: {save_dir}")
    
    # Use previously determined label and result directories
    gt_folder = labels_dir
    result_folder = save_dir

    gt_files = {os.path.splitext(f)[0] for f in os.listdir(gt_folder)}
    result_files = {os.path.splitext(f)[0] for f in os.listdir(result_folder)}

    missing_files = gt_files - result_files
    if missing_files:
        print(f"Warning: Missing result files for {len(missing_files)} ground truth images: {missing_files}")

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
        save_root = input("Enter model save path (default: /root/checkpoints): ") or '/root/checkpoints'
        train_model(model, train_loader, criterion, optimizer, save_root=save_root, num_epochs=8)
        torch.save(model.state_dict(), f'{save_root}/model_sd_latest.pth')
        # torch.save(model, 'model_full.pth')
    else:
        # Load training
        model_path = input("Enter model path (default: /root/checkpoints/model_sd_epoch_10.pth): ") or '/root/checkpoints/model_sd_epoch_10.pth'
        # my_model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))  
        evaluate_model(model, test_loader)

    