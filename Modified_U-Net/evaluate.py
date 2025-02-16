import os
import numpy as np
from skimage.io import imread
import cv2

def resize_to_match(image, reference_image, interpolation=cv2.INTER_NEAREST):
    """
    Resize `image` to match the size of `reference_image`.

    Parameters:
        image (numpy array): The image to be resized.
        reference_image (numpy array): The reference image whose size is the target size.
        interpolation (int): Interpolation method for resizing (default: cv2.INTER_NEAREST for masks).
    
    Returns:
        numpy array: Resized image.
    """
    target_size = (reference_image.shape[1], reference_image.shape[0])  # (width, height)

        # OpenCV 不支持 bool 类型，转换为 uint8
    if image.dtype == np.bool_:
        image = image.astype(np.uint8)
    
    return cv2.resize(image, target_size, interpolation=interpolation)

def calculate_dice_and_iou(gt_folder, result_folder):
    """
    Calculate Dice and IoU metrics for segmentation results compared to ground truth.

    Parameters:
        gt_folder (str): Path to the folder containing ground truth images.
        result_folder (str): Path to the folder containing result images.

    Returns:
        metrics (list): A list of tuples containing file name, Dice, and IoU.
    """
    metrics = []

    # Check if folders exist
    if not os.path.isdir(gt_folder):
        print(f"Error: Ground truth folder '{gt_folder}' does not exist.")
        return metrics
    if not os.path.isdir(result_folder):
        print(f"Error: Results folder '{result_folder}' does not exist.")
        return metrics

    # 获取 result 文件夹中的所有文件名（不包含路径）
    result_files = {os.path.splitext(f)[0]: f for f in os.listdir(result_folder)}

    # Iterate through ground truth files
    for gt_filename in os.listdir(gt_folder):
        gt_name, gt_ext = os.path.splitext(gt_filename)  # 分离文件名和扩展名
        gt_path = os.path.join(gt_folder, gt_filename)

        # 检查 result 文件夹中是否有对应的文件（忽略扩展名）
        if gt_name not in result_files:
            print(f"Warning: No matching result file found for '{gt_filename}'. Skipping.")
            continue

        result_filename = result_files[gt_name]  # 获取完整的 result 文件名
        result_path = os.path.join(result_folder, result_filename)

        # Load images
        gt_image = imread(gt_path, as_gray=True).astype(bool)
        result_image = imread(result_path, as_gray=True).astype(bool)

        # Check if images are empty
        if gt_image.sum() == 0 or result_image.sum() == 0:
            print(f"Warning: One of the images is empty for '{gt_filename}'. Skipping this image.")
            continue

        # 自动调整尺寸
        if result_image.shape != gt_image.shape:
            result_image = resize_to_match(result_image, gt_image)

        # Calculate Dice and IoU
        intersection = np.logical_and(gt_image, result_image).sum()
        union = np.logical_or(gt_image, result_image).sum()
        dice = (2 * intersection) / (gt_image.sum() + result_image.sum() + 1e-6)  # Avoid division by zero
        iou = intersection / (union + 1e-6)  # Avoid division by zero

        # Check for NaN values and skip this result if any
        if np.isnan(dice) or np.isnan(iou):
            print(f"Warning: NaN value encountered for '{gt_filename}'. Skipping this image.")
            continue

        # Append metrics
        metrics.append((gt_filename, dice, iou))
        print(f"File: {gt_filename}, Dice: {dice:.4f}, IoU: {iou:.4f}")

    return metrics

def calculate_statistics(metrics):
    """
    Calculate average and variance for Dice and IoU.

    Parameters:
        metrics (list): A list of tuples containing file name, Dice, and IoU.

    Returns:
        stats (dict): A dictionary containing average and variance for Dice and IoU.
    """
    dice_scores = [metric[1] for metric in metrics]
    iou_scores = [metric[2] for metric in metrics]

    stats = {
        "Dice_mean": np.mean(dice_scores),
        "Dice_variance": np.var(dice_scores),
        "IoU_mean": np.mean(iou_scores),
        "IoU_variance": np.var(iou_scores),
    }
    return stats

if __name__ == "__main__":
    # Input ground truth and result folders
    gt_folder = input("Enter the path to the ground truth folder: ").strip()
    result_folder = input("Enter the path to the results folder: ").strip()
    csv_folder = input("Enter the path to the output csv folder: ").strip()
    # Ensure the CSV folder exists, create it if it doesn't
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
        print(f"Created folder: {csv_folder}")

    # Calculate metrics
    results = calculate_dice_and_iou(gt_folder, result_folder)

    # Calculate statistics
    stats = calculate_statistics(results)

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