import os
import numpy as np
from skimage.io import imread

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

    # Iterate through ground truth files
    for filename in os.listdir(gt_folder):
        gt_path = os.path.join(gt_folder, filename)
        result_path = os.path.join(result_folder, filename)
        if not os.path.isfile(result_path):
            continue
        # Ensure the corresponding result file exists
        if not os.path.isfile(result_path):
            print(f"Warning: Result file for '{filename}' not found.")
            continue

        # Load images
        gt_image = imread(gt_path, as_gray=True).astype(bool)
        result_image = imread(result_path, as_gray=True).astype(bool)

        # Calculate Dice and IoU
        intersection = np.logical_and(gt_image, result_image).sum()
        union = np.logical_or(gt_image, result_image).sum()
        dice = (2 * intersection) / (gt_image.sum() + result_image.sum() + 1e-6)  # Avoid division by zero
        iou = intersection / (union + 1e-6)  # Avoid division by zero

        # Append metrics
        metrics.append((filename, dice, iou))
        print(f"File: {filename}, Dice: {dice:.4f}, IoU: {iou:.4f}")

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