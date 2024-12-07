import os
from PIL import Image
import numpy as np
import argparse

def ensure_three_channels(image):
    """Ensure the image has three channels."""
    if len(image.shape) == 2:  # Grayscale image
        return np.stack([image] * 3, axis=-1)
    return image

def visualize_and_combine_results(original_dir, gt_dir, seg_dir, output_path):
    """
    Combine segmentation results into a single large image.

    Parameters:
        original_dir (str): Path to the folder containing original images.
        gt_dir (str): Path to the folder containing ground truth masks.
        seg_dir (str): Path to the folder containing segmentation masks.
        output_path (str): Path to save the combined image.
    """
    # Get file names that are common across all directories
    original_files = set(os.listdir(original_dir))
    gt_files = set(os.listdir(gt_dir))
    seg_files = set(os.listdir(seg_dir))

    common_files = original_files & gt_files & seg_files

    if not common_files:
        print("No common files found in the provided directories.")
        return

    rows = []  # Store all rows of combined images
    for file_name in sorted(common_files):
        original_path = os.path.join(original_dir, file_name)
        gt_path = os.path.join(gt_dir, file_name)
        seg_path = os.path.join(seg_dir, file_name)

        # Load images and convert to same size
        original_image = Image.open(original_path)
        gt_image = Image.open(gt_path)
        seg_image = Image.open(seg_path)

        # Ensure all images have the same size
        width, height = original_image.size
        gt_image = gt_image.resize((width, height))
        seg_image = seg_image.resize((width, height))

        # Combine three images horizontally
        combined_row = np.hstack([
            ensure_three_channels(np.array(original_image)),
            ensure_three_channels(np.array(gt_image)),
            ensure_three_channels(np.array(seg_image))
        ])
        rows.append(combined_row)

    # Combine all rows vertically
    combined_image = np.vstack(rows)

    # Save the final combined image
    combined_image = Image.fromarray(combined_image)
    combined_image.save(output_path)
    print(f"Combined image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine segmentation results into a single large image.")
    parser.add_argument("original_dir", type=str, help="Path to the folder containing original images.")
    parser.add_argument("gt_dir", type=str, help="Path to the folder containing ground truth masks.")
    parser.add_argument("seg_dir", type=str, help="Path to the folder containing segmentation masks.")
    parser.add_argument("output_path", type=str, help="Path to save the combined image.")

    args = parser.parse_args()

    visualize_and_combine_results(args.original_dir, args.gt_dir, args.seg_dir, args.output_path)
