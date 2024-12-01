import os
import shutil
import argparse
'''
input this command to run this script:
    python match_and_copy.py [path of the original dataset (bigger one)] [path of the reference dataset (smaller one)] [folder path you want to store the output dataset]
for example:
    python match_and_copy.py Dataset/BUS_all_dataset_resize/test/images Dataset/BUS_reduced/labels Dataset/BUS_reduced/images
You can add suffix and file type in the end of the command, they are optional arguments (their default set is _pred and .png):
    python match_and_copy.py Dataset/BUS_all_dataset_resize/test/images Dataset/BUS_reduced/labels Dataset/BUS_reduced/images --suffix _pred --extension .png
'''
def match_and_copy_files(input_folder, reference_folder, output_folder, suffix_to_remove="_pred", file_extension=".png"):
    """
    Copies files from the input folder to the output folder based on matching file names in the reference folder.

    Args:
        input_folder (str): Path to the input folder containing files to copy.
        reference_folder (str): Path to the reference folder with files providing matching names.
        output_folder (str): Path to the output folder where matched files will be copied.
        suffix_to_remove (str): Suffix to remove from file names in the reference folder for matching (default: "_pred").
        file_extension (str): File extension to filter and match (default: ".png").
    """
    # Ensure the target folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract base names (with suffix removed) from the reference folder
    

    reference_files = [f for f in os.listdir(reference_folder) if f.endswith(file_extension)]
    reference_basenames = []
    # Iterate through the reference folder to get the name without the suffix
    for f in reference_files:
        if f.endswith(file_extension):
            ff = f.replace(suffix_to_remove, "")
            reference_basenames.append(os.path.splitext(ff)[0])
    
    # Iterate through the input folder and match files
    for file in os.listdir(input_folder):
        if file.endswith(file_extension):
            base_name = os.path.splitext(file)[0]
            # print(base_name)
            if base_name in reference_basenames:
                source_path = os.path.join(input_folder, file)
                target_path = os.path.join(output_folder, file)
                shutil.copy2(source_path, target_path)
                print(f"Copied: {file}")
            # else:
            #     print("input base_name is NOT in reference_basenames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files from an input folder to an output folder based on matching names in a reference folder.")
    parser.add_argument("input_folder", help="Path to the input folder containing files to copy.")
    parser.add_argument("reference_folder", help="Path to the reference folder for matching file names.")
    parser.add_argument("output_folder", help="Path to the output folder where matched files will be copied.")
    parser.add_argument("--suffix", default="_pred", help="Suffix to remove from file names in the reference folder (default: '_pred').")
    parser.add_argument("--extension", default=".png", help="File extension to match (default: '.png').")

    args = parser.parse_args()

    match_and_copy_files(args.input_folder, args.reference_folder, args.output_folder, args.suffix, args.extension)
