import os

def remove_suffix_from_filenames(folder_path, target_suffix):
    """
    Remove a specific suffix from filenames while preserving the original file extension.

    Parameters:
        folder_path (str): The path to the folder containing the files.
        target_suffix (str): The suffix to remove from the filenames (e.g., "_pred").
    """
    try:
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            print(f"Error: The folder '{folder_path}' does not exist.")
            return

        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            # Split the filename into name and extension
            name, ext = os.path.splitext(filename)

            # Check if the filename ends with the target suffix
            if name.endswith(target_suffix):
                # Generate the new filename
                new_name = name[: -len(target_suffix)] + ext

                # Construct full paths
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_name)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    folder_path = input("Enter the folder path: ").strip()
    target_suffix = input("Enter the suffix to remove (e.g., '_pred'): ").strip()
    remove_suffix_from_filenames(folder_path, target_suffix)
