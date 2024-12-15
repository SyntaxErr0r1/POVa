"""
Check if every label file has a corresponding image file in the dataset directory and vice versa.
Usage: python dataset_consistency.py /path/to/dataset
"""

import os
import sys

def check_dataset_consistency(root_dir):
    # Define paths to images and labels directories
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'labels')

    # Get lists of file names in each directory (excluding extensions for comparison)
    labels_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))}
    images_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))}

    # Find images files without corresponding labels files
    missing_labels = images_files - labels_files

    # Print results
    if missing_labels:
        print("The following image files do not have corresponding label files:")
        for missing in missing_labels:
            # skip if filename contains 'negative' or 'neg'
            if 'neg' in missing:
                continue
            # print in yellow color
            print("\033[93m", missing)
        print("\033[0m")
        print("Note: Files with 'negative' or 'neg' in their names are skipped. The file still might be a negative sample though.")
    else:
        print("All image files have corresponding label files.")

    # Find labels files without corresponding images files
    missing_images = labels_files - images_files

    # Print results
    if missing_images:
        print("The following label files do not have corresponding image files:")
        for missing in missing_images:
            # print in red color
            print("\033[91m", missing)
        print("\033[0m")
    else:
        print("All label files have corresponding image files.")

# Example usage
if __name__ == "__main__":
    
    # get root directory from command line argument
    if len(sys.argv) != 2:
        print("Usage: python dataset_consistency.py /path/to/dataset")
        sys.exit(1)

    root_directory = sys.argv[1]
    check_dataset_consistency(root_directory)
