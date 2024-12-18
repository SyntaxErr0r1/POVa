import os
import shutil
from tqdm import tqdm

def copy_files(src, dst, file_name_list=None):
    # Ensure the destination directory exists
    os.makedirs(dst, exist_ok=True)
    
    counter = 0
    if file_name_list is not None:
        # Copy the files in the file_name_list
        for filename in file_name_list:
            src_file = os.path.join(src, filename)
            dst_file = os.path.join(dst, filename)
            
            # Copy the file if it is a file (and not a directory)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)  # Use copy2 to preserve metadata
    else:
        # List files in the source directory
        for filename in os.listdir(src):
            if filename.split('.')[1] != 'jpg' and filename.split('.')[1] != 'png':
                continue
            src_file = os.path.join(src, filename)
            dst_file = os.path.join(dst, filename)
            
            # Copy the file if it is a file (and not a directory)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)  # Use copy2 to preserve metadata
                counter += 1
    
    print(f"Copied {counter} files from {src} to {dst}")

train_images_dst = './datasets/Kvasir-SEG_splitted/train/images'
train_masks_dst = './datasets/Kvasir-SEG_splitted/train/labels'

val_kvasir_images_dst = './datasets/Kvasir-SEG_splitted/val_kvasir/images'
val_kvasir_masks_dst = './datasets/Kvasir-SEG_splitted/val_kvasir/labels'


# ---------- Kvassir-SEG --------------
print("Copying Kvassir-SEG...")

images_dir = './datasets/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images'
masks_dir = './datasets/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/masks'

# Read file splits
def read_split(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() + ".jpg" for line in f.readlines()]

train_files = read_split('./datasets/Kvasir-SEG/train.txt')
val_files = read_split('./datasets/Kvasir-SEG/val.txt')

# create directories
os.makedirs(train_images_dst, exist_ok=True)
os.makedirs(train_masks_dst, exist_ok=True)
os.makedirs(val_kvasir_images_dst, exist_ok=True)
os.makedirs(val_kvasir_masks_dst, exist_ok=True)

# Copy images
copy_files(images_dir, train_images_dst, train_files)
copy_files(images_dir, val_kvasir_images_dst, val_files)

# Copy masks
copy_files(masks_dir, train_masks_dst, train_files)
copy_files(masks_dir, val_kvasir_masks_dst, val_files)

print("Done")
