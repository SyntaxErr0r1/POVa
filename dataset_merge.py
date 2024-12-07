import os
import shutil

def copy_files(src, dst):
    # Ensure the destination directory exists
    os.makedirs(dst, exist_ok=True)
    
    # List files in the source directory
    for filename in os.listdir(src):
        if filename.split('.')[1] != 'jpg':
            continue
        src_file = os.path.join(src, filename)
        dst_file = os.path.join(dst, filename)
        
        # Copy the file if it is a file (and not a directory)
        if os.path.isfile(src_file) and not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)  # Use copy2 to preserve metadata

train_images_dst = './datasets/merged/train/images'
val_kvasir_images_dst = './datasets/merged/val_kvasir/images'
val_clinic_images_dst = './datasets/merged/val_clinic/images'

train_masks_dst = './datasets/merged/train/labels'
val_kvasir_masks_dst = './datasets/merged/val_kvasir/labels'
val_clinic_masks_dst = './datasets/merged/val_clinic/labels'

# ---------- CVC-ClinicDB --------------
print("Copying CVC-ClinicDB...")

src_labels = './datasets/CVC-ClinicDB/PNG/Ground Truth'
src_images = './datasets/CVC-ClinicDB/PNG/Original'

copy_files(src_labels, val_clinic_masks_dst)
copy_files(src_images, val_clinic_images_dst)

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

# Copy images
for filename in train_files:
    if not os.path.exists(os.path.join(train_images_dst, filename)):
        shutil.copy2(os.path.join(images_dir, filename), os.path.join(train_images_dst, filename))
    if not os.path.exists(os.path.join(train_masks_dst, filename)):
        shutil.copy2(os.path.join(masks_dir, filename), os.path.join(train_masks_dst, filename))  # Use copy2 to preserve metadata

for filename in val_files:
    if not os.path.exists(os.path.join(val_kvasir_images_dst, filename)):
        shutil.copy2(os.path.join(images_dir, filename), os.path.join(val_kvasir_images_dst, filename))  # Use copy2 to preserve metadata
    if not os.path.exists(os.path.join(val_kvasir_masks_dst, filename)):
        shutil.copy2(os.path.join(masks_dir, filename), os.path.join(val_kvasir_masks_dst, filename))  # Use copy2 to preserve metadata
        
# ---------- PolpGen --------------
print("Copying PolyGen...")
data_path = './datasets/PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3'

    # ------------- data_C1 - data_C6 ------------- 
for dir in os.listdir(data_path):
    if dir.split('_')[0] == 'data':
        copy_files(os.path.join(data_path, dir, 'images_' + dir.split('_')[1]), train_images_dst)
        copy_files(os.path.join(data_path, dir, 'masks_' + dir.split('_')[1]), train_masks_dst)

    # ------------- Sequence negative ------------- 
data_path_neg = os.path.join(data_path, 'sequenceData/negativeOnly')
for dir in os.listdir(data_path_neg):
    # Skip .DS_Store file
    if dir[0] == '.':
        continue
    copy_files(os.path.join(data_path_neg, dir), train_images_dst)

    # ------------- Sequence positive ------------- 
data_path_pos = os.path.join(data_path, 'sequenceData/positive')
for dir in os.listdir(data_path_pos):
    # Skip .DS_Store file
    if dir[0] == '.':
        continue
    dirpath = os.path.join(data_path_pos, dir)
    copy_files(os.path.join(dirpath, 'images_' + dir), train_images_dst)
    copy_files(os.path.join(dirpath, 'masks_' + dir), train_masks_dst)

print("Done")