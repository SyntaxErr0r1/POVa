import os
from pathlib import Path

# Cesty k datasetu
images_dir = Path("./datasets/Merged_newest/train_augmented_small_random_sam/images")
masks_dir = Path("./datasets/Merged_newest/train_augmented_small_random_sam/labels")

# Načtení seznamu všech masek
mask_names = set(mask.stem for mask in masks_dir.glob("*"))

# Procházení složky s obrázky
for image in images_dir.glob("*"):
    image_name = image.stem  # Název souboru bez přípony
    if image_name not in mask_names:
        print(f"Deleting {image} as it has no matching mask...")
        image.unlink()  # Smaže soubor
