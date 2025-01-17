from augment_funcs import *
import argparse
from PIL import Image
import os
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Augment a dataset.")
    parser.add_argument("-data", required=True, help="Filepath to the training dataset (images and labels directories).")

    args = parser.parse_args()

    # ------------------ Load TRAIN dataset ----------------
    image_dir = os.path.join(args.data, "images")
    label_dir = os.path.join(args.data, "labels")
    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        raise ValueError("Training dataset must contain 'images' and 'labels' directories.")

    augmentations = [gaussian_noise, shot_noise, impulse_noise, speckle_noise, gaussian_blur,
                     glass_blur, defocus_blur, motion_blur, fog, spatter, contrast,
                     brightness, saturate, jpeg_compression, pixelate, elastic_transform, horizontal_flip, vertical_flip]

    images = os.listdir(image_dir)
    images_len = len(images)
    augmented_dir = f"{args.data}_augmented_small_random"

    os.makedirs(os.path.join(augmented_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(augmented_dir, "labels"), exist_ok=True)

    image_count = 0
    for image_path in images:
        count = 1
        image = Image.open(os.path.join(args.data, "images", image_path)).convert("RGB")

        # Select random 2 augmentations
        random_augmentations = random.sample(augmentations, 2)

        for augment in random_augmentations:
            severity = random.randint(3, 5)  # Generate random severity from 3 to 5
            augmented_img = augment(image, severity=severity)
            augmented_img = np.clip(augmented_img, 0, 255).astype(np.uint8)  # Fix pixel range and dtype
            augmented_image_pil = Image.fromarray(augmented_img, mode="RGB")  # Ensure mode is RGB
            augmented_image_pil.save(os.path.join(augmented_dir, "images", f"{image_path.split('.')[0]}({augment.__name__}).{image_path.split('.')[1]}"))
            image.save(os.path.join(augmented_dir, "images", image_path))

            # If a mask exists, augment or duplicate it
            if image_path in os.listdir(label_dir):
                image_mask = Image.open(os.path.join(args.data, "labels", image_path))
                # Augment mask too for these augmentations
                if augment.__name__ in ["motion_blur", "pixelate", "elastic_transform", "horizontal_flip", "vertical_flip"]:
                    mask_array = np.array(image_mask.convert("L"))
                    mask_array = (mask_array > 127).astype(np.uint8) * 255  # Convert to 0 and 255
                    if mask_array.ndim == 3:
                        mask_array = mask_array[:, :, 0]  # Remove extra dimension if present

                    mask_pil = Image.fromarray(mask_array, mode="L")
                    augmented_mask_pil = augment(mask_pil, severity=severity)  # Pass mask_pil to augment
                    augmented_mask_array = np.array(augmented_mask_pil)
                    
                    if augmented_mask_array.ndim == 3:
                        augmented_mask_array = augmented_mask_array[:, :, 0]

                    binary_augmented_mask = (augmented_mask_array > 127).astype(np.uint8) * 255  # Re-binarize
                    augmented_mask_pil = Image.fromarray(binary_augmented_mask, mode="L")
                    augmented_mask_pil.save(os.path.join(augmented_dir, "labels", f"{image_path.split('.')[0]}({augment.__name__}).{image_path.split('.')[1]}"))   
                else:
                    # Copy the original mask as augmented
                    image_mask.save(os.path.join(augmented_dir, "labels", f"{image_path.split('.')[0]}({augment.__name__}).{image_path.split('.')[1]}"))   
                # Save the original mask
                image_mask.save(os.path.join(augmented_dir, "labels", image_path))

            count += 1
        image_count += 1

        print(f"\rProgress: {(image_count/images_len) * 100:.2f}%", end="", flush=True)

    print("\nAugmentation complete!")

if __name__ == "__main__":
    main()
