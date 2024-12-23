import os
import cv2
import numpy as np

# Script to plot augmentation methods on a single image

# This file assumes that the specified directory contains both original image and images created through augmentation
# Images created through augmentation are expected to have the augmentation method in brackets at the end of the filename
# Example "214OLCV1_100H0006(brightness).jpeg"

# Directory containing augmented images
directory = "./datasets/mer/train_test_augmented/images"

# Gets augmentation method from a image filename
def get_augmentation_method(filename):
    start = filename.find("(")
    end = filename.find(")")
    if start != -1 and end != -1:
        return filename[start+1:end].capitalize()
    return "Original"

# Plots augmented images in a grid
def plot_images_grid(directory, image_size=(128, 128), images_per_row=5, padding=10):
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    if not image_files:
        print("No images found in the directory.")
        return

    original_images = []
    augmented_images = []

    for image_file in image_files:
        method = get_augmentation_method(image_file)
        if method == "Original":
            original_images.append(image_file)
        else:
            augmented_images.append(image_file)

    sorted_files = original_images + augmented_images
    processed_images = []
    labels = []

    for image_file in sorted_files:
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read {image_file}")
            continue

        resized_image = cv2.resize(image, image_size)
        processed_images.append(resized_image)

        method = get_augmentation_method(image_file)
        labels.append(method)

    num_images = len(processed_images)
    num_rows = (num_images + images_per_row - 1) // images_per_row
    grid_width = images_per_row * (image_size[0] + padding) - padding
    grid_height = num_rows * (image_size[1] + 20 + padding) - padding

    collage = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    for idx, (image, label) in enumerate(zip(processed_images, labels)):
        row = idx // images_per_row
        col = idx % images_per_row

        x = col * (image_size[0] + padding)
        y = row * (image_size[1] + 20 + padding)

        collage[y + 20:y + 20 + image_size[1], x:x + image_size[0]] = image

        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_x = x + (image_size[0] - label_size[0]) // 2
        label_y = y + 15
        cv2.putText(collage, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite("augment_collage.jpg", collage)
    print(f"Collage saved to augment_collage.jpg")



plot_images_grid(directory)