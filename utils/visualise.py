"""
Script for selecting sample images from the validation set and visualizing the ground truth and predicted masks.

Works with both UNet and SAM models.
"""

import eval

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.utils.metrics import IoU, Fscore
from sam.segment_anything_ori import sam_model_registry
from seg_dataset_sam import SegmentationDataset
from torchvision.transforms import ToTensor, Normalize, Compose
from rich.logging import RichHandler
import logging
import argparse
import matplotlib.pyplot as plt

def main():
    print("Hello")
    parser = argparse.ArgumentParser(description="Evaluate a binary segmentation model.")
    parser.add_argument("--arch", choices=["Unet", "SAM"], default="Unet", help="Specify the architecture of model for evaluation (default: Unet).")
    parser.add_argument("--model", required=True, help="Filepath to the segmentation model.")
    parser.add_argument("--data", required=True, help="Filepath to the evaluation dataset (images and labels directories).")
    parser.add_argument("--name", required=False, default="image", help="Name of the saved images.")

    args = parser.parse_args()

    # ------------------ Load val dataset ----------------
    image_dir = os.path.join(args.data, "images")
    label_dir = os.path.join(args.data, "labels")
    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        raise ValueError("Evaluation dataset must contain 'images' and 'labels' directories.")
    
    logging.info("Evaluation Parameters:")
    logging.info(f"Architecture:     {args.arch}")
    logging.info(f"Evaluating model: {args.model}")
    logging.info(f"Evaluation data:  {args.data}")

    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Same as in train.py
    ])

    # Load model
    if args.arch == "Unet":
        dataset = SegmentationDataset(image_dir=image_dir, mask_dir=label_dir, transform=transform, sam=False)
        model = eval.load_Unet(args.model)
    elif args.arch == "SAM":
        #TODO: Implement SAM model loading
        dataset = SegmentationDataset(image_dir=image_dir, mask_dir=label_dir, transform=transform, sam=True)
        model = eval.load_SAM(args.model)

    # model.eval()

    # if does not exist, create figures directory
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Visualize 5 images from the dataset so that the ids are 0th 20%th 40%th 60%th 80%th and 100%th 
    idxs = [0, len(dataset)//5, len(dataset)//5*2, len(dataset)//5*3, len(dataset)//5*4, len(dataset)-1]
    for (i, idx) in enumerate(idxs):
        image, mask, *_ = dataset[idx]
        image = image.unsqueeze(0) 
        mask = mask.unsqueeze(0)

        # Predict mask
        with torch.no_grad():
            pred_mask = model(image)
            pred_mask = torch.sigmoid(pred_mask)

        # Convert to numpy arrays
        image = ( image * 0.5 + 0.5 ).squeeze().permute(1, 2, 0).numpy()
        mask = mask.squeeze().numpy()
        pred_mask = pred_mask.squeeze().numpy()

        # Plot images
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        # plt.show()

        # save image
        plt.savefig(f"figures/{args.name}_{i}.png")
        

if __name__ == "__main__":
    ###################################
    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt="%H:%M:%S",
        handlers=[RichHandler()]
    )
    logging.getLogger("rich")
    ###################################
    main()
