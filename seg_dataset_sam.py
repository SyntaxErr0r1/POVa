import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize
import numpy as np
from sam.segment_anything_ori.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt

# Custom Dataset
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256), sam=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        self.target_size = target_size
        self.sam = sam
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dummy_prompt = self.generate_grid_points()
        self.dummy_labels = np.ones(len(self.dummy_prompt))

    def generate_grid_points(self, step=32):
        y_coords = np.arange(0, self.target_size[0], step)
        x_coords = np.arange(0, self.target_size[1], step)
        grid_points = np.array([[x, y] for y in y_coords for x in x_coords])
        return grid_points

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        original_size = image.size
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("1")
        else:
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8)).convert("L")

        #assert os.path.exists(mask_path), f"Mask file {mask_path} not found for {img_path}"

        resize = Resize(self.target_size)
        image = resize(image)
        mask = resize(mask)

        if self.transform:
            image = self.transform(image)
            
        mask = torch.tensor(np.asarray(mask), dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        assert mask.min() >= 0 and mask.max() <= 1, "Masks must be normalized to [0, 1]"


        if self.sam is None:
            return image, mask
        else:
            return image, mask, self.dummy_prompt, self.dummy_labels

    def __len__(self):
        return len(self.images)
