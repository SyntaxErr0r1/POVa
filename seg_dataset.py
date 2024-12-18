import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize
import numpy as np

# Custom Dataset
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        self.target_size = target_size

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8)).convert("L")

        # assert os.path.exists(mask_path), f"Mask file {mask_path} not found for {img_path}"

        resize = Resize(self.target_size)
        image = resize(image)
        mask = resize(mask)

        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(np.asarray(mask) / 255.0, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        assert mask.min() >= 0 and mask.max() <= 1, "Masks must be normalized to [0, 1]"

        return image, mask

    def __len__(self):
        return len(self.images)
