import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize
import numpy as np
from sam.segment_anything_ori.utils.transforms import ResizeLongestSide

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

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        original_size = image.size
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8)).convert("L")

        # assert os.path.exists(mask_path), f"Mask file {mask_path} not found for {img_path}"

        resize = Resize(self.target_size)
        image = resize(image)
        mask = resize(mask)

        if self.sam is not None:
            image_array = np.array(image)
            transform = ResizeLongestSide(self.sam.image_encoder.img_size)
            image = transform.apply_image(image_array)
            

        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(np.asarray(mask) / 255.0, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        assert mask.min() >= 0 and mask.max() <= 1, "Masks must be normalized to [0, 1]"

        if self.sam is None:
            return image, mask
        else:
            input_size = torch.tensor(self.target_size, dtype=torch.int)
            original_size = torch.tensor(original_size, dtype=torch.int)
            print(f"input_size: {input_size}")
            print(f"original_size: {original_size}")
            box = torch.tensor([0,0,self.target_size[0],self.target_size[1]], dtype=torch.float32)
            image, mask, box = image.to(self.device), mask.to(self.device), box.to(self.device)
            
            #transform = ResizeLongestSide(self.sam.image_encoder.img_size)
            #input_image = transform.apply_image(image)
            #print(f"image: {image.shape}")
            #transformed_image = image.unsqueeze(0)#image.permute(2, 0, 1).contiguous()[None, :, :, :]
            #print(f"transformed_image: {transformed_image.shape}")
            image = self.sam.preprocess(image)
            
            
            return image, mask, box, input_size, original_size

    def __len__(self):
        return len(self.images)
