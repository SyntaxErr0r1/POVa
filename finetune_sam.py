import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import SAM_fine_tune.src.utils as utils
from SAM_fine_tune.src.dataloader import DatasetSegmentation, collate_fn
from SAM_fine_tune.src.processor import Samprocessor
from SAM_fine_tune.src.segment_anything import build_sam_vit_b, SamPredictor
from SAM_fine_tune.src.lora import LoRA_sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
import argparse
from torchvision.transforms import ToTensor, Normalize, Compose
from pathlib import Path
import logging
from rich.logging import RichHandler

from seg_dataset import SegmentationDataset

"""
This file is inspired by https://github.com/WangRongsheng/SAM-fine-tune/tree/main project.
"""
##
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    datefmt="%H:%M:%S",
    handlers=[RichHandler()]
)
logging.getLogger("rich")

##
# Save model checkpoint
# @param model: Model to save
# @param optimizer: Optimizer state
# @param epoch: Epoch number
def save_checkpoint(model, optimizer, epoch):
    cptPath = Path(args.cfg)
    cptName = "SAM_"+cptPath.stem+".pth"
    
    cptDir = Path("models")
    cptDir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, cptDir/cptName)

## Parse the arguments
parser = argparse.ArgumentParser(description="Script for finetuning SAM using LoRA.")
parser.add_argument("--cfg", required=True, type=str, help="Path to configuration file.")
args = parser.parse_args()

# Load the config file
logging.info("Loading configuration file...")
with open(args.cfg, "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SAM model
logging.info(f"Loading SAM checkpoint: {config_file['SAM']['CHECKPOINT']}...")
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])

# Create SAM LoRA
logging.info(f"Initializing LoRA for rank: {config_file['SAM']['RANK']}...")
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
model = sam_lora.sam

# Process the dataset
# processor = Samprocessor(model)

# Load the dataset
logging.info(f"Initializing dataset...")
transform = Compose([
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_dataset = SegmentationDataset(config_file["DATASET"]["TRAIN_IMAGES"], config_file["DATASET"]["TRAIN_MASKS"], transform=transform, sam=True)
val_dataset = SegmentationDataset(config_file["DATASET"]["VAL_IMAGES"], config_file["DATASET"]["VAL_MASKS"], transform=transform, sam=True)

# Create a dataloader
train_dataloader = DataLoader(train_dataset, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
# Initialize optimize and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=config_file["TRAIN"]["LEARNING_RATE"], weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

# Set model to train and into the device
model.train()
model.to(device)

total_loss = []

logging.info("TRAINING phase")
for epoch in range(num_epochs):
    epoch_losses = []

    for i, batch in enumerate(tqdm(train_dataloader)):
      #print(batch)
      outputs = model(batched_input=batch,
                      multimask_output=False)

      stk_gt, stk_out = utils.stacking_batch(batch, outputs)
      stk_out = stk_out.squeeze(1)
      #stk_gt = stk_gt.unsqueeze(1) # We need to get the [B, C, H, W] starting from [H, W]
      
      loss = seg_loss(stk_out, stk_gt)#.float().to(device))
      
      optimizer.zero_grad()
      loss.backward()
      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    #print(f'EPOCH: {epoch}')
    #print(f'Mean loss training: {mean(epoch_losses)}')
    logging.info(f"\tEpoch [{epoch+1}/{epoch}] Mean Loss: {mean(epoch_losses):.4f}")

# Save the parameters of the model in safetensors format
#rank = config_file["SAM"]["RANK"]
#sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
logging.info("Saving checkpoint...")
save_checkpoint(model, optimizer, num_epochs)
