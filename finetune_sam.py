import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from rich.logging import RichHandler
from pathlib import Path
import argparse
import yaml
from statistics import mean
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from torch.nn.functional import threshold, normalize
from tqdm import tqdm
import torch.multiprocessing as mp

## SAM Model imports
from sam.segment_anything_ori import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

## Dataset imports
from seg_dataset_sam import SegmentationDataset

## Libs imports
from libs.save_checkpoint import *
from libs.plot_results import *

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
## Parse the arguments
parser = argparse.ArgumentParser(description="Script for finetuning SAM using LoRA.")
parser.add_argument("--cfg", required=True, type=str, help="Path to configuration file.")
args = parser.parse_args()

###################################
## Load the config file
logging.info("Loading configuration file...")
with open(args.cfg, "r") as ymlfile:
   cfg = yaml.load(ymlfile, Loader=yaml.Loader)

logging.info("LOGGING INFO")
logging.info(f"Configuration file: {args.cfg}")
logging.info("Finetuning setup")
logging.info(f"\tImage encoder:  \t{cfg['SAM']['FINETUNE']['IMAGE_ENCODER']['ENABLED']}")
logging.info(f"\t\tLearning rate: {cfg['SAM']['FINETUNE']['IMAGE_ENCODER']['LEARNING_RATE']}")
logging.info(f"\t\tWeight decay:  {cfg['SAM']['FINETUNE']['IMAGE_ENCODER']['WEIGHT_DECAY']}")
logging.info(f"\tPrompt encoder: \t{cfg['SAM']['FINETUNE']['PROMPT_ENCODER']['ENABLED']}")
logging.info(f"\tMask decoder:   \t{cfg['SAM']['FINETUNE']['MASK_DECODER']['ENABLED']}")
logging.info(f"\t\tLearning rate: {cfg['SAM']['FINETUNE']['MASK_DECODER']['LEARNING_RATE']}")
logging.info(f"\t\tWeight decay:  {cfg['SAM']['FINETUNE']['MASK_DECODER']['WEIGHT_DECAY']}")
logging.info("")
logging.info(f"Dataset")
logging.info(f"\tTRAIN: {cfg['DATASET']['TRAIN_IMAGES']}")
logging.info(f"\tVAL:   {cfg['DATASET']['VAL_IMAGES']}")
logging.info("")
###################################
## Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}...")

###################################
## Load SAM checkpoint

logging.info(f"Loading SAM checkpoint: {cfg['SAM']['CHECKPOINT']}...")

# ------ Load original SAM checkpoint
if cfg['SAM']['ORIG']:
    optimizer_config = []
    sam_model = sam_model_registry[cfg['SAM']['CHECKPOINT_TYPE']](checkpoint=cfg['SAM']['CHECKPOINT'])

    ## MASK DECODER finetuning
    if cfg['SAM']['FINETUNE']['MASK_DECODER']['ENABLED']:
        logging.debug(f"Setting mask decoder as finetuning model part...")
        optimizer_config.append({'params': sam_model.mask_decoder.parameters(), "lr": cfg['SAM']['FINETUNE']['MASK_DECODER']['LEARNING_RATE'], "weight_decay": cfg['SAM']['FINETUNE']['MASK_DECODER']['WEIGHT_DECAY']})

    ## IMAGE ENCODER finetuning
    if cfg['SAM']['FINETUNE']['IMAGE_ENCODER']['ENABLED']:
        logging.debug(f"Setting image encoder as finetuning model part...")
        optimizer_config.append({'params': sam_model.image_encoder.parameters(), "lr": cfg['SAM']['FINETUNE']['IMAGE_ENCODER']['LEARNING_RATE'], "weight_decay": cfg['SAM']['FINETUNE']['IMAGE_ENCODER']['WEIGHT_DECAY']})
    
    optimizer = torch.optim.AdamW(optimizer_config)

# ------ Continue learning from previous checkpoint
# Load pretrained checkpoint
#if not cfg['SAM']['ORIG']:
#    checkpoint = torch.load(cfg['SAM']['CHECKPOINT'], map_location=device)
#    sam_model = sam_model_registry[cfg['SAM']['CHECKPOINT_TYPE']]()
#    sam_model.load_state_dict(checkpoint['model_state_dict'])
#    if 'optimizer_state_dict' in checkpoint:
#        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

## Set model to device
sam_model.to(device)

###################################
## Dataset initialization
logging.info(f"Initializing dataset...")
transform = Compose([
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

target_size = (cfg['TRAIN']['IMAGE_SIZE'], cfg['TRAIN']['IMAGE_SIZE'])
persistent_workers = True if cfg['TRAIN']['NUM_WORKERS'] > 0 else False

train_dataset = SegmentationDataset(cfg['DATASET']['TRAIN_IMAGES'], cfg['DATASET']['TRAIN_MASKS'], transform=transform, target_size=target_size, sam=sam_model)
val_dataset = SegmentationDataset(cfg['DATASET']['VAL_IMAGES'], cfg['DATASET']['VAL_MASKS'], transform=transform, target_size=target_size, sam=sam_model)
train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=cfg['TRAIN']['NUM_WORKERS'], persistent_workers=persistent_workers)
val_loader = DataLoader(val_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=False, num_workers=cfg['TRAIN']['NUM_WORKERS'], persistent_workers=persistent_workers)

logging.debug(f"train_loader: {len(train_loader)}")
###################################
## Training params setup
loss = DiceLoss()
metric = IoU()

###################################
# For each part of the model, set requires_grad to True or False
for param in sam_model.image_encoder.parameters():
    param.requires_grad = cfg['SAM']['FINETUNE']['IMAGE_ENCODER']['ENABLED']
for param in sam_model.prompt_encoder.parameters():
    param.requires_grad = cfg['SAM']['FINETUNE']['PROMPT_ENCODER']['ENABLED']
for param in sam_model.mask_decoder.parameters():
    param.requires_grad = cfg['SAM']['FINETUNE']['MASK_DECODER']['ENABLED']
    
#for name, param in model_part.named_parameters():
#    logging.debug(f"{name} {param.requires_grad}")
#    break
###################################

logging.info("TRAINING phase")
for epoch in range(cfg['TRAIN']['EPOCHS']):
    sam_model.train()
    train_loss_epoch = []
    loss_batch = 0
    
    # ------------------ Training ------------------
    #with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['TRAIN']['EPOCHS']}", unit="batch") as train_bar:
    for images, masks, points, labels in train_loader:
        
        images, masks, points, labels = images.to(device), masks.to(device), points.to(device), labels.to(device)
        
        if cfg['SAM']['FINETUNE']['IMAGE_ENCODER']['ENABLED']:
            # If finetuning image encoder all other parts must be hot
            image_embedding = sam_model.image_encoder(images)
        else:
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(images)
        
        if cfg['SAM']['FINETUNE']['IMAGE_ENCODER']['ENABLED']:
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=(points, labels),
                    boxes=None,
                    masks=None,
                )
        else:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=(points, labels),
                    boxes=None,
                    masks=None,
                )
        if cfg['SAM']['FINETUNE']['MASK_DECODER']['ENABLED'] or cfg['SAM']['FINETUNE']['IMAGE_ENCODER']['ENABLED']:
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                )
        else:
            with torch.no_grad():
                low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                )
        
        logging.debug(f"[Image embedding] requires_grad: {image_embedding.requires_grad}")
        logging.debug(f"[Sparse embeddings] requires_grad: {sparse_embeddings.requires_grad}")
        logging.debug(f"[Low res masks] requires_grad: {low_res_masks.requires_grad}")
        
        upscaled_masks = sam_model.postprocess_masks(low_res_masks, target_size, target_size).to(device)
        preds = torch.sigmoid(upscaled_masks).to(device)
        
        loss_value = loss(preds, masks)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        train_loss_epoch.append(loss_value.item())
        
        loss_batch = loss_value.item()
        #train_bar.set_postfix(loss=f"{loss_batch:.4f}")

    train_loss_avg = mean(train_loss_epoch)
    
    # ------------------ Validation ------------------
    sam_model.eval()
    val_loss_epoch = []
    val_iou_epoch = []
    with torch.no_grad():
        #with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg['TRAIN']['EPOCHS']}", unit="batch") as val_bar:
            for images, masks, points, labels in val_loader:
                images, masks, points, labels = images.to(device), masks.to(device), points.to(device), labels.to(device)
                image_embedding = sam_model.image_encoder(images)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=(points, labels),
                    boxes=None,
                    masks=None,
                )
                low_res_masks, iou_predictions = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                upscaled_masks = sam_model.postprocess_masks(low_res_masks, target_size, target_size).to(device)
                preds = torch.sigmoid(upscaled_masks).to(device)
                
                val_loss = loss(preds, masks)
                val_iou = metric(preds, masks)
                val_loss_epoch.append(val_loss.item())
                val_iou_epoch.append(val_iou.item())

    val_loss_avg = mean(val_loss_epoch)
    val_iou_avg = mean(val_iou_epoch)
    plot_results(sam_model, val_loader, epoch, device, target_size)
    logging.info(f"\tEpoch [{epoch+1}/{cfg['TRAIN']['EPOCHS']}] Train Loss: {train_loss_avg:.4f} | Val loss: {val_loss_avg:.4f} | IoU: {val_iou_avg:.4f}")
   
   
    if (epoch+1) % 5 == 0:
        save_checkpoint(args, sam_model, optimizer, epoch+1)

# Save model checkpoint
save_checkpoint(args, sam_model, optimizer, cfg['TRAIN']['EPOCHS'])
