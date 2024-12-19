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
###################################

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
    cptName = "SAM_"+cptPath.stem+"_e"+str(epoch)+".pth"
    logging.info("Saving checkpoint: {cptName}")
    
    cptDir = Path("models")
    cptDir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, cptDir/cptName)

def plot_results(sam_model, val_loader, epoch):
    images, masks, points, labels = next(iter(val_loader))  # Get a random batch
    images, masks, points, labels = images.to(device), masks.to(device), points.to(device), labels.to(device)
    
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(images)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=(points, labels),
            #labels=labels,
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
    preds = torch.sigmoid(upscaled_masks).float()

    batch_size = images.size(0)
    rows = batch_size
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))

    for idx in range(batch_size):
        image = images[idx].cpu().detach().permute(1, 2, 0).numpy() * 0.5 + 0.5
        image = np.clip(image, 0, 1)
        
        mask = masks[idx].cpu().detach().numpy().squeeze()
        pred_mask = (preds[idx].cpu().detach().numpy().squeeze() > 0.5).astype(np.uint8)

        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title("Input Image")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(mask, cmap="gray")
        axes[idx, 1].set_title("Ground Truth Mask")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(pred_mask, cmap="gray")
        axes[idx, 2].set_title("Predicted Mask")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"results_batch_e{epoch+1}.png")
 
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

## Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}...")

###################################
## Load SAM model
    
logging.info(f"Loading SAM checkpoint: {cfg['SAM']['CHECKPOINT']}...")
# Load original SAM checkpoint
if cfg['SAM']['ORIG']:
    sam_model = sam_model_registry[cfg['SAM']['CHECKPOINT_TYPE']](checkpoint=cfg['SAM']['CHECKPOINT'])
    
# Set up model part for finetuning
if cfg['SAM']['FINETUNE']['MASK_DECODER']:
    model_part=sam_model.mask_decoder
optimizer = torch.optim.AdamW(model_part.parameters(), lr=cfg['TRAIN']['LEARNING_RATE'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])

# Load pretrained checkpoint
if not cfg['SAM']['ORIG']:
    checkpoint = torch.load(cfg['SAM']['CHECKPOINT'], map_location=device)
    sam_model = sam_model_registry[cfg['SAM']['CHECKPOINT_TYPE']]()
    sam_model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
sam_model.to(device)

logging.info("Finetuning setup")
logging.info(f"\tImage encoder:  \t{cfg['SAM']['FINETUNE']['IMAGE_ENCODER']}")
logging.info(f"\tPrompt encoder: \t{cfg['SAM']['FINETUNE']['PROMPT_ENCODER']}")
logging.info(f"\tMask decoder:   \t{cfg['SAM']['FINETUNE']['MASK_DECODER']}")
logging.info("")
logging.info("Finetuning params")
logging.info(f"\tLearning rate:  \t{cfg['TRAIN']['LEARNING_RATE']}")
logging.info(f"\tWeight decay:   \t{cfg['TRAIN']['WEIGHT_DECAY']}")
logging.info("")
###################################

# Dataset
logging.info(f"Initializing dataset...")
transform = Compose([
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
#mp.set_start_method('spawn', force=True)
target_size = (cfg['TRAIN']['IMAGE_SIZE'], cfg['TRAIN']['IMAGE_SIZE'])
persistent_workers = True if cfg['TRAIN']['NUM_WORKERS'] > 0 else False

train_dataset = SegmentationDataset(cfg['DATASET']['TRAIN_IMAGES'], cfg['DATASET']['TRAIN_MASKS'], transform=transform, target_size=target_size, sam=sam_model)
val_dataset = SegmentationDataset(cfg['DATASET']['VAL_IMAGES'], cfg['DATASET']['VAL_MASKS'], transform=transform, target_size=target_size, sam=sam_model)
train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=cfg['TRAIN']['NUM_WORKERS'], persistent_workers=persistent_workers)
val_loader = DataLoader(val_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=False, num_workers=cfg['TRAIN']['NUM_WORKERS'], persistent_workers=persistent_workers)
#val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg['TRAIN']['NUM_WORKERS'], persistent_workers=persistent_workers)
print(f"train_loader: {len(train_loader)}")
###################################

    
loss = DiceLoss()
metric = IoU()

#############
# DEBUG
for param in sam_model.image_encoder.parameters():
    param.requires_grad = cfg['SAM']['FINETUNE']['IMAGE_ENCODER']
for param in sam_model.prompt_encoder.parameters():
    param.requires_grad = cfg['SAM']['FINETUNE']['PROMPT_ENCODER']
for param in sam_model.mask_decoder.parameters():
    param.requires_grad = cfg['SAM']['FINETUNE']['MASK_DECODER']
    
#for name, param in model_part.named_parameters():
#    print(name, param.requires_grad)
#############

logging.info("TRAINING phase")
for epoch in range(cfg['TRAIN']['EPOCHS']):
    sam_model.train()
    train_loss_epoch = []
    loss_batch = 0
    
    #with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['TRAIN']['EPOCHS']}", unit="batch") as train_bar:
    for images, masks, points, labels in train_loader:
        images, masks, points, labels = images.to(device), masks.to(device), points.to(device), labels.to(device)
        #print(f"images:{images.shape}\nmasks:{masks.shape}\nbox:{box.shape}")
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(images)
        
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=(points, labels),
                #labels=labels,
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
        
        loss_value = loss(preds, masks)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        train_loss_epoch.append(loss_value.item())
        
        loss_batch = loss_value.item()
            #train_bar.set_postfix(loss=f"{loss_batch:.4f}")

    train_loss_avg = mean(train_loss_epoch)
    
    # Validation
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
                    #labels=labels,
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
    plot_results(sam_model, val_loader, epoch)
    logging.info(f"\tEpoch [{epoch+1}/{cfg['TRAIN']['EPOCHS']}] Train Loss: {train_loss_avg:.4f} | Val loss: {val_loss_avg:.4f} | IoU: {val_iou_avg:.4f}")
   
   
    if (epoch+1) % 5 == 0:
        save_checkpoint(sam_model, optimizer, epoch+1)

# Save model checkpoint
save_checkpoint(sam_model, optimizer, cfg['TRAIN']['EPOCHS'])
