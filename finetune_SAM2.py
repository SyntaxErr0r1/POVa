import torch
import logging
from rich.logging import RichHandler
from pathlib import Path
import argparse
import yaml
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from segmentation_models_pytorch.utils.losses import DiceLoss
from torch.nn.functional import threshold, normalize
from tqdm import tqdm
import torch.multiprocessing as mp

## SAM Model imports
from sam.segment_anything_ori.build_sam import sam_model_registry

## Dataset imports
from seg_dataset import SegmentationDataset
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
    cptName = "SAM_"+cptPath.stem+".pth"
    
    cptDir = Path("models")
    cptDir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, cptDir/cptName)
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
# Dataset
logging.info(f"Initializing dataset...")
transform = Compose([
    ToTensor()
])
#mp.set_start_method('spawn', force=True)
target_size = (cfg['TRAIN']['IMAGE_SIZE'], cfg['TRAIN']['IMAGE_SIZE'])
train_dataset = SegmentationDataset(cfg['DATASET']['TRAIN_IMAGES'], cfg['DATASET']['TRAIN_MASKS'], transform=transform, target_size=target_size, sam=True)
val_dataset = SegmentationDataset(cfg['DATASET']['VAL_IMAGES'], cfg['DATASET']['VAL_MASKS'], transform=transform, target_size=target_size, sam=True)
train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=cfg['TRAIN']['NUM_WORKERS'], persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=False, num_workers=cfg['TRAIN']['NUM_WORKERS'], persistent_workers=True)

###################################
## Load SAM model
logging.info(f"Loading SAM checkpoint: {cfg['SAM']['CHECKPOINT']}...")
sam_model = sam_model_registry[cfg['SAM']['CHECKPOINT_TYPE']](checkpoint=cfg['SAM']['CHECKPOINT'])
sam_model.to(device)

logging.info("Finetuning setup")
logging.info(f"\tImage encoder:  \t{cfg['SAM']['FINETUNE']['IMAGE_ENCODER']}")
logging.info(f"\tPrompt encoder: \t{cfg['SAM']['FINETUNE']['PROMPT_ENCODER']}")
logging.info(f"\tMask decoder:   \t{cfg['SAM']['FINETUNE']['MASK_DECODER']}")

if cfg['SAM']['FINETUNE']['MASK_DECODER']:
    model_part=sam_model.mask_decoder
    
optimizer = torch.optim.Adam(model_part.parameters())
loss = DiceLoss()

logging.info("TRAINING phase")
for epoch in range(cfg['TRAIN']['EPOCHS']):
    sam_model.train()
    train_loss_total = 0
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['TRAIN']['EPOCHS']}", unit="batch") as train_bar:
    #for images, masks, box in train_loader:
        for images, masks, box in train_bar:
            images, masks, box = images.to(device), masks.to(device), box.to(device)
            #print(f"images:{images.shape}\nmasks:{masks.shape}\nbox:{box.shape}")
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(images)
            
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box,
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
            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)
            
            loss_value = loss(binary_mask, masks)  # Compute loss
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            train_loss_total += loss_value.item()

    train_loss_avg = train_loss_total / len(train_loader)

    # Validation
    sam_model.eval()
    val_loss_total = 0
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg['TRAIN']['EPOCHS']}", unit="batch") as val_bar:
            for images, masks, box in val_bar:
                images, masks, box = images.to(device), masks.to(device), box.to(device)
                image_embedding = sam_model.image_encoder(images)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box,
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
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)
                
                val_loss = loss(binary_mask, masks)
                val_loss_total += val_loss.item()

    val_loss_avg = val_loss_total / len(val_loader)
    
    logging.info(f"\tEpoch [{epoch+1}/{epoch}] Train Loss: {train_loss_avg:.4f} | Val loss: {val_loss_avg:.4f}")

# Save model checkpoint
save_checkpoint(sam_model, optimizer, cfg['TRAIN']['EPOCHS'])
