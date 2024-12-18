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
from seg_dataset_SAM22 import SegmentationDataset
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

 
def plot_results(sam_model, val_loader):
    images, masks, box = next(iter(val_loader))  # Get a random batch
    images, masks, box = images.to(device), masks.to(device), box.to(device)
    
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(images)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
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
    binary_mask = (torch.sigmoid(upscaled_masks) > 0.5).float()  # Binary mask (values 0 or 1)
    print(f"iou_predictions={iou_predictions}")

    # Select a random image from the batch
    random_idx = np.random.randint(0, images.size(0))
    image = images[random_idx].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5  # [H, W, C]
    image = np.clip(image, 0, 1)
    # Denormalizace na rozsah [0, 1]
    #image = (image - image.min()) / (image.max() - image.min())
    
    #image = image.convert("RGB")

    # Ground truth and predicted mask (binary)
    mask = masks[random_idx].cpu().detach().numpy().squeeze()
    print(f"mask: {mask}")
    pred_mask = (binary_mask[random_idx].cpu().detach().numpy().squeeze() > 0.5).astype(np.uint8)
    print(f"pred_mask: {pred_mask}")

    print(f"Image shape: {image.shape}")
    print(f"Binary mask shape: {binary_mask[random_idx].shape}")
    print(f"Ground truth mask shape: {masks[random_idx].shape}")

    # Plot
    plt.figure(figsize=(12, 4))
    
    # Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    # Ground Truth Mask (binary)
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="binary")
    plt.title("Ground Truth Mask (Binary)")
    plt.axis("off")

    # Predicted Mask (binary)
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="binary")
    plt.title("Predicted Mask (Binary)")
    plt.axis("off")

    plt.savefig("results.png")
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
sam_model = sam_model_registry[cfg['SAM']['CHECKPOINT_TYPE']](checkpoint=cfg['SAM']['CHECKPOINT'])
sam_model.to(device)

logging.info("Finetuning setup")
logging.info(f"\tImage encoder:  \t{cfg['SAM']['FINETUNE']['IMAGE_ENCODER']}")
logging.info(f"\tPrompt encoder: \t{cfg['SAM']['FINETUNE']['PROMPT_ENCODER']}")
logging.info(f"\tMask decoder:   \t{cfg['SAM']['FINETUNE']['MASK_DECODER']}")
logging.info("----------")
logging.info("Finetuning params")
logging.info(f"\tLearning rate:  \t{cfg['TRAIN']['LEARNING_RATE']}")
logging.info(f"\tWeight decay:   \t{cfg['TRAIN']['WEIGHT_DECAY']}")
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

if cfg['SAM']['FINETUNE']['MASK_DECODER']:
    model_part=sam_model.mask_decoder
    
optimizer = torch.optim.AdamW(model_part.parameters(), lr=cfg['TRAIN']['LEARNING_RATE'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
loss = DiceLoss()
metric = IoU()

logging.info("TRAINING phase")
for epoch in range(cfg['TRAIN']['EPOCHS']):
    sam_model.train()
    train_loss_epoch = []
    loss_batch = 0
    plot_results(sam_model, val_loader)
    break
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
            binary_mask = normalize(threshold(torch.sigmoid(upscaled_masks), 0.5, 0)).to(device)
            
            loss_value = loss(binary_mask, masks)  # Compute loss
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            train_loss_epoch.append(loss_value.item())
            
            loss_batch = loss_value.item()
            train_bar.set_postfix(loss=f"{loss_batch:.4f}")
            print(f"iou_predictions={iou_predictions}")

    train_loss_avg = mean(train_loss_epoch)
    
    # Validation
    sam_model.eval()
    val_loss_epoch = []
    val_iou_epoch = []
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg['TRAIN']['EPOCHS']}", unit="batch") as val_bar:
            for images, masks, box in val_bar:
                images, masks, box = images.to(device), masks.to(device), box.to(device)
                image_embedding = sam_model.image_encoder(images)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
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
                #binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)
                binary_mask = normalize(threshold(torch.sigmoid(upscaled_masks), 0.5, 0)).to(device)
                
                val_loss = loss(binary_mask, masks)
                val_iou = metric(binary_mask, masks)
                val_loss_epoch.append(val_loss.item())
                val_iou_epoch.append(val_iou.item())

    val_loss_avg = mean(val_loss_epoch)
    val_iou_avg = mean(val_iou_epoch)
    plot_results(sam_model, val_loader)
    logging.info(f"\tEpoch [{epoch+1}/{cfg['TRAIN']['EPOCHS']}] Train Loss: {train_loss_avg:.4f} | Val loss: {val_loss_avg:.4f} | IoU: {val_iou_avg:.4f}")
    """
    ## SamPredictor
    predictor_tuned = SamPredictor(sam_model)
    val_loss_epoch = []
    val_iou_epoch = []
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg['TRAIN']['EPOCHS']}", unit="batch") as val_bar:
            for images, masks, box in val_bar:
                images, masks, box = images.to(device), masks.to(device), box.to(device)
                print(f"images_tuned: {images.shape}")
                print(f"images_tuned: {images.squeeze(0).shape}")
                print(f"images_tuned: {images.squeeze(0).permute(1, 2, 0).shape}")
                predictor_tuned.set_image(images.squeeze(0))

                masks_tuned, _, _ = predictor_tuned.predict(
                    point_coords=None,
                    box=None,
                    multimask_output=False,
                )
                
                print(f"masks_tuned: {masks_tuned.shape}")
                print(f"masks_tuned: {masks_tuned}")
                
                masks_tuned1 = torch.as_tensor(masks_tuned > 0, dtype=torch.float32)
                new_tensor = masks_tuned1.unsqueeze(0).to(device)
                
                print(f"masks_tuned: {new_tensor.shape}")
                print(f"masks_tuned: {new_tensor}")
                
                val_loss = loss(new_tensor, masks.squeeze(0))
                val_iou = metric(new_tensor, masks.squeeze(0))
                val_loss_epoch.append(val_loss.item())
                val_iou_epoch.append(val_iou.item())
    """
    """
    predictor_tuned = SamAutomaticMaskGenerator(sam_model)
    val_loss_epoch = []
    val_iou_epoch = []
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg['TRAIN']['EPOCHS']}", unit="batch") as val_bar:
            for images, masks, box in val_bar:
                images, masks, box = images.to(device), masks.to(device), box.to(device)
                print(f"images_tuned: {images.shape}")
                print(f"images_tuned: {images.squeeze(0).shape}")
                predictor_tuned.set_image(images.squeeze(0))

                masks_tuned, _, _ = predictor_tuned.predict(
                    point_coords=None,
                    box=None,
                    multimask_output=False,
                )
                
                print(f"masks_tuned: {masks_tuned.shape}")
                print(f"masks_tuned: {masks_tuned}")
                
                val_loss = loss(masks_tuned, masks.squeeze(0))
                val_iou = metric(masks_tuned, masks.squeeze(0))
                val_loss_epoch.append(val_loss.item())
                val_iou_epoch.append(val_iou.item())
    
    val_loss_avg = mean(val_loss_epoch)
    val_iou_avg = mean(val_iou_epoch)
    
    logging.info(f"\tEpoch [{epoch+1}/{cfg['TRAIN']['EPOCHS']}] Train Loss: {train_loss_avg:.4f} | Val loss: {val_loss_avg:.4f} | IoU: {val_iou_avg:.4f}")
    """

# Save model checkpoint
save_checkpoint(sam_model, optimizer, cfg['TRAIN']['EPOCHS'])
