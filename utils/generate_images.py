import os
import torch
import argparse
from seg_dataset_sam import SegmentationDataset
from torchvision.transforms import ToTensor, Normalize, Compose
from segmentation_models_pytorch.utils.metrics import IoU, Fscore, Accuracy, Precision, Recall
from tqdm import tqdm
from rich.logging import RichHandler
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

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

def stepSAM(sam_model, images, points, labels, target_size):
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
    preds = torch.sigmoid(upscaled_masks).float()
    return preds

def stepUnet(model, images, masks):
    images = images.to(device)
    masks = masks.to(device)

    predictions = model(images)
    predictions = torch.sigmoid(predictions) > 0.5
    return predictions
    
def plot_results(sam_model, unet_model, val_loader, epoch, device, target_size):
    images, masks, points, labels = next(iter(val_loader))  # Get a random batch
    images, masks, points, labels = images.to(device), masks.to(device), points.to(device), labels.to(device)
    
    with torch.no_grad():
        predsSAM = stepSAM(sam_model, images, points, labels, target_size)
        predsUnet = stepUnet(unet_model, images, masks)
        

    batch_size = images.size(0)
    rows = 4
    cols = batch_size

    # Nastavení figure a mřížky
    fig = plt.figure(figsize=(3 * cols, 15))
    gs = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.0)  # Bez mezer mezi subplots

    # Popisky vlevo (první sloupec)
    row_titles = ["Input Image", "Ground Truth", "Unet", "SAM"]

    # Obrázky
    for idx in range(batch_size):
        image = images[idx].cpu().detach().permute(1, 2, 0).numpy() * 0.5 + 0.5
        image = np.clip(image, 0, 1)

        mask = masks[idx].cpu().detach().numpy().squeeze()
        maskUnet = (predsUnet[idx].cpu().detach().numpy().squeeze() > 0.5).astype(np.uint8)
        maskSAM = (predsSAM[idx].cpu().detach().numpy().squeeze() > 0.5).astype(np.uint8)

        # Přidání obrázků do gridu
        for row, img in enumerate([image, mask, maskUnet, maskSAM]):
            ax = fig.add_subplot(gs[row, idx])
            if idx == 0:  # Pouze pro první sloupec
                ax.text(-0.2, 0.5, row_titles[row], fontsize=30, ha="right", va="center", 
                        rotation=90, transform=ax.transAxes)
            if row == 0:
                ax.imshow(img)  # RGB obrázek
            else:
                ax.imshow(img, cmap="gray")  # Maska
            ax.axis("off")  # Vypnutí os

    # Úplné odstranění paddingu
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Uložit obrázek bez paddingu
    plt.savefig(f"results_transposed_batch_e{epoch+1}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


#device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_Unet(model_path):
    # --------- UNet -------------------------------
    from segmentation_models_pytorch import Unet
    model = Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model

def load_SAM(model_path):
    # ----------- SAM ----------------
    from sam.segment_anything_ori import sam_model_registry
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    model = sam_model_registry["vit_b"]()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Generate outputs from Unet and SAM models.")
    parser.add_argument("--unet_model", required=True, help="Filepath to the Unet segmentation model.")
    parser.add_argument("--sam_model", required=True, help="Filepath to the SAM segmentation model.")
    parser.add_argument("--data", required=True, help="Filepath to the evaluation dataset (images and labels directories).")
    parser.add_argument("--batch_size", required=False, default=1, type=int, help="Specify the architecture of model for evaluation (default: Unet).")

    args = parser.parse_args()
    
    # ------------------ Load val dataset ----------------
    image_dir = os.path.join(args.data, "images")
    label_dir = os.path.join(args.data, "labels")
    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        raise ValueError("Evaluation dataset must contain 'images' and 'labels' directories.")

    logging.info("Evaluation Parameters:")
    logging.info(f"Evaluation data:  {args.data}")
    
    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Same as in train.py
    ])

    # Load model
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=label_dir, transform=transform, sam=False)
    unet_model = load_Unet(args.unet_model)
    unet_model.to(device)
    unet_model.eval()
        
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=label_dir, transform=transform, target_size=(1024,1024), sam=True)
    sam_model = load_SAM(args.sam_model)
    sam_model.to(device)
    sam_model.eval()
    
    logging.info("Model loaded successfully...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logging.info("Generating outputs...")
    
    for epoch in range(20):
        plot_results(sam_model, unet_model, dataloader, epoch, device, (1024,1024))
    

if __name__ == "__main__":
    main()
