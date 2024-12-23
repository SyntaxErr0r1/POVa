import os
import torch
import argparse
from seg_dataset_sam import SegmentationDataset
from torchvision.transforms import ToTensor, Normalize, Compose
from segmentation_models_pytorch.utils.metrics import IoU, Fscore, Accuracy, Precision, Recall
from tqdm import tqdm
from rich.logging import RichHandler
import logging

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

# Define metric functions
fscore_metric = Fscore()
accuracy_metric = Accuracy()
precision_metric = Precision()
recall_metric = Recall()
iou_metric = IoU()

# Initialize metrics
total_scores = {
    "total_fscore": 0.0,
    "total_accuracy": 0.0,
    "total_precision": 0.0,
    "total_recall": 0.0,
    "total_iou": 0.0,
    "num_batches": 0.0
}

#device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_Unet(model_path):
    # --------- UNet -------------------------------
    from segmentation_models_pytorch import Unet
    model = Unet(encoder_name="efficientnet-b5", encoder_weights=None, in_channels=3, classes=1, decoder_attention_type="scse")
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

def accumulate_scores(predictions, labels):
    total_scores["total_fscore"]    += fscore_metric(predictions, labels).item()
    total_scores["total_accuracy"]  += accuracy_metric(predictions, labels).item()
    total_scores["total_precision"] += precision_metric(predictions, labels).item()
    total_scores["total_recall"]    += recall_metric(predictions, labels).item()
    total_scores["total_iou"]       += iou_metric(predictions, labels).item()
    total_scores["num_batches"]     += 1

def infer_Unet(model, dataloader, device):
    for images, masks, *_ in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        predictions = model(images)
        predictions = torch.sigmoid(predictions) > 0.5

        # Accumulate scores for all metrics
        accumulate_scores(predictions, masks)

def infer_SAM(sam_model, dataloader, device):
    for images, masks, points, labels in tqdm(dataloader):
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
        upscaled_masks = sam_model.postprocess_masks(low_res_masks, (1024,1024), (1024,1024)).to(device)
        predictions = torch.sigmoid(upscaled_masks).to(device)
        
        # Accumulate scores for all metrics
        accumulate_scores(predictions, masks)

def evaluate_model(arch, model, dataloader):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        if arch == "Unet":
            infer_Unet(model, dataloader, device)
        elif arch == "SAM":
            infer_SAM(model, dataloader, device)

    # Calculate average scores across batches
    avg_fscore      = total_scores["total_fscore"]      / total_scores["num_batches"] if total_scores["num_batches"] > 0 else 0
    avg_accuracy    = total_scores["total_accuracy"]    / total_scores["num_batches"] if total_scores["num_batches"] > 0 else 0
    avg_precision   = total_scores["total_precision"]   / total_scores["num_batches"] if total_scores["num_batches"] > 0 else 0
    avg_recall      = total_scores["total_recall"]      / total_scores["num_batches"] if total_scores["num_batches"] > 0 else 0
    avg_iou         = total_scores["total_iou"]         / total_scores["num_batches"] if total_scores["num_batches"] > 0 else 0

    return {
        "Fscore": avg_fscore,
        "Accuracy": avg_accuracy,
        "Precision": avg_precision,
        "Recall": avg_recall,
        "IoU": avg_iou
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a binary segmentation model.")
    parser.add_argument("--arch", choices=["Unet", "SAM"], default="Unet", help="Specify the architecture of model for evaluation (default: Unet).")
    parser.add_argument("--model", required=True, help="Filepath to the segmentation model.")
    parser.add_argument("--data", required=True, help="Filepath to the evaluation dataset (images and labels directories).")
    parser.add_argument("--batch_size", required=False, default=1, type=int, help="Specify the architecture of model for evaluation (default: Unet).")

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
        model = load_Unet(args.model)
    elif args.arch == "SAM":
        dataset = SegmentationDataset(image_dir=image_dir, mask_dir=label_dir, transform=transform, target_size=(1024,1024), sam=True)
        model = load_SAM(args.model)
    else:
        raise ValueError("Invalid model architecture specified.")
    
    logging.info("Model loaded successfully...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logging.info("Starting evaluation...")
    score = evaluate_model(args.arch, model, dataloader)

    print("Evaluation Scores:")
    for metric, value in score.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
