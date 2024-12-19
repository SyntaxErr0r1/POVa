import os
import torch
import argparse
from seg_dataset import SegmentationDataset
from torchvision.transforms import ToTensor, Normalize, Compose
from segmentation_models_pytorch.utils.metrics import IoU, Fscore, Accuracy, Precision, Recall
from tqdm import tqdm


def load_model(model_path):
    # --------- UNet -------------------------------
    from segmentation_models_pytorch import Unet
    model = Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # ----------- TODO Other models ----------------
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model

def evaluate_model(model, dataloader):
    model.eval()
    device = next(model.parameters()).device

    # Initialize metrics
    total_fscore = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_iou = 0
    num_batches = 0

    # Define metric functions
    fscore_metric = Fscore()
    accuracy_metric = Accuracy()
    precision_metric = Precision()
    recall_metric = Recall()
    iou_metric = IoU()

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            predictions = torch.sigmoid(predictions) > 0.5

            # Accumulate scores for all metrics
            total_fscore += fscore_metric(predictions, labels).item()
            total_accuracy += accuracy_metric(predictions, labels).item()
            total_precision += precision_metric(predictions, labels).item()
            total_recall += recall_metric(predictions, labels).item()
            total_iou += iou_metric(predictions, labels).item()
            num_batches += 1

    # Calculate average scores across batches
    avg_fscore = total_fscore / num_batches if num_batches > 0 else 0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
    avg_precision = total_precision / num_batches if num_batches > 0 else 0
    avg_recall = total_recall / num_batches if num_batches > 0 else 0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0

    return {
        "Fscore": avg_fscore,
        "Accuracy": avg_accuracy,
        "Precision": avg_precision,
        "Recall": avg_recall,
        "IoU": avg_iou
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a binary segmentation model.")
    parser.add_argument("-model", required=True, help="Filepath to the segmentation model.")
    parser.add_argument("-data", required=True, help="Filepath to the evaluation dataset (images and labels directories).")

    args = parser.parse_args()

    # ------------------ Load val dataset ----------------
    image_dir = os.path.join(args.data, "images")
    label_dir = os.path.join(args.data, "labels")
    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        raise ValueError("Evaluation dataset must contain 'images' and 'labels' directories.")
    
    print("Evaluating model", args.model ,"on dataset", args.data)

    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Same as in train.py
    ])

    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=label_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = load_model(args.model)

    score = evaluate_model(model, dataloader)

    print("Evaluation Scores:")
    for metric, value in score.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()