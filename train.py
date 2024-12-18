import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU, Fscore
from torchvision.transforms import ToTensor, Normalize, Compose
from tqdm import tqdm
from time import gmtime, strftime
import time
import json
from torch.utils.tensorboard import SummaryWriter

from seg_dataset import SegmentationDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:", torch.cuda.get_device_name("cuda") if torch.cuda.is_available() else "CPU")

def get_dataloader(dataset_path, batch_size=8, shuffle=True):
    """Returns a DataLoader for the provided dataset path."""

    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    images_path = os.path.join(dataset_path, "images")
    masks_path = os.path.join(dataset_path, "labels")

    dataset = SegmentationDataset(images_path, masks_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader

# Datasets
#todo: proper train path
train_path = "datasets/merged/val_kvasir"
train_loader = get_dataloader(train_path, batch_size=8, shuffle=True)
val_loader = get_dataloader("datasets/merged/val", batch_size=8, shuffle=False)
val_kvasir_loader = get_dataloader("datasets/merged/val_kvasir", batch_size=8, shuffle=False)
val_clinic_loader = get_dataloader("datasets/merged/val_clinic", batch_size=8, shuffle=False)

#todo: more epochs
model_params = {
    "model": "Unet",
    "encoder": "resnet34",
    "encoder_weights": "imagenet",
    "activation": None,
    "classes": 1,
    "learning_rate": 1e-4,
    "epochs": 2,
    "train_path": train_path
}

# Model, Loss, and Optimizer
model = Unet(
    encoder_name=model_params["encoder"], 
    encoder_weights=model_params["encoder_weights"], 
    classes=model_params["classes"],
    activation=model_params["activation"]
)
model = model.to(device)  # Move model to GPU

loss = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=model_params["learning_rate"])
metric_fscore = Fscore()
metric_iou = IoU()

# Directory to save the model
os.makedirs("./models", exist_ok=True)

def evaluate(val_loader, validation_set_name=""):
    """Evaluate the model on the provided validation set."""

    val_loss_total = 0
    val_iou_total = 0
    val_fscore_total = 0

    for val_images, val_masks in val_loader:
        val_images, val_masks = val_images.to(device), val_masks.to(device)
        val_preds = model(val_images)  # Predicted shape: (batch_size, 1, height, width)
        val_preds = torch.sigmoid(val_preds)  # Apply sigmoid for probabilities

        if val_masks.ndim == 3:  # Ensure masks have same shape as predictions
            val_masks = val_masks.unsqueeze(1)  # Add channel dimension

        val_loss = loss(val_preds, val_masks)
        val_iou = metric_iou(val_preds, val_masks)
        va_fscore = metric_fscore(val_preds, val_masks)

        val_loss_total += val_loss.item()
        val_iou_total += val_iou.item()
        val_fscore_total += va_fscore.item()

    val_loss_avg = val_loss_total / len(val_loader)
    val_iou_avg = val_iou_total / len(val_loader)
    val_fscore_avg = val_fscore_total / len(val_loader)

    print(f"Validation Loss ({validation_set_name}): {val_loss_avg:.4f}, IoU: {val_iou_avg:.4f} F1: {val_fscore_avg:.4f}")

    writer.add_scalar(f"Loss/val ({validation_set_name})", val_loss_avg, epoch)
    writer.add_scalar(f"IoU/val ({validation_set_name})", val_iou_avg, epoch)
    writer.add_scalar(f"F1/val ({validation_set_name})", val_fscore_avg, epoch)

    return val_loss_avg, val_iou_avg, val_fscore_avg


print("Starting Training...")

writer = SummaryWriter()

# Training Loop
for epoch in range(model_params["epochs"]):  # Number of epochs
    print(f"Starting Epoch {epoch + 1} [{strftime('%Y-%m-%d %H:%M:%S', gmtime())}]")

    time_start = time.time()

    model.train()
    train_loss_total = 0
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)  # Move data to GPU
        preds = model(images)
        preds = torch.sigmoid(preds)  # Sigmoid for binary segmentation
        loss_value = loss(preds, masks)  # Compute loss
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        train_loss_total += loss_value.item()

    train_loss_avg = train_loss_total / len(train_loader)
    print(f"Training Loss: {train_loss_avg:.4f}")

    writer.add_scalar("Loss/train", train_loss_avg, epoch)

    # Validation
    model.eval()
    with torch.no_grad():
        
        val_loss_kvasir, val_iou_kvasir, val_fscore_kvasir = evaluate(val_kvasir_loader, "Kvasir")
        val_loss_clinic, val_iou_clinic, val_fscore_clinic = evaluate(val_clinic_loader, "Clinic")

        time_end = time.time()
        # print duration of epoch in hh:mm:ss
        print(f"Epoch {epoch + 1} took {strftime('%H:%M:%S', gmtime(time_end - time_start))}")




    # --------------------- Plot random result in between epochs --------------------------

    val_images, val_masks = next(iter(val_loader))  # Get a random batch
    val_images, val_masks = val_images.to(device), val_masks.to(device)
    val_preds = model(val_images)
    val_preds = torch.sigmoid(val_preds)

    # Select a random image from the batch
    random_idx = np.random.randint(0, val_images.size(0))
    image = val_images[random_idx].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5  # Denormalize
    mask = val_masks[random_idx].cpu().detach().numpy().squeeze()
    pred_mask = (val_preds[random_idx].cpu().detach().numpy().squeeze() > 0.5).astype(np.uint8)  # Threshold

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    # Export to TensorBoard
    writer.add_figure("Validation Sample", fig, epoch)

    # Close the figure to release memory
    plt.close(fig)

    # ----------------------------------------------------------------------------------------


# Final Validation Performance
print("\n\nFinal Validation Performance:")
print(f"Validation Loss (Kvasir): {val_loss_kvasir:.4f}, IoU: {val_iou_kvasir:.4f} F1: {val_fscore_kvasir:.4f}")
print(f"Validation Loss (Clinic): {val_loss_clinic:.4f}, IoU: {val_iou_clinic:.4f} F1: {val_fscore_clinic:.4f}")


model_save_name = f"unet_segmentation_{strftime('%m-%d_%H-%M', gmtime())}"
# Save the trained model
model_save_path = os.path.join(f"models/{model_save_name}.pth")
torch.save(model.state_dict(), model_save_path)

# Save the training parameters to a JSON file
model_params["validation_loss_kvasir"] = val_loss_kvasir
model_params["validation_iou_kvasir"] = val_iou_kvasir
model_params["validation_fscore_kvasir"] = val_fscore_kvasir

model_params["validation_loss_clinic"] = val_loss_clinic
model_params["validation_iou_clinic"] = val_iou_clinic
model_params["validation_fscore_clinic"] = val_fscore_clinic

metadata_save_path = os.path.join(f"models/{model_save_name}.json")
with open(metadata_save_path, "w") as f:
    json.dump(model_params, f, indent=4)

print(f"Model saved to {model_save_path}")

writer.close()

# # push to Hugging Face 
# # check if HuggingFace "HF_TOKEN" environment variable is set
# if "HF_TOKEN" in os.environ:
#     from huggingface_hub import login
#     hf_token = os.environ["HF_TOKEN"]
#     # Login to the Hugging Face model hub
#     login(token=hf_token)

#     # Push the model to the Hugging Face model hub
#     model.save_pretrained(model_save_path, model_card_kwargs={"dataset": "POVa Dataset"}, push_to_hub=True)
