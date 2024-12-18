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

from seg_dataset import SegmentationDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset Paths
root_dir = "./"   # for easier training on Google Colab
train_images = os.path.join(root_dir,"datasets/merged/train/images")
train_masks = os.path.join(root_dir,"datasets/merged/train/labels")
val_images = os.path.join(root_dir,"datasets/merged/val/images")
val_masks = os.path.join(root_dir,"datasets/merged/val/labels")

# Transformations
transform = Compose([
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Datasets and Loaders
train_dataset = SegmentationDataset(train_images, train_masks, transform=transform)
val_dataset = SegmentationDataset(val_images, val_masks, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model, Loss, and Optimizer
model = Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=1, activation=None)
model = model.to(device)  # Move model to GPU

loss = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
metric_fscore = Fscore()
metric_iou = IoU()

# Directory to save the model
os.makedirs("./models", exist_ok=True)


print("Starting Training...")

# Training Loop
for epoch in range(6):  # Number of epochs
    print(f"Epoch {epoch + 1}")
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

    # Validation
    model.eval()
    val_loss_total = 0
    val_iou_total = 0
    val_fscore_total = 0
    with torch.no_grad():
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
        print(f"Validation Loss: {val_loss_avg:.4f}, IoU: {val_iou_avg:.4f} F1: {val_fscore_avg:.4f}")

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

    # Plot
    plt.figure(figsize=(12, 4))
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

    plt.show()

    # ----------------------------------------------------------------------------------------


# Final Validation Performance
print("\n\nFinal Validation Performance:")
print(f"Validation Loss: {val_loss_avg:.4f}, IoU: {val_iou_avg:.4f} F1: {val_fscore_avg:.4f}")



# Save the trained model
model_save_path = os.path.join(root_dir,"models/unet_segmentation_6ep.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# # push to Hugging Face 
# # check if HuggingFace "HF_TOKEN" environment variable is set
# if "HF_TOKEN" in os.environ:
#     from huggingface_hub import login
#     hf_token = os.environ["HF_TOKEN"]
#     # Login to the Hugging Face model hub
#     login(token=hf_token)

#     # Push the model to the Hugging Face model hub
#     model.save_pretrained(model_save_path, model_card_kwargs={"dataset": "POVa Dataset"}, push_to_hub=True)
