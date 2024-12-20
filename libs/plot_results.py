import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_results(sam_model, val_loader, epoch, device, target_size):
    images, masks, points, labels = next(iter(val_loader))  # Get a random batch
    images, masks, points, labels = images.to(device), masks.to(device), points.to(device), labels.to(device)
    
    with torch.no_grad():
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
