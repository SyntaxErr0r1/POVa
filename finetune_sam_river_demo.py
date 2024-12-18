# common
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from numpy import zeros
from numpy.random import randint
import torch
import os
import cv2
from statistics import mean
from torch.nn.functional import threshold, normalize
# Data Viz
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

## Import training data
image_path = "./datasets/Train/Train/Images"
total_images = len(os.listdir(image_path))
all_image_paths = sorted(glob(image_path + "/*.jpg"))
print(f"Total Number of Images : {total_images}")

lable_path = "./datasets/Train/Train/Labels"
total_lables = len(os.listdir(lable_path))
all_lable_paths = sorted(glob(lable_path + "/*.png"))
print(f"Total Number of Images : {total_lables}")

train_image_paths = all_image_paths[0:total_images]
train_lable_paths = all_lable_paths[0:total_lables]

## Import validation data
image_path = "./datasets/Val/Val/Images"
Val_total_images = len(os.listdir(image_path))
all_image_paths = sorted(glob(image_path + "/*"))
print(f"Total Number of Images : {Val_total_images}")

lable_path = "./datasets/Val/Val/labels"
Val_total_lables = len(os.listdir(lable_path))
all_lable_paths = sorted(glob(lable_path + "/*.png"))
print(f"Total Number of Images : {Val_total_lables}")

Val1_image_paths = all_image_paths[0:Val_total_images]
Val1_lable_paths = all_lable_paths[0:Val_total_lables]

##
# Please dont run this line if you would like to use the original size of input images.
desired_size=(512, 512)

ground_truth_masks = {}
for k in range(0, len(train_image_paths)):
    gt_grayscale = cv2.imread(train_lable_paths[k], cv2.IMREAD_GRAYSCALE)
    if desired_size is not None:
        gt_grayscale = cv2.resize(gt_grayscale, desired_size, interpolation=cv2.INTER_LINEAR)

    ground_truth_masks[k] = (gt_grayscale > 0)
    
ground_truth_masksv = {}
for s in range(0, len(Val1_lable_paths)):
    gt_grayscale = cv2.imread(Val1_lable_paths[s], cv2.IMREAD_GRAYSCALE)
    if desired_size is not None:
        gt_grayscale = cv2.resize(gt_grayscale, desired_size, interpolation=cv2.INTER_LINEAR)

    ground_truth_masksv[s] = (gt_grayscale > 0)
    
#####
model_type = 'vit_b'
checkpoint = './sam/checkpoints/sam_vit_b_01ec64.pth'
device = 'cuda:0'

from sam.segment_anything_ori import SamPredictor, sam_model_registry
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model.to(device)
sam_model.train()

##### Step 1: Preprocess the images
from collections import defaultdict
from sam.segment_anything_ori.utils.transforms import ResizeLongestSide

transformed_data = defaultdict(dict)
for k in range(len(train_image_paths)):  # Fix the loop iteration
    image = cv2.imread(train_image_paths[k])
    if desired_size is not None:
        image = cv2.resize(image, desired_size, interpolation=cv2.INTER_LINEAR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_image = sam_model.preprocess(transformed_image)
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])

    transformed_data[k]['image'] = input_image
    transformed_data[k]['input_size'] = input_size
    transformed_data[k]['original_image_size'] = original_image_size
    
lr = 1e-5
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
loss_fn   = torch.nn.BCEWithLogitsLoss()
keys = list(ground_truth_masks.keys())
keys1 = list(ground_truth_masksv.keys())
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define batch size
batch_size = 8
num_epochs = 3


#####
# FINETUNING
def train_on_batch(keys, batch_start, batch_end):
    batch_losses = []
    batch_accuracies = []

    for k in keys[batch_start:batch_end]:
        input_image = transformed_data[k]['image'].to(device)
        print(f"input_image: {input_image.shape}")
        input_size = transformed_data[k]['input_size']
        original_image_size = transformed_data[k]['original_image_size']

        # No grad here as we don't want to optimize the encoders
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)

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

        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        binary_mask = (threshold(torch.sigmoid(upscaled_masks), 0.5, 0))
        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
        gt_mask_resized = gt_mask_resized > 0.5
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

        loss = loss_fn(binary_mask, gt_binary_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

        # Calculate accuracy for training data
        train_accuracy = calculate_accuracy(binary_mask, gt_binary_mask)
        batch_accuracies.append(train_accuracy)

    return batch_losses, batch_accuracies

def calculate_accuracy(predictions, targets):
    binary_predictions = (predictions > 0.5).float()
    accuracy = (binary_predictions == targets).float().mean()
    return accuracy.item()

losses = []
val_losses = []
accuracies = []
best_val_loss = float('inf')  # Initialize best validation loss to positive infinity
val_acc = []

for epoch in range(num_epochs):
    
    epoch_losses = []
    epoch_accuracies = []

    # Training loop with batch processing
    for batch_start in range(0, len(keys), batch_size):
        batch_end = min(batch_start + batch_size, len(keys))

        batch_losses, batch_accuracies = train_on_batch(keys, batch_start, batch_end)

        # Calculate accuracy for the current batch
        batch_accuracy = mean(batch_accuracies)
        epoch_accuracies.extend(batch_accuracies)

        # Calculate mean training loss for the current batch
        batch_loss = mean(batch_losses)
        epoch_losses.append(batch_loss)

        print(f'Batch: [{batch_start+1}-{batch_end}]')
        print(f'Batch Loss: {batch_loss}')
        print(f'Batch Accuracy: {batch_accuracy}')

    # Calculate mean training loss for the current epoch
    mean_train_loss = mean(epoch_losses)
    mean_train_accuracy = mean(epoch_accuracies)
    losses.append(mean_train_loss)
    accuracies.append(mean_train_accuracy)

    print(f'EPOCH: {epoch}')
    print(f'Mean training loss: {mean_train_loss}')
    print(f'Mean training accuracy: {mean_train_accuracy}')
    
    predictor_tuned = SamPredictor(sam_model)

    # Validation loop
    val_loss = 0.0
    val_accuracy = 0.0
    num_val_examples = 0
    with torch.no_grad():
        for s in keys1[:len(Val1_image_paths)]:  # Replace validation_keys with your validation data keys
            image = cv2.imread(Val1_image_paths[s])
            if desired_size is not None:
               image = cv2.resize(image, desired_size, interpolation=cv2.INTER_LINEAR)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"image: {image.shape}")
            # Forward pass on validation data
            predictor_tuned.set_image(image)

            masks_tuned, _, _ = predictor_tuned.predict(
                point_coords=None,
                box=None,
                multimask_output=False,
            )
            print(f"masks_tuned: {masks_tuned.shape}")
            print(f"masks_tuned: {masks_tuned}")

            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masksv[s], (1, 1, ground_truth_masksv[s].shape[0], ground_truth_masksv[s].shape[1]))).to(device)
            gt_mask_resized = gt_mask_resized > 0.5
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            masks_tuned1 = torch.as_tensor(masks_tuned > 0, dtype=torch.float32)
            new_tensor = masks_tuned1.unsqueeze(0).to(device)

            # Calculate validation loss
            val_loss += loss_fn(new_tensor, gt_binary_mask).item()

            # Calculate accuracy for validation data
            val_accuracy += calculate_accuracy(new_tensor, gt_binary_mask)
            num_val_examples += 1

    # Calculate mean validation loss for the current epoch
    val_loss /= num_val_examples
    val_losses.append(val_loss)
    print(f'Mean validation loss: {val_loss}')

    # Calculate mean validation accuracy for the current epoch
    mean_val_accuracy = val_accuracy / num_val_examples
    val_acc.append(mean_val_accuracy)
    print(f'Mean validation accuracy: {mean_val_accuracy}')

    # Save the model checkpoint if the validation accuracy improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        models_path = './models'
        torch.save(sam_model.state_dict(), os.path.join(models_path, 'SAM5122weights_ViTB.pth'))

    # Clear GPU cache after each epoch
    torch.cuda.empty_cache()
