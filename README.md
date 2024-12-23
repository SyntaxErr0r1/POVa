# Polyp Segmentation from Images

This project is part of the Computer Vision (POVa) course at VUT FIT. This project aims to create and train machine-learning models for polyp segmentation in colonoscopy images.

## Authors

Ľuboš Martinček, Eva Mičánková, Juraj Dedič

## Prepare python environment

Create python environment:
`python3 -m venv env22`
Activate python environment:
`source env22/bin/activate`
Install required packages:
`pip install -r requirements.txt`

## Download merged dataset / Merge the datasets

Download merged dataset:

1. Prepaired merged dataset is available for download here: [Merged_newest.zip](https://drive.google.com/file/d/19frkLsWn46HJgc64ti7FWDLZoXbDune5/view?usp=drive_link)
2. Unzip dataset into `./datasets/merged` folder

Merge using `dataset_merge.py` script:

1. Create a 'datasets' folder
2. Download CVC-ClinicDB, Kvasir-SEG and PolypGen2021_MultiCenterData_v3 datasets into this folder, link: [Datasets](https://drive.google.com/drive/folders/1TE8Di181fkII9du4kxLZe7V6_ZjY3o20?usp=drive_link)
3. Unzip datasets into this folder
4. Run `python ./datasets/dataset_merge.py`

### Datasets used

<!-- table -->

| Dataset name                    | Train images | Validation images | Description |
| ------------------------------- | ------------ | ----------------- | ----------- |
| CVC-ClinicDB                    | -            | 612               | -           |
| Kvasir-SEG                      | 880          | 120               | -           |
| PolypGen2021_MultiCenterData_v3 | 8,037        | -                 | -           |

### Merged_newest.zip stats

<!-- table -->

| Dataset split                | Num of images |
| ---------------------------- | ------------- |
| train                        | 8,917         |
| train_augmented_small_random | 26,751        |
| val_kvasir                   | 120           |
| val_clinic                   | 612           |

## Augmentation

1. Run `python augment.py` with arguments
   -data path_to_dataset (must contain `images` and `labels` directories)

2. Run `python augment_small.py` with arguments
   -data path_to_dataset (must contain `images` and `labels` directories) creates smaller and randomized version of the dataset

Augmented train split is available for download here: [train_augmented_small_random.zip](https://drive.google.com/file/d/1q9Q1o15nKnhKeSsZ_K2x6lEJGZm8t6I-/view?usp=drive_link)

## Deployment

#### SAM

Download pretrained [SAM checkpoints](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)

Run SAM finetuning:
`python3 finetune_sam.py --cfg ./sam/configs/<YAML configuration file>`

## Evaluation

1. Run `python eval.py` with arguments
   `--arch ["Unet", "SAM"]` (default architecture is `Unet`)
   `--model path_to_saved_model`
   `--data path_to_evaluation_dataset` (must contain `images` and `labels` directories)
   `--batch_size` (default is `1`)

### Evaluation Results

Checkpoints are available for download here: [Checkpoints](https://drive.google.com/drive/folders/1oSuGKIIHufZv7Mifya1xgN3rmqlcPPwl?usp=drive_link)

#### UNet Training Results

<!-- table -->
<!-- | UNet  |            | Standard      | Kvasir-SEG   | 0.8252 | 0.7461 | 4 epochs    |
| UNet  |            | Standard      | CVC-ClinicDB | 0.7611 | 0.6822 | 4 epochs    |
| UNet  |            | Standard      | Kvasir-SEG   | 0.8264 | 0.7477 | 10 epochs   |
| UNet  |            | Standard      | CVC_ClinicDB | 0.7254 | 0.6498 | 10 epochs   | -->

| Model | Checkpoint | Train Dataset | Test Dataset | F1     | IoU    | Description |
| ----- | ---------- | ------------- | ------------ | ------ | ------ | ----------- |
| UNet + Resnet34  |     [UNet+Resnet34+Standard (unet_segmentation_12-19_08-36).pth](https://drive.google.com/file/d/1RRK7I18bC0OcvOCE67Co_LH6KjDrpla7/view?usp=drive_link)       | Standard     | Kvasir-SEG   | 0.8564 | 0.7873 | 39 epochs   |
| UNet + Resnet34  |            | Standard     | CVC-ClinicDB | 0.7856 | 0.7134 | 39 epochs   |
| UNet + Resnet34  |     [UNet+Resnet34+Augmented (unet_segmentation_12-21_11-21)](https://drive.google.com/file/d/15GFF-NmsA6rHIh4FtqoqRmiuSWnzGOmE/view?usp=drive_link)       | Augmented     | Kvasir-SEG   | 0.8466 | 0.7729 | 41 epochs   |
| UNet + Resnet34  |            | Augmented     | CVC-ClinicDB | 0.8060 | 0.7307 | 41 epochs   |


<!-- non aug
Starting Epoch 41 [2024-12-19 08:26:54]
Training Loss: 0.0985
Validation Loss (Kvasir): 0.1591, IoU: 0.7319 F1: 0.8409
Validation Loss (Clinic): 0.1780, IoU: 0.7084 F1: 0.8221
Epoch 41 took 00:09:32
Model saved to models/unet_segmentation_12-19_08-36.pth

Kvasir
Evaluation Scores:
  Fscore: 0.8564
  Accuracy: 0.0000
  Precision: 0.8674
  Recall: 0.9049
  IoU: 0.7873

Clinic
Evaluation Scores:
  Fscore: 0.7856
  Accuracy: 0.0000
  Precision: 0.8863
  Recall: 0.8035
  IoU: 0.7134 

<!-- aug -->
<!-- Starting Epoch 39 [2024-12-21 10:54:35]
Training Loss: 0.1007
Validation Loss (Kvasir): 0.1604, IoU: 0.7304 F1: 0.8395
Validation Loss (Clinic): 0.1761, IoU: 0.7101 F1: 0.8239
Epoch 39 took 00:26:57
Model saved to models/unet_segmentation_12-21_11-21.pth 

Kvasir
Evaluation Scores:
  Fscore: 0.8466
  Accuracy: 0.0000
  Precision: 0.8967
  Recall: 0.8627
  IoU: 0.7729

Clinic
Evaluation Scores:
  Fscore: 0.8060
  Accuracy: 0.0000
  Precision: 0.9040
  Recall: 0.8081
  IoU: 0.7307 -->

#### Segment Anything Finetuning Results

Finetunind decoder only experiments results:

<!-- table -->

| Model   | Checkpoint                                                                                                                                     | Train Dataset | Test Dataset | F1     | IoU    | Description |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------ | ------ | ------ | ----------- |
| SAM(md) | [SAM_f_merged_newest_meta_7_md_e69_iou0.5655.pth](https://drive.google.com/file/d/1213oLXWgeDYt9kXu3ZUJxOzcMVpkYVe-/view?usp=drive_link)       | Standard      | Kvasir-SEG   | 0.7422 | 0.6481 | 69 epochs   |
| SAM(md) |                                                                                                                                                | Standard      | CVC-ClinicDB | 0.6869 | 0.6014 |             |
| SAM(md) | [SAM_f_sam_merged_newest_meta_7_md_e97_iou0.5541.pth](https://drive.google.com/file/d/1vLQfT2lj2w02e3GCxVInJUs1Ul_WFs97/view?usp=drive_link)   | Standard SAM  | Kvasir-SEG   | 0.7757 | 0.6896 | 97 epochs   |
| SAM(md) |                                                                                                                                                | Standard SAM  | CVC-ClinicDB | 0.6661 | 0.5859 |             |
| SAM(md) | [SAM_f_merged_newest_A_meta_7_md_e11_iou0.5464.pth](https://drive.google.com/file/d/1nV1gkicSD1Ss0rHZQ_cMK3PwEoN9_lqR/view?usp=drive_link)     | Augmented     | Kvasir-SEG   | 0.7444 | 0.6530 | 11 epochs   |
| SAM(md) |                                                                                                                                                | Augmented     | CVC-ClinicDB | 0.6720 | 0.5896 |             |
| SAM(md) | [SAM_f_sam_merged_newest_A_meta_7_md_e31_iou0.5566.pth](https://drive.google.com/file/d/1jw9TyfcWLbPigz78bqqPJbq7EXuZB2xM/view?usp=drive_link) | Augmented SAM | Kvasir-SEG   | 0.7687 | 0.6851 | 31 epochs   |
| SAM(md) |                                                                                                                                                | Augmented SAM | CVC-ClinicDB | 0.6528 | 0.5766 |             |

Finetuning encoders only experiments results:

<!-- table -->

| Model   | Checkpoint                                                                                                                                     | Train Dataset | Test Dataset | F1     | IoU    | Description |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------ | ------ | ------ | ----------- |
| SAM(ie) | [SAM_f_merged_newest_meta_7_ie_e34_iou0.7617.pth](https://drive.google.com/file/d/1W5aEcjeNHR0tV9KEPIGs7YTiPahtZ1F4/view?usp=drive_link)       | Standard      | Kvasir-SEG   | 0.8787 | 0.8102 | 34 epochs   |
| SAM(ie) |                                                                                                                                                | Standard      | CVC-ClinicDB | 0.8386 | 0.7685 |             |
| SAM(ie) | [SAM_f_sam_merged_newest_meta_7_ie_e29_iou0.7794.pth](https://drive.google.com/file/d/1KIvbLZOZXCnDrjxVhx9OBK0FnxIuF1-O/view?usp=drive_link)   | Standard SAM  | Kvasir-SEG   | 0.8807 | 0.8147 | 29 epochs   |
| SAM(ie) |                                                                                                                                                | Standard SAM  | CVC-ClinicDB | 0.8590 | 0.7896 |             |
| SAM(ie) | [SAM_f_merged_newest_A_meta_7_ie_e6_iou0.7474.pth](https://drive.google.com/file/d/14Y6wAPnX--ou-JArU_2FB7Fk2aB5ld52/view?usp=drive_link)      | Augmented     | Kvasir-SEG   | 0.8597 | 0.7843 | 6 epochs    |
| SAM(ie) |                                                                                                                                                | Augmented     | CVC-ClinicDB | 0.8431 | 0.7650 |             |
| SAM(ie) | [SAM_f_sam_merged_newest_A_meta_7_ie_e18_iou0.7855.pth](https://drive.google.com/file/d/1XUaK04B22O11jl2aqlaJvEVdSM9AQjwJ/view?usp=drive_link) | Augmented SAM | Kvasir-SEG   | 0.8837 | 0.8166 | 18 epochs   |
| SAM(ie) |                                                                                                                                                | Augmented SAM | CVC-ClinicDB | 0.8659 | 0.7975 |             |

Finetuning encoders and decoder experiments results:

<!-- table -->

| Model     | Checkpoint                                                                                                                                       | Train Dataset | Test Dataset | F1     | IoU    | Description |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- | ------------ | ------ | ------ | ----------- |
| SAM(iemd) | [SAM_f_merged_newest_meta_7_iemd_e32_iou0.7528.pt](https://drive.google.com/file/d/1aoLrisNZO8-CoK_QRq_jOa2JiP3UBaCc/view?usp=drive_link)        | Standard      | Kvasir-SEG   | 0.8803 | 0.8172 | 32 epochs   |
| SAM(iemd) |                                                                                                                                                  | Standard      | CVC-ClinicDB | 0.8206 | 0.7527 |             |
| SAM(iemd) | [SAM_f_sam_merged_newest_meta_7_iemd_e52_iou0.7848.pth](https://drive.google.com/file/d/1b6_VULKrvRkLOHWvVXCldgIYkWuJGnYt/view?usp=drive_link)   | Standard SAM  | Kvasir-SEG   | 0.8801 | 0.8178 | 52 epochs   |
| SAM(iemd) |                                                                                                                                                  | Standard SAM  | CVC-ClinicDB | 0.8541 | 0.7886 |             |
| SAM(iemd) | [SAM_merged_newest_A_meta_7_iemd_e11_iou0.7632.pth](https://drive.google.com/file/d/1HnMYQobFkQnpMJrbQHTSCGCsQZWtTAb7/view?usp=drive_link)       | Augmented     | Kvasir-SEG   | 0.8846 | 0.8196 | 11 epochs   |
| SAM(iemd) |                                                                                                                                                  | Augmented     | CVC-ClinicDB | 0.8437 | 0.7736 |             |
| SAM(iemd) | [SAM_f_sam_merged_newest_A_meta_7_iemd_e34_iou0.7976.pth](https://drive.google.com/file/d/1E3xzZkFpo8MEHnc5Ta59lhy3zkosa3TL/view?usp=drive_link) | Augmented SAM | Kvasir-SEG   | 0.8831 | 0.8189 | 34 epochs   |
| SAM(iemd) |                                                                                                                                                  | Augmented SAM | CVC-ClinicDB | 0.8714 | 0.8058 |             |

_Note: SAM datasets versions do not contain images without polyps._


kvasir
Evaluation Scores:
  Fscore: 0.8879
  IoU: 0.8294

clinic
Evaluation Scores:
  Fscore: 0.8643
  IoU: 0.7988

standard scse

kvasir
  Fscore: 0.8784
  IoU: 0.8179

clinic
Evaluation Scores:
  Fscore: 0.8460
  IoU: 0.7783

## UNet (with scSE attention) + EfficientNet-b5 experiment 
| Model | Checkpoint | Train Dataset | Test Dataset | F1     | IoU    | Description |
| ----- | ---------- | ------------- | ------------ | ------ | ------ | ----------- |
| UNet[scSE] + EfficientNet-b5  |     [UNet+EfficientNet-b5-scse (unet_scse_standard_12-23_02-41)](https://drive.google.com/file/d/11IKcwPnYRhdTQafzYxrekmqewsZNoC3r/view?usp=drive_link)       | Standard     | Kvasir-SEG   | 0.8784 | 0.8179 |   |
| UNet[scSE] + EfficientNet-b5  |            | Standard     | CVC-ClinicDB | 0.8460 | 0.7783 |    |
| UNet[scSE] + EfficientNet-b5  |     [UNet+EfficientNet-b5-scse (unet_scse_augmented_12-23_13-51)](https://drive.google.com/file/d/16Psc45lOQu5lOroc-erNYI7xhR5_Ets0/view?usp=drive_link)       | Augmented     | Kvasir-SEG   | 0.8879 | 0.8294 |    |
| UNet[scSE] + EfficientNet-b5  |            | Augmented     | CVC-ClinicDB | 0.8643 | 0.7988 |    |


## References

1. [Segment anything Model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#segment-anything)
2. [How To Fine-Tune The Segment Anything Model](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)
3. [Segmentation Models Pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)