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

<!-- table -->

| Model | Checkpoint | Train Dataset | Test Dataset | F1     | IoU    | Description |
| ----- | ---------- | ------------- | ------------ | ------ | ------ | ----------- |
| UNet  |            | Standard      | Kvasir-SEG   | 0.8252 | 0.7461 | 4 epochs    |
| UNet  |            | Standard      | CVC-ClinicDB | 0.7611 | 0.6822 | 4 epochs    |
| UNet  |            | Standard      | Kvasir-SEG   | 0.8264 | 0.7477 | 10 epochs   |
| UNet  |            | Standard      | CVC_ClinicDB | 0.7254 | 0.6498 | 10 epochs   |

#### Segment Anything Finetuning Results

Finetunind decoder only experiments results:

<!-- table -->

| Model   | Checkpoint                                            | Train Dataset | Test Dataset | F1     | IoU    | Description |
| ------- | ----------------------------------------------------- | ------------- | ------------ | ------ | ------ | ----------- |
| SAM(md) | SAM_f_merged_newest_meta_7_md_e69_iou0.5655.pth       | Standard      | Kvasir-SEG   | 0.7422 | 0.6481 | 69 epochs   |
| SAM(md) |                                                       | Standard      | CVC-ClinicDB | 0.6869 | 0.6014 |             |
| SAM(md) | SAM_f_sam_merged_newest_meta_7_md_e97_iou0.5541.pth   | Standard SAM  | Kvasir-SEG   | 0.7757 | 0.6896 | 97 epochs   |
| SAM(md) |                                                       | Standard SAM  | CVC-ClinicDB | 0.6661 | 0.5859 |             |
| SAM(md) | SAM_f_merged_newest_A_meta_7_md_e11_iou0.5464.pth     | Augmented     | Kvasir-SEG   | 0.7444 | 0.6530 | 11 epochs   |
| SAM(md) |                                                       | Augmented     | CVC-ClinicDB | 0.6720 | 0.5896 |             |
| SAM(md) | SAM_f_sam_merged_newest_A_meta_7_md_e31_iou0.5566.pth | Augmented SAM | Kvasir-SEG   | 0.7687 | 0.6851 | 31 epochs   |
| SAM(md) |                                                       | Augmented SAM | CVC-ClinicDB | 0.6528 | 0.5766 |             |

Finetuning encoders only experiments results:

<!-- table -->

| Model   | Checkpoint                                            | Train Dataset | Test Dataset | F1     | IoU    | Description |
| ------- | ----------------------------------------------------- | ------------- | ------------ | ------ | ------ | ----------- |
| SAM(ie) | SAM_f_merged_newest_meta_7_ie_e34_iou0.7617.pth       | Standard      | Kvasir-SEG   | 0.8787 | 0.8102 | 34 epochs   |
| SAM(ie) |                                                       | Standard      | CVC-ClinicDB | 0.8386 | 0.7685 |             |
| SAM(ie) | SAM_f_sam_merged_newest_meta_7_ie_e29_iou0.7794.pth   | Standard SAM  | Kvasir-SEG   | 0.8807 | 0.8147 | 29 epochs   |
| SAM(ie) |                                                       | Standard SAM  | CVC-ClinicDB | 0.8590 | 0.7896 |             |
| SAM(ie) | SAM_f_merged_newest_A_meta_7_ie_e6_iou0.7474.pth      | Augmented     | Kvasir-SEG   | 0.8597 | 0.7843 | 6 epochs    |
| SAM(ie) |                                                       | Augmented     | CVC-ClinicDB | 0.8431 | 0.7650 |             |
| SAM(ie) | SAM_f_sam_merged_newest_A_meta_7_ie_e18_iou0.7855.pth | Augmented SAM | Kvasir-SEG   | 0.8837 | 0.8166 | 18 epochs   |
| SAM(ie) |                                                       | Augmented SAM | CVC-ClinicDB | 0.8659 | 0.7975 |             |

Finetuning encoders and decoder experiments results:

<!-- table -->

| Model     | Checkpoint                                              | Train Dataset | Test Dataset | F1     | IoU    | Description |
| --------- | ------------------------------------------------------- | ------------- | ------------ | ------ | ------ | ----------- |
| SAM(iemd) | SAM_f_merged_newest_meta_7_iemd_e32_iou0.7528.pt        | Standard      | Kvasir-SEG   |        |        |             |
| SAM(iemd) |                                                         | Standard      | CVC-ClinicDB |        |        |             |
| SAM(iemd) | SAM_f_sam_merged_newest_meta_7_iemd_e52_iou0.7848.pth   | Standard SAM  | Kvasir-SEG   | 0.8801 | 0.8178 | 52 epochs   |
| SAM(iemd) |                                                         | Standard SAM  | CVC-ClinicDB | 0.8541 | 0.7886 |             |
| SAM(iemd) | SAM_merged_newest_A_meta_7_iemd_e11_iou0.7632.pth       | Augmented     | Kvasir-SEG   |        |        |             |
| SAM(iemd) |                                                         | Augmented     | CVC-ClinicDB |        |        |             |
| SAM(iemd) | SAM_f_sam_merged_newest_A_meta_7_iemd_e34_iou0.7976.pth | Augmented SAM | Kvasir-SEG   | 0.8831 | 0.8189 | 34 epochs   |
| SAM(iemd) |                                                         | Augmented SAM | CVC-ClinicDB | 0.8714 | 0.8058 |             |

_Note: SAM datasets versions do noc contain images without polyps_

## References

1. [Segment anything Model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#segment-anything)
2. [How To Fine-Tune The Segment Anything Model](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)
