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

1. Prepaired merged dataset is available for download here: https://drive.google.com/file/d/1tSpQWfRs7Qqi37wb6D5Dep6mbB2TT_Da/view?usp=drive_link
2. Unzip dataset into `./datasets/merged` folder

For merging:

1. Create a 'datasets' folder
2. Download CVC-ClinicDB, Kvasir-SEG and PolypGen2021_MultiCenterData_v3 datasets into this folder, link: [Merged_newest.zip](https://drive.google.com/file/d/19frkLsWn46HJgc64ti7FWDLZoXbDune5/view?usp=drive_link)
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

| Model     | Train Dataset | Test Dataset | F1     | IoU    | Description |
| --------- | ------------- | ------------ | ------ | ------ | ----------- |
| UNet      | Standard      | Kvasir-SEG   | 0.8252 | 0.7461 | 4 epochs    |
| UNet      | Standard      | CVC-ClinicDB | 0.7611 | 0.6822 | 4 epochs    |
| UNet      | Standard      | Kvasir-SEG   | 0.8264 | 0.7477 | 10 epochs   |
| UNet      | Standard      | CVC_ClinicDB | 0.7254 | 0.6498 | 10 epochs   |
| SAM(md)   | Standard      | Kvasir-SEG   |        |        |             |
| SAM(md)   | Standard      | CVC-ClinicDB |        |        |             |
| SAM(ie)   | Standard      | Kvasir-SEG   |        |        |             |
| SAM(ie)   | Standard      | CVC-ClinicDB |        |        |             |
| SAM(iemd) | Standard      | Kvasir-SEG   |        |        |             |
| SAM(iemd) | Standard      | CVC-ClinicDB |        |        |             |
| SAM(md)   | Augmented     | Kvasir-SEG   |        |        |             |
| SAM(md)   | Augmented     | CVC-ClinicDB |        |        |             |
| SAM(ie)   | Augmented     | Kvasir-SEG   |        |        |             |
| SAM(ie)   | Augmented     | CVC-ClinicDB |        |        |             |
| SAM(iemd) | Augmented     | Kvasir-SEG   |        |        |             |
| SAM(iemd) | Augmented     | CVC-ClinicDB |        |        |             |

## References

1. [Segment anything Model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#segment-anything)
2. [How To Fine-Tune The Segment Anything Model](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)
