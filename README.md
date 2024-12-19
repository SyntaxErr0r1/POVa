# Polyp Segmentation from Images

This project is part of the Computer Vision (POVa) course at VUT FIT. This project aims to create and train machine-learning models for polyp segmentation in colonoscopy images.

## Authors

Ľuboš Martinček, Eva Mičánková, Juraj Dedič

## Download merged dataset / Merge the datasets

Download merged dataset:

1. Prepaired merged dataset is available for download here: https://drive.google.com/file/d/1tSpQWfRs7Qqi37wb6D5Dep6mbB2TT_Da/view?usp=drive_link
2. Unzip dataset into `./datasets/merged` folder

For merging:

1. Create a 'datasets' folder
2. Download CVC-ClinicDB, Kvasir-SEG and PolypGen2021_MultiCenterData_v3 datasets into this folder, link: https://drive.google.com/drive/folders/1TE8Di181fkII9du4kxLZe7V6_ZjY3o20
3. Unzip datasets into this folder
4. Run `python ./datasets/dataset_merge.py`

## Augmentation
1. Run `python augment.py` with arguments
    -data path_to_dataset (must contain `images` and `labels` directories)


## Datasets used

<!-- table -->

| Dataset name                    | Train images | Validation images | Description |
| ------------------------------- | ------------ | ----------------- | ----------- |
| CVC-ClinicDB                    | -            | 612               | -           |
| Kvasir-SEG                      | 880          | 120               | -           |
| PolypGen2021_MultiCenterData_v3 | 8037         | -                 | -           |

## Deployment
## Evaluation
1. Run `python eval.py` with arguments
    -model path_to_saved_model (only UNet supported right now)
    -data path_to_evaluation_dataset (must contain `images` and `labels` directories)


### Prepare python environment

Create python environment:
`python3 -m venv env22`
Activate python environment:
`source env22/bin/activate`
Install required packages:
`pip install -r requirements.txt`


### Evaluation

| Model | Train Dataset | Test Dataset | F1 | IoU | Description |
| --- | --- | --- | --- | --- | --- |
| UNet | Standard | Kvasir-SEG | 0.8252 | 0.7461 | 4 epochs |
| UNet | Standard | CVC-ClinicDB | 0.7611 | 0.6822 | 4 epochs |
| UNet | Standard | Kvasir-SEG | 0.8264 | 0.7477 | 10 epochs |
| UNet | Standard | CVC_ClinicDB | 0.7254 | 0.6498 | 10 epochs |