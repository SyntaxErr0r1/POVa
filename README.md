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

## Datasets used

<!-- table -->

| Dataset name                    | Train images | Validation images | Description |
| ------------------------------- | ------------ | ----------------- | ----------- |
| CVC-ClinicDB                    | -            | 612               | -           |
| Kvasir-SEG                      | 880          | 120               | -           |
| PolypGen2021_MultiCenterData_v3 | 8037         | -                 | -           |

## Deployment

### Prepare python environment

Create python environment:
`python3 -m venv env22`
Activate python environment:
`source env22/bin/activate`
Install required packages:
`pip install -r requirements.txt`

## Resources

We utilized and modified the code from the (finetune-anything)[https://github.com/ziqi-jin/finetune-anything] project, distributed under the MIT License, as a foundation for fine-tuning the SAM model for polyp segmentation. The original code was modified extensively to meet our specific requirements.
