# Polyp Segmentation from Images
This project is part of the Computer Vision (POVa) course at VUT FIT. This project aims to create and train machine-learning models for polyp segmentation in colonoscopy images. 

## Authors
Ľuboš Martinček, Eva Mičánková, Juraj Dedič

## Merging the datasets
1. Create a 'datasets' folder
2. Download CVC-ClinicDB, Kvasir-SEG and PolypGen2021_MultiCenterData_v3 datasets into this folder, link: https://drive.google.com/drive/folders/1TE8Di181fkII9du4kxLZe7V6_ZjY3o20
3. Unzip datasets into this folder
4. Run `python ./datasets/dataset_merge.py`


## Datasets used

<!-- table -->
| Dataset name | Train images | Validation images | Description |
| --- | --- | --- | --- |
| CVC-ClinicDB | - | 612 | - |
| Kvasir-SEG | 880 | 120 | - |
| PolypGen2021_MultiCenterData_v3 | 8037 | - | - |