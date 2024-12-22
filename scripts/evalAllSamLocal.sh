#!/bin/bash

cpt_list=(
    "SAM_f_merged_newest_A_meta_7_ie_e6_iou0.7474.pth"
    "SAM_f_merged_newest_meta_7_md_e69_iou0.5655.pth"
    "SAM_f_sam_merged_newest_meta_7_ie_e29_iou0.7794.pth"
    "SAM_f_merged_newest_A_meta_7_md_e11_iou0.5464.pth"
    "SAM_f_sam_merged_newest_A_meta_7_ie_e18_iou0.7855.pth"
    "SAM_f_sam_merged_newest_meta_7_iemd_e52_iou0.7848.pth"
    "SAM_f_merged_newest_meta_7_ie_e34_iou0.7617.pth"
    "SAM_f_sam_merged_newest_A_meta_7_iemd_e34_iou0.7976.pth"
    "SAM_f_sam_merged_newest_meta_7_md_e97_iou0.5541.pth"
    "SAM_f_sam_merged_newest_A_meta_7_md_e31_iou0.5566.pth"
    "SAM_f_merged_newest_meta_7_iemd_e32_iou0.7528.pth"
    "SAM_merged_newest_A_meta_7_iemd_e11_iou0.7632.pth"
)

EVAL_DIR_KVASIR=./datasets/Merged_newest/val_kvasir
EVAL_DIR_CLINIC=./datasets/Merged_newest/val_clinic

mkdir -p ./logs-eval

# Loop through the checkpoint list
for file in "${cpt_list[@]}"; do
    echo "Submitting checkpoint: $file"
    CPT=./models/final/$file
    OUTPUT_LOG_K=./logs-eval/$file.K.log
    OUTPUT_LOG_C=./logs-eval/$file.C.log
    ERROR_LOG_K=./logs-eval/$file.K.err
    ERROR_LOG_C=./logs-eval/$file.C.err

    python3 eval.py --arch SAM --model $CPT --data $EVAL_DIR_KVASIR > $OUTPUT_LOG_K 2> $ERROR_LOG_K
    python3 eval.py --arch SAM --model $CPT --data $EVAL_DIR_CLINIC > $OUTPUT_LOG_C 2> $ERROR_LOG_C
done
