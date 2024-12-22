#!/bin/bash

#PBS -N SAMPolypSegmentation
#PBS -q gpu
#PBS -l select=1:ncpus=10:mem=64gb:scratch_local=200gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o $PBS_O_WORKDIR/logs/output.log
#PBS -e $PBS_O_WORKDIR/logs/error.log

##
# Run: qsub -v CPT=<.pth file in [./models/final/]> ./scripts/evalSam.sh

CPT=./models/final/$CPT
EVAL_DIR_KVASIR=./datasets/Merged_newest/val_kvasir
EVAL_DIR_CLINIC=./datasets/Merged_newest/val_clinic

# Load necessary modules (adjust according to your cluster environment)
module load singularity

# Navigate to the working directory
cd $PBS_O_WORKDIR

# Prepare scratch space if needed (optional, adjust based on your requirements)
SCRATCH_DIR=$SCRATCHDIR
mkdir -p $SCRATCH_DIR

LOG_DIR=$PBS_O_WORKDIR/logs-eval/$PBS_JOBID
mkdir -p $LOG_DIR

# Logs
OUTPUT_LOG=$LOG_DIR/output.log
ERROR_LOG=$LOG_DIR/error.log

echo "Load and run PyTorch container..." > $OUTPUT_LOG

singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.10-py3.SIF <<EOF

echo "Install segmentation_models..." > $OUTPUT_LOG
pip install -U git+https://github.com/qubvel-org/segmentation_models.pytorch

echo "Install requirements..." > $OUTPUT_LOG
pip3 install -r requirements.txt

# Run the Python training script
echo "Run SAM finetuning..." > $OUTPUT_LOG
if command -v python3 &> /dev/null; then
    python3 eval.py --arch SAM --model $CPT --data $EVAL_DIR_KVASIR > $OUTPUT_LOG 2> $ERROR_LOG
    python3 eval.py --arch SAM --model $CPT --data $EVAL_DIR_CLINIC > $OUTPUT_LOG 2> $ERROR_LOG
else
    python eval.py --arch SAM --model $CPT --data $EVAL_DIR_KVASIR > $OUTPUT_LOG 2> $ERROR_LOG
    python eval.py --arch SAM --model $CPT --data $EVAL_DIR_CLINIC > $OUTPUT_LOG 2> $ERROR_LOG
fi

EOF

# Cleanup scratch (if applicable)
rm -rf $SCRATCH_DIR
