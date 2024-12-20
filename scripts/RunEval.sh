#!/bin/bash

#PBS -N PolypSegmentationUnet
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=64gb:scratch_local=200gb:ngpus=1
#PBS -l walltime=1:00:00
#PBS -j oe

MODEL_PATH=./models/unet_segmentation.pth
EVAL_DATA_DIR=./datasets/merged/val

# Load necessary modules (adjust according to your cluster environment)
module load singularity

# Navigate to the working directory
cd $PBS_O_WORKDIR

# Prepare scratch space if needed (optional, adjust based on your requirements)
SCRATCH_DIR=$SCRATCHDIR
mkdir -p $SCRATCH_DIR

# Load and run the container
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:24.10-py3.SIF <<EOF

# Install segmentation_models.pytorch
pip3 install -U git+https://github.com/qubvel-org/segmentation_models.pytorch

pip3 install -r requirements.txt

mkdir -p $PBS_O_WORKDIR/logs/$PBS_JOBID

# Run the Python training script
if command -v python3 &> /dev/null; then
    python3 eval.py --model $MODEL_PATH --data $EVAL_DATA_DIR > $PBS_O_WORKDIR/logs/$PBS_JOBID/output.log 2> $PBS_O_WORKDIR/logs/$PBS_JOBID/error.log
else
    python  eval.py --model $MODEL_PATH --data $EVAL_DATA_DIR > $PBS_O_WORKDIR/logs/$PBS_JOBID/output.log 2> $PBS_O_WORKDIR/logs/$PBS_JOBID/error.log
fi

EOF

# Cleanup scratch (if applicable)
rm -rf $SCRATCH_DIR
