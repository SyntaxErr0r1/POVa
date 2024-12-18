#!/bin/bash
#PBS -N test_script
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=15gb:scratch_local=5gb:ngpus=1:gpu_mem=24gb
#PBS -l walltime=12:00:00

## ===== Variables =====
USER_NAME=xmican10
HOME_DIR=/storage/brno2/home/$USER_NAME
PROJECT_DIR_NAME=POVa
PROJECT_DIR=$HOME_DIR/$PROJECT_DIR_NAME

DATASET_NAME=Kvasir-SEG_splitted

LOG_DIR=OUTPUTS-g-$DATASET_NAME-$MODEL_CFG-$PBS_JOBID #LOG_DIR=OUTPUTS-g-$MODEL_CFG-$PBS_JOBID
OUTPUT_PTH=$PROJECT_DIR/$LOG_DIR/test-$PBS_JOBID.pth
OUT_FILE=$PROJECT_DIR/$LOG_DIR/log-$PBS_JOBID.txt

DATASET_DIR=$HOME_DIR/datasets/$DATASET_NAME

## ===== Start of logs =====
mkdir -p $PROJECT_DIR/$LOG_DIR
echo "LOG_DIR: $LOG_DIR" 2>&1 >> $OUT_FILE
echo "PBS_JOBID: $PBS_JOBID" 2>&1 >> $OUT_FILE
echo "Hostname: $(hostname -f)" 2>&1 >> $OUT_FILE
echo "Scratch directory: $SCRATCHDIR" 2>&1 >> $OUT_FILE

## ===== IDK some necessary setup =====
export TMPDIR=$SCRATCHDIR
export PATH=$PATH:/storage/brno2/home/$USER_NAME/.local/bin:$HOME/.local/bin

## ===== PYTHON =====
echo "#----- PYTHON" 2>&1 >> $OUT_FILE
python3 -V 2>&1 >> $OUT_FILE
module add python/3.10.4-gcc-8.3.0-ovkjwzd
python3 -m ensurepip --upgrade
python3 -V 2>&1 >> $OUT_FILE
python3 -m pip --version 2>&1 >> $OUT_FILE

# Install requiremets
python3 -m pip install -r requirements.txt 2>&1 >> $OUT_FILE
echo ">> Done..." 2>&1 >> $OUT_FILE

## ===== Dataset copy =====
echo "#----- Dataset loading" 2>&1 >> $OUT_FILE
cp $DATASET_DIR $SCRATCHDIR
echo "#Dataset successfully copied into SCRATCHDIR" 2>&1 >> $OUT_FILE
