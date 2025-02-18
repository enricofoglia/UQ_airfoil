#!/bin/bash

# Path to the Python script
PYTHON_SCRIPT="post_processing.py"

# arguments
epochs=300
samples=800
hidden=64
fourier=25
batch=16
model_type="simple"
z0=5.0
p=0.1
identifier="SGLD"
lr=0.00001
gamma=0.35


# Run the Python script and log output
LOG_DIR="/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/scripts/paper/UQ_airfoil/out/logs/"
LOG_FILE="${LOG_DIR}postproc_${identifier}_${model_type}_${epochs}_${samples}_${hidden}_${fourier}_${batch}_${lr}_${gamma}.logs"
python3 $PYTHON_SCRIPT -e $epochs -s $samples -i $hidden -f $fourier -b $batch -p $p --z0 $z0 --model_type $model_type --identifier $identifier --lr $lr --gamma $gamma >> $LOG_FILE 2>&1

# Exit with the status of the Python script
exit $?
