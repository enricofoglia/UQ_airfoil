#!/bin/bash

# Path to the Python script
PYTHON_SCRIPT="post_processing.py"

# arguments
epochs=200
samples=800
hidden=64
fourier=25
batch=16
model_type="ensemble"
z0=5.0
p=0.1
identifier="warm_restarts"
lr=0.0001
gamma=0.5
blocks=4
ens_size=5



# Run the Python script and log output
LOG_DIR="/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/scripts/paper/UQ_airfoil/out/logs/"
LOG_FILE="${LOG_DIR}postproc_${identifier}_${model_type}_${epochs}_${samples}_${hidden}_${fourier}_${batch}_${lr}_${gamma}.logs"

# Model path on Pando
MODEL_DIR="/home/daep/e.foglia/Documents/02_UQ/01_airfrans/03_results/trained_models/"
MODEL_NAME="warm_restarts_simple_250_800_64_25_16_0.001_0.33_4_1.0SGLD_0.pt"
MODEL_PATH="${MODEL_DIR}${MODEL_NAME}"

# Local model path
LOCAL_MODEL_DIR="/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/scripts/paper/UQ_airfoil/out/trained_models/"
LOCAL_MODEL_PATH="${LOCAL_MODEL_DIR}${MODEL_NAME}"

echo "Starting post-processing script" > $LOG_FILE

# Download model
# scp e.foglia@pando:${MODEL_PATH} ./model.pt
cp ${LOCAL_MODEL_PATH} ./model.pt

python3 $PYTHON_SCRIPT -e $epochs -s $samples -i $hidden -f $fourier -b $batch -p $p -n $ens_size --z0 $z0 --model_type $model_type --identifier $identifier --lr $lr --gamma $gamma >> $LOG_FILE 2>&1

# Remove the model
rm ./model.pt

# Exit with the status of the Python script
exit $?
