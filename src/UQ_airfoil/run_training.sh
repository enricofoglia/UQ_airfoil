epochs=20
samples=800
hidden=64
fourier=25
batch=16
model_type="simple"
z0=5.0
p=0.1
identifier="wn_map"
lr=0.001
gamma=0.35
init="he"
temp=0.001
blocks=4
prior_std=1.0


# Run the Python script and log output
LOG_DIR="/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/scripts/paper/UQ_airfoil/out/logs/"
LOG_FILE="training_${identifier}_${model_type}_${epochs}_${samples}_${hidden}_${fourier}_${batch}_${lr}_${gamma}.logs"

python3 main.py -e $epochs -s $samples -i $hidden -f $fourier -b $batch -p $p --z0 $z0 --model_type $model_type --identifier $identifier --lr $lr --gamma $gamma --prior_std $prior_std --init $init --temp $temp --blocks $blocks >> $LOG_FILE 2>&1

mv $LOG_FILE $LOG_DIR 