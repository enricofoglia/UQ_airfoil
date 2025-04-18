import os
import os.path as osp 
import copy
import random

from dataclasses import dataclass

import torch 

import numpy as np
import matplotlib.pyplot as plt

from dataset import AirfRANSDataset
import utils 
import metrics 

# global paths
ROOT_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/AirfRANS'
OUT_DIR = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/scripts/paper/UQ_airfoil/out'

# set safe types
utils.set_safe_types()

# set random seed for repeatability
utils.set_seed(0)

# set maplotlib global parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"],
    "text.latex.preamble": r"\usepackage{amsmath,amsfonts}\usepackage[cm]{sfmath}",
    'axes.linewidth' : 2,
    'lines.linewidth' : 2,
    'axes.labelsize' : 16,
    'xtick.labelsize' : 14,
    'ytick.labelsize' : 14,
    'axes.titlesize' : 16,
    'legend.fontsize': 14
})

# load dataset
N = 25 # number of Fourier modes
n_points = 250 # number of points in the graphs
train_dataset = AirfRANSDataset('full', True, DATA_DIR, normalize=True, force_reload=False)
train_glob = train_dataset.get_global()

mean = train_dataset.glob_mean
std = train_dataset.glob_std
test_dataset = AirfRANSDataset('full', False, DATA_DIR, normalize=(mean,std), force_reload=False)
test_glob = test_dataset.get_global()

# set model parameters
hidden = 64
blocks = 4
ens_size = 5
p = 0.1 # dropout probability
z0 = 5.0 # zigzag constant

@dataclass
class Args:
    def __init__(self):
       pass

global_args = Args()
global_args.blocks = blocks
global_args.hidden = hidden
global_args.fourier = N

ens_params = copy.deepcopy(global_args)
ens_params.ens_size = ens_size
ens_params.model_type = 'ensemble'

zigzag_params = copy.deepcopy(global_args)
zigzag_params.z0 = z0
zigzag_params.model_type = 'zigzag'

dropout_params = copy.deepcopy(global_args)
dropout_params.drop_prob = p
dropout_params.model_type = 'dropout'

sgld_params = copy.deepcopy(global_args)
sgld_params.ens_size = 5
sgld_params.model_type = 'ensemble'

# initialize models
ensemble = utils.ModelFactory.create(ens_params).to('cpu')
zigzag = utils.ModelFactory.create(zigzag_params).to('cpu')
dropout = utils.ModelFactory.create(dropout_params).to('cpu')
sgld = utils.ModelFactory.create(sgld_params).to('cpu')

# load weights
ens_dir = osp.join(OUT_DIR, 'trained_models', 'ensemble')
for n, fname in enumerate(os.listdir(ens_dir)):
    ensemble[n].load_state_dict(torch.load(osp.join(ens_dir, fname), map_location='cpu')).eval()

zigzag.load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'zigzag_zigzag_200_800_64_25_16_0.001_0.5_4.pt'), map_location='cpu')).eval()

dropout.load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'def_dropout_200_800_64_25_16_0.001_0.33_4_0.1.pt'), map_location='cpu'))

sgld_dir = osp.join(OUT_DIR, 'trained_models', 'SGDL_warm_restarts')
for n, fname in enumerate(os.listdir(sgld_dir)):
    sgld[n].load_state_dict(torch.load(osp.join(sgld_dir, fname), map_location='cpu')).eval()

# print model parameters
print( '+----------------------------+')
print( '| Model type   | # of params |')
print( '+--------------+-------------+')
n_params = utils.count_parameters(ensemble[0])
print(f'| Ensemble     | {n_params:>11} | x {ens_size} !')
n_params = utils.count_parameters(zigzag)
print(f'| ZigZag       | {n_params:>11} |')
n_params = utils.count_parameters(dropout)
print(f'| Dropout      | {n_params:>11} |')
n_params = utils.count_parameters(sgld[0])
print(f'| SGDL         | {n_params:>11} | x {sgld_params["n_models"]} !')
print( '+--------------+-------------+')

# get predictions
graph = random.choice(test_dataset)
x = graph.x[:, 0].numpy()
gt = graph.y.numpy()

pred_ens, var_ens = ensemble(test_dataset, return_var=True)
pred_zz, var_zz = zigzag(test_dataset, return_var=True)
pred_dp, var_dp = dropout(test_dataset, return_var=True, T = 100)
pred_sgld, var_sgld = sgld(test_dataset, return_var=True)

def plot_single_prediction(ax, x, gt, cp, var, label):
    std = var**0.5
    ax.plot(x, gt, color='k', label='Ground truth')
    ax.plot(x, cp, color='tab:blue', label='Prediction')

    midpoint = np.argmin(x)  # This should find the trailing edge

    # Upper surface (first half of the data)
    upper_x = x[:midpoint+1]
    upper_pred = gt.squeeze()[:midpoint+1]
    upper_std = std.squeeze()[:midpoint+1]

    # Lower surface (second half of the data)
    lower_x = x[midpoint:]
    lower_pred = gt.squeeze()[midpoint:]
    lower_std = std.squeeze()[midpoint:]

    # Use proper zorder to ensure the uncertainty band doesn't disappear when crossing
    ax.fill_between(upper_x, upper_pred+2*upper_std, upper_pred-2*upper_std, 
                alpha=0.3, color='tab:blue', zorder=1)
    ax.fill_between(lower_x, lower_pred+2*lower_std, lower_pred-2*lower_std, 
                alpha=0.3, color='tab:blue', label=r'uncertainty ($\pm 2\sigma$)', zorder=2)
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.text(0.9, 0.9, label, transform=ax.transAxes, 
            horizontalalignment='right', verticalalignment='bottom')
    ax.grid(which='major', linewidth=0.5, color='gainsboro', zorder=0)

fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True, sharex=True)
plot_single_prediction(axs[0], x, gt, pred_ens, var_ens, 'Ensemble')
plot_single_prediction(axs[1], x, gt, pred_zz, var_zz, 'ZigZag')
plot_single_prediction(axs[2], x, gt, pred_dp, var_dp, 'Dropout')
plot_single_prediction(axs[3], x, gt, pred_sgld, var_sgld, 'SGDL')
axs[-1].legend()
axs[0].set_ylabel(r'$c_p$')
for ax in axs:
    ax.set_xlabel(r'$x/c$')

plt.show()