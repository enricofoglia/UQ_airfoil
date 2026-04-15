import os
import os.path as osp 
import copy
import random

from dataclasses import dataclass

import torch 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.isotonic import IsotonicRegression

from tqdm import tqdm
from scipy import stats
from scipy.integrate import simpson

from dataset import AirfRANSDataset
import utils 
import metrics 

from typing import (
    Optional,
    Union,
    Tuple,
)

# ===== Usefull functions =====
def plot_single_prediction(ax, x, gt, cp, var, label):
    std = var**0.5
    ax.plot(x, gt, color='k', label='Ground truth')
    ax.plot(x, cp, '--',color='tab:blue', label='Prediction')

    midpoint = np.argmin(x)  # This should find the trailing edge

    # Upper surface (first half of the data)
    upper_x = x[:midpoint+1]
    upper_pred = cp.squeeze()[:midpoint+1]
    upper_std = std.squeeze()[:midpoint+1]

    # Lower surface (second half of the data)
    lower_x = x[midpoint:]
    lower_pred = cp.squeeze()[midpoint:]
    lower_std = std.squeeze()[midpoint:]

    # Use proper zorder to ensure the uncertainty band doesn't disappear when crossing)
    ax.fill_between(upper_x, upper_pred+2*upper_std, upper_pred-2*upper_std, 
                alpha=0.3, color='tab:blue', zorder=1)
    ax.fill_between(lower_x, lower_pred+2*lower_std, lower_pred-2*lower_std, 
                alpha=0.3, color='tab:blue', label=r'uncertainty ($\pm 2\sigma$)', zorder=2)

    ax.text(0.9, 0.1, label, transform=ax.transAxes, 
            horizontalalignment='right', verticalalignment='bottom')
    ax.grid(which='major', linewidth=0.5, color='gainsboro', zorder=0)
    ax.set_ylim(ax.get_ylim()[::-1])

def correlation_plot(ax, gts, preds, label, resolution=100):
    r2 = r2_score(gts, preds)
    mse = np.mean((gts-preds)**2)

    ax.scatter(preds[::resolution], gts[::resolution], alpha=0.5, s=10)
    ax.plot([min(gts[::resolution]), max(gts[::resolution])],[min(gts[::resolution]), max(gts[::resolution])], 'k--')
    ax.text(0.9, 0.1, f'{label}: {r2:.2f}', transform=ax.transAxes, 
            horizontalalignment='right', verticalalignment='bottom')
    ax.set_xlim([min(gts[::resolution]), max(gts[::resolution])])
    ax.set_ylim([min(gts[::resolution]), max(gts[::resolution])])

    return r2, mse

def prediction_error_plot(ax, gts, preds, resolution=100):
    err = gts - preds
    ax.plot(preds[::resolution], err[::resolution], 'o', alpha=0.5, markersize=2)
    ax.plot([min(gts[::resolution]), max(gts[::resolution])], [0,0], 'k--')
    ax.set_xlim([min(gts[::resolution]), max(gts[::resolution])])


def auce_plot(ax, y:np.ndarray, preds:np.ndarray, std:np.ndarray,
              label:str=None, resolution:int=100,
              color:str='tab:blue', linestyle:str='-') -> Union[float, Tuple[float,np.ndarray,np.ndarray]]:
    err = y - preds
    abs_err = np.abs(err)

    # CDF and quantiles of the error
    abs_err_sorted = np.sort(abs_err)
    p_err = np.arange(0,len(abs_err))/(len(abs_err)-1)

    p_pred = []
    for p in p_err:
        q = stats.halfnorm.ppf(p, scale=std)
        count = np.sum(abs_err<=q)
        p_pred.append(count/len(q))
    p_pred = np.array(p_pred)

    # AUCE score
    indices = np.argsort(p_pred)
    auce = simpson(np.abs(p_err[indices] - p_pred[indices]),p_pred[indices])

    ax.plot(p_err[::resolution], p_pred[::resolution], color=color, linestyle=linestyle)
    ax.fill_between(p_err[::resolution], p_pred[::resolution], p_err[::resolution], color=color, alpha=0.3)
    ax.plot([0,1],[0,1],'k--')
    if label is not None:
        ax.text(0.9, 0.1, f'{label}:{auce:.3f}', transform=ax.transAxes, 
            horizontalalignment='right', verticalalignment='bottom')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    return auce 

def auce_plot_stats(ax, y: np.ndarray, preds: list, std: list,
              label: str = None, resolution: int = 100,
              color: str = 'tab:blue', linestyle: str = '-') -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Plot the AUCE (Area Under Calibration Error) with error bars/shaded regions.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    y : np.ndarray
        The true values
    preds : list of np.ndarray
        List of prediction arrays
    std : list of np.ndarray
        List of standard deviation arrays
    label : str, optional
        Label for the plot
    resolution : int, optional
        Resolution of the plot (skips points for cleaner visualization)
    color : str, optional
        Color for the plot
    linestyle : str, optional
        Line style for the plot
        
    Returns:
    --------
    float or tuple
        AUCE score and optionally arrays of p_err and p_pred
    """
    # Calculate AUCE for each prediction/std pair
    auce_scores = []
    all_p_err = []
    all_p_pred = []
    
    for pred_array, std_array in zip(preds, std):
        err = y[::resolution] - pred_array[::resolution]
        abs_err = np.abs(err)
        
        # CDF and quantiles of the error
        abs_err_sorted = np.sort(abs_err)
        p_err = np.arange(0, len(abs_err)) / (len(abs_err) - 1)
        
        p_pred = []
        for p in p_err:
            q = stats.halfnorm.ppf(p, scale=std_array[::resolution])
            count = np.sum(abs_err <= q)
            p_pred.append(count / len(q))
        p_pred = np.array(p_pred)
        
        # AUCE score
        indices = np.argsort(p_pred)
        auce = simpson(np.abs(p_err[indices] - p_pred[indices]), p_pred[indices])
        auce_scores.append(auce)
        
        all_p_err.append(p_err)
        all_p_pred.append(p_pred)
    
    # Calculate mean and std of p_pred across all predictions
    mean_p_err = np.mean(all_p_err, axis=0)
    mean_p_pred = np.mean(all_p_pred, axis=0)
    std_p_pred = np.std(all_p_pred, axis=0)
    
    # Calculate mean and std of AUCE scores
    mean_auce = np.mean(auce_scores)
    std_auce = np.std(auce_scores)
    
    # Plot mean line
    ax.plot(mean_p_err, mean_p_pred, 
            color=color, linestyle=linestyle, label=label)
    
    # Plot shaded region for standard deviation
    ax.fill_between(
        mean_p_err,
        np.maximum(0, mean_p_pred - std_p_pred),
        np.minimum(1, mean_p_pred + std_p_pred),
        color=color, alpha=0.2
    )
    
    # Reference line
    ax.plot([0, 1], [0, 1], 'k--')
    
    # Add label with mean AUCE and std
    if label is not None:
        ax.text(0.9, 0.1, f'{label}: {mean_auce:.3f} ± {std_auce:.3f}', 
                transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return mean_auce, std_auce, mean_p_err, mean_p_pred

def ece_plot(ax, y_test:np.ndarray, mu:np.ndarray, var:np.ndarray,
             label:str=None,
              B:Optional[int] = 20, binning:Optional[str]='equal',
              use_last:Optional[bool]=False,
              color:str='tab:blue', marker:str='o')->Union[float, Tuple[float,np.ndarray,np.ndarray]]:

    # sort and bin by increasing std
    indices = np.argsort(var)
    var = var[indices]
    y_test = y_test[indices]
    mu = mu[indices]
    if binning == 'equal':
        bin_edges = np.linspace(var.min(), var.max(), B + 1)
        bins = np.digitize(var, bin_edges)
    elif binning == 'quantile':
        bins = np.digitize(var, np.quantile(var, np.linspace(0,1,B + 1)))
    elif binning == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=B, random_state=0, init='k-means++', n_init=1).fit(var[..., None])
        bins = kmeans.labels_
    else:
        raise ValueError('binning must be equal or quantile')

    rmse = []
    rmv  = []
    rmv2 = []
    bin_size = []
    for i in np.unique(bins):
        y_sample = y_test[bins==i]
        bin_size.append(len(y_sample))
        pred_mu = mu[bins==i]
        pred_var = var[bins==i]

        rmse.append(np.sqrt(np.mean((y_sample.squeeze()-pred_mu.squeeze())**2)))
        rmv2.append(np.sqrt(np.var(np.abs(y_sample.squeeze()-pred_mu.squeeze()))))
        rmv.append(np.sqrt(np.mean(pred_var)))

    rmv = np.array(rmv)
    rmse = np.array(rmse)
    bin_size = np.array(bin_size)

    indices = np.argsort(rmv)
    rmv = rmv[indices]
    rmse = rmse[indices]
    bin_size = bin_size[indices]

    if not use_last:
        rmv = rmv[:-1]
        rmse = rmse[:-1]
        bin_size = bin_size[:-1]

    ece = np.sum(np.abs(rmv-rmse)*bin_size)/np.sum(bin_size) # true expectation over bins of different sizes
    
    cv = np.sqrt(np.sqrt(var).var())/np.mean(np.sqrt(var))

    
    ax.plot(rmv, rmse, color=color)
    ax.fill_between(rmv, rmse, rmv, color=color, alpha=0.3) 
    ax.plot([min(rmv), max(rmv)], [min(rmv), max(rmv)], 'k--')
    ax.scatter(rmv, rmse, s=bin_size*100/max(bin_size), edgecolors='k', c=color, marker=marker,zorder=5)
    if label is not None:
        ax.text(0.9, 0.1, label, transform=ax.transAxes,
            bbox={'boxstyle':'Round', 'alpha':0.75, 'color':'white'},
            horizontalalignment='right', verticalalignment='bottom')

    return ece, cv


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
    'axes.labelsize' : 18,
    'xtick.labelsize' : 16,
    'ytick.labelsize' : 16,
    'axes.titlesize' : 18,
    'legend.fontsize': 18,
    'font.size' : 16,
})
color1 = '#DDA853'
color2 = '#27548A'
color3 = '#183B4E'

# load dataset
N = 25 # number of Fourier modes
n_points = 250 # number of points in the graphs
train_dataset = AirfRANSDataset('full', True, DATA_DIR, normalize=True, force_reload=False)
train_glob = train_dataset.get_global()

mean = train_dataset.glob_mean
std = train_dataset.glob_std
test_dataset = AirfRANSDataset('full', False, DATA_DIR, normalize=(mean,std), force_reload=False)
test_glob = test_dataset.get_global()

# split test dataset into test and recalibration
test_dataset, recal_dataset = torch.utils.data.random_split(test_dataset, [int(len(test_dataset)*0.5), int(len(test_dataset)*0.5)])

# set model parameters
hidden = 64
blocks = 4
ens_size = 10
p = 0.1 # dropout probability
z0 = 5.0 # zigzag constant

@dataclass
class Args:
    # def __init__(self):
    #    pass
    blocks: int 
    hidden: int
    fourier: int

global_args = Args(blocks=blocks, hidden=hidden, fourier=N) 
# global_args.blocks = blocks
# global_args.hidden = hidden
# global_args.fourier = N

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
ens_dir = osp.join(OUT_DIR, 'trained_models', 'ensemble_wn')
for n, fname in enumerate(os.listdir(ens_dir)):
    ensemble[n].load_state_dict(torch.load(osp.join(ens_dir, fname), map_location='cpu'))

zigzag.load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'wn_zigzag_150_800_64_25_16_0.001_0.33_4_1.0_5.0.pt'), map_location='cpu'))

dropout.load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'wn_dropout_150_800_64_25_16_0.001_0.33_4_1.0.pt'), map_location='cpu'))

sgld_dir = osp.join(OUT_DIR, 'trained_models', 'SGLD_wr_wn')
for n, fname in enumerate(os.listdir(sgld_dir)):
    sgld[n].load_state_dict(torch.load(osp.join(sgld_dir, fname), map_location='cpu'))

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
print(f'| SGDL         | {n_params:>11} | x {sgld_params.ens_size} !')
print( '+--------------+-------------+')

# Plot prediction on a single graph
print( '+----------------------------+')
print( '| Single graph predictions   |')
print( '+----------------------------+')
from utils import compute_lift

for i in tqdm(range(1), desc='Single predictions ...'):
    graph = random.choice(test_dataset)
    x = graph.pos[:, 0].numpy()
    y = graph.pos[:, 1].numpy()
    gt = graph.y.numpy()

    with torch.no_grad():
        pred_ens, var_ens = ensemble(graph, return_var=True)
        pred_zz, var_zz = zigzag(graph, return_var=True)
        pred_dp, var_dp = dropout(graph, return_var=True, T = 10)
        pred_sgld, var_sgld = sgld(graph, return_var=True)

    # compute lift
    lift_ens = compute_lift( pred_ens.squeeze().numpy(), x, y)
    lift_zz = compute_lift( pred_zz.squeeze().numpy(), x, y)
    lift_sgld = compute_lift( pred_sgld.squeeze().numpy(), x, y)
    lift_dp = compute_lift( pred_dp.squeeze().numpy(), x, y)

    fig, axs = plt.subplots(2,2, figsize=(10, 8), sharey=False, sharex=True, layout='constrained')
    plot_single_prediction(axs[0,0], x, gt, pred_zz.numpy(), var_zz.numpy(), f'ZigZag; $c_L={lift_zz:.2f}$')
    plot_single_prediction(axs[0,1], x, gt, pred_sgld.numpy(), var_sgld.numpy(), f'SGLD; $c_L={lift_sgld:.2f}$')
    plot_single_prediction(axs[1,0], x, gt, pred_dp.numpy(), var_dp.numpy(), f'Dropout; $c_L={lift_dp:.2f}$')
    plot_single_prediction(axs[1,1], x, gt, pred_ens.numpy(), var_ens.numpy(), f'Ensemble; $c_L={lift_ens:.2f}$')

    axs[0,1].legend()
    axs[0,0].set_ylabel(r'$c_p$')
    axs[1,0].set_ylabel(r'$c_p$')
    axs[1,0].set_xlabel(r'$x/c$')
    axs[1,1].set_xlabel(r'$x/c$')

    plt.savefig(osp.join(OUT_DIR, 'figures', f'predictions_single_airfoil_square_{i}.pdf'), bbox_inches='tight')

plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to move on.")
plt.close('all') # all open plots are correctly closed after each run

# plot learning rate history
def cosine_annealing_with_warm_restarts(iteration, T_0, T_mult=1, eta_min=0, eta_max=1):
    T_cur = iteration
    T_i = T_0
    
    # Find which cycle we're in
    cycle = 0
    while T_cur >= T_i:
        T_cur -= T_i
        T_i = T_i * T_mult
        cycle += 1
    
    # Calculate the learning rate
    if T_i == 0:
        return eta_min
    
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))
x1 = np.linspace(0,100, 1000)
x2 = np.linspace(0,250, 1000)

lr1 = 1e-3*np.ones_like(x1)
lr2 = np.array([cosine_annealing_with_warm_restarts(i, 50, 1, 1e-6, 1e-3) for i in x2])

fig, ax = plt.subplots(figsize=(8, 3))
ax.semilogy(x1, lr1, color='tab:blue')
ax.semilogy(x2+100, lr2, color='tab:blue')
ax.scatter([150,200,250,300,350], [1e-6] * 5, marker='o', color='k', edgecolor='white', label='sample points',zorder=5)
ax.axvspan(100, 350, facecolor='k', alpha=0.1)
ax.set_xlabel('epoch')
ax.set_ylabel(r'$\epsilon^{(k)}$')
ax.grid(which='both', linewidth=0.5, color='gainsboro',zorder=0)
ax.set_xlim([0,350])
ax.legend()

plt.savefig(osp.join(OUT_DIR, 'figures', 'lr_history.pdf'), bbox_inches='tight')

print( '+----------------------------+')
print( '| Statistics test dataset    |')
print( '+----------------------------+')

preds_ens = []
preds_zz  = []
preds_dp  = []
preds_sgld = []

stds_ens = []
stds_zz  = []
stds_dp  = []
stds_sgld = []

gt    = []

cl_preds_ens = []
cl_preds_zz  = []
cl_preds_dp  = []
cl_preds_sgld = []

cl_stds_ens = []
cl_stds_zz  = []
cl_stds_dp  = []
cl_stds_sgld = []

cl_gt = []

with torch.no_grad():
    for graph in tqdm(test_dataset, desc='Processing test dataset ...'):

        pred_ens, var_ens = ensemble(graph, return_var=True)
        pred_zz, var_zz = zigzag(graph, return_var=True)
        pred_dp, var_dp = dropout(graph, return_var=True, T = 10)
        pred_sgld, var_sgld = sgld(graph, return_var=True)
        
        gt.append(graph.y.numpy())
        preds_ens.append(pred_ens.numpy())
        preds_zz.append(pred_zz.numpy())
        preds_dp.append(pred_dp.numpy())
        preds_sgld.append(pred_sgld.numpy())

        stds_ens.append(torch.sqrt(var_ens).numpy())
        stds_zz.append(torch.sqrt(var_zz).numpy())
        stds_dp.append(torch.sqrt(var_dp).numpy())
        stds_sgld.append(torch.sqrt(var_sgld).numpy())

        # compute lift
        cl_gt.append(compute_lift(graph.y.squeeze().numpy(), graph.pos[:, 0].numpy(), graph.pos[:, 1].numpy()))
        cl_preds_ens.append(compute_lift(pred_ens.squeeze().numpy(), graph.pos[:, 0].numpy(), graph.pos[:, 1].numpy()))
        cl_preds_zz.append(compute_lift(pred_zz.squeeze().numpy(), graph.pos[:, 0].numpy(), graph.pos[:, 1].numpy()))
        cl_preds_dp.append(compute_lift(pred_dp.squeeze().numpy(), graph.pos[:, 0].numpy(), graph.pos[:, 1].numpy()))
        cl_preds_sgld.append(compute_lift(pred_sgld.squeeze().numpy(), graph.pos[:, 0].numpy(), graph.pos[:, 1].numpy()))

        # compute integrated std
        cl_stds_ens.append(np.sqrt(compute_lift(var_ens.squeeze().numpy(), graph.pos[:, 0].numpy(), graph.pos[:, 1].numpy())))
        cl_stds_zz.append(np.sqrt(compute_lift(var_zz.squeeze().numpy(), graph.pos[:, 0].numpy(), graph.pos[:, 1].numpy())))
        cl_stds_dp.append(np.sqrt(compute_lift(var_dp.squeeze().numpy(), graph.pos[:, 0].numpy(), graph.pos[:, 1].numpy())))
        cl_stds_sgld.append(np.sqrt(compute_lift(var_sgld.squeeze().numpy(), graph.pos[:, 0].numpy(), graph.pos[:, 1].numpy())))

        
        

preds_ens = np.concatenate(preds_ens).squeeze()
preds_zz  = np.concatenate(preds_zz).squeeze()
preds_dp  = np.concatenate(preds_dp).squeeze()
preds_sgld = np.concatenate(preds_sgld).squeeze()

stds_ens = np.concatenate(stds_ens).squeeze()
stds_zz  = np.concatenate(stds_zz).squeeze()
stds_dp  = np.concatenate(stds_dp).squeeze()
stds_sgld = np.concatenate(stds_sgld).squeeze()

gts = np.concatenate(gt).squeeze()

cl_preds_ens = np.array(cl_preds_ens).squeeze()
cl_preds_zz  = np.array(cl_preds_zz).squeeze()
cl_preds_dp  = np.array(cl_preds_dp).squeeze()
cl_preds_sgld = np.array(cl_preds_sgld).squeeze()

cl_stds_ens = np.array(cl_stds_ens).squeeze()
cl_stds_zz  = np.array(cl_stds_zz).squeeze()
cl_stds_dp  = np.array(cl_stds_dp).squeeze()
cl_stds_sgld = np.array(cl_stds_sgld).squeeze()

cl_gt = np.array(cl_gt).squeeze()


fig, axs = plt.subplots(2,4, figsize=(14,6), sharey=False, sharex=True, layout='constrained')
r2_zz, mse_zz = correlation_plot(axs[0,0], gts, preds_zz, 'ZigZag')
r2_sgld, mse_sgld = correlation_plot(axs[0,1], gts, preds_sgld, 'SGLD')
r2_dp, mse_dp = correlation_plot(axs[0,2], gts, preds_dp, 'Dropout')
r2_ens, mse_ens = correlation_plot(axs[0,3], gts, preds_ens, 'Ensemble')
prediction_error_plot(axs[1,0], gts, preds_zz)
prediction_error_plot(axs[1,1], gts, preds_sgld)
prediction_error_plot(axs[1,2], gts, preds_dp)
prediction_error_plot(axs[1,3], gts, preds_ens)

for ax in axs[1,:]:
    ax.set_xlabel(r'$c_{p,\text{pred}}$')

axs[0,0].set_ylabel(r'$c_{p,\text{true}}$')
axs[1,0].set_ylabel(r'Residual ($c_{p,\text{pred}} - c_{p,\text{true}}$)')
plt.savefig(osp.join(OUT_DIR, 'figures', 'correlation_plot_airfrans.pdf'), bbox_inches='tight')

# =======================================================================
# plot correlation and prediction error lift

fig, axs = plt.subplots(2,4, figsize=(14,6), sharey=False, sharex=True, layout='constrained')
r2_zz, mse_zz = correlation_plot(axs[0,0], cl_gt, cl_preds_zz, 'ZigZag', resolution=1)
r2_sgld, mse_sgld = correlation_plot(axs[0,1], cl_gt, cl_preds_sgld, 'SGLD', resolution=1)
r2_dp, mse_dp = correlation_plot(axs[0,2], cl_gt, cl_preds_dp, 'Dropout', resolution=1)
r2_ens, mse_ens = correlation_plot(axs[0,3], cl_gt, cl_preds_ens, 'Ensemble', resolution=1)
prediction_error_plot(axs[1,0], cl_gt, cl_preds_zz, resolution=1)
prediction_error_plot(axs[1,1], cl_gt, cl_preds_sgld, resolution=1)
prediction_error_plot(axs[1,2], cl_gt, cl_preds_dp, resolution=1)
prediction_error_plot(axs[1,3], cl_gt, cl_preds_ens, resolution=1)

for ax in axs[1,:]:
    ax.set_xlabel(r'$c_{L,\text{pred}}$')

axs[0,0].set_ylabel(r'$c_{L,\text{true}}$')
axs[1,0].set_ylabel(r'Residual ($c_{L,\text{pred}} - c_{L,\text{true}}$)')
plt.savefig(osp.join(OUT_DIR, 'figures', 'correlation_plot_airfrans_cl.pdf'), bbox_inches='tight')

# =======================================================================


fig, axs = plt.subplots(2,2, figsize=(10, 8), sharey=False, sharex=True, layout='constrained')
auce_zz = auce_plot(axs[0,0], gts[::100], preds_zz[::100], stds_zz[::100], 'ZigZag', resolution=1)
auce_sgld = auce_plot(axs[0,1], gts[::100], preds_sgld[::100], stds_sgld[::100], 'SGLD', resolution=1)
auce_dp = auce_plot(axs[1,0], gts[::100], preds_dp[::100], stds_dp[::100], 'Dropout', resolution=1)
auce_ens = auce_plot(axs[1,1], gts[::100], preds_ens[::100], stds_ens[::100], 'Ensemble', resolution=1)

for ax in axs[1,:]:
    ax.set_xlabel(r'Predicted probability')
for ax in axs[:,0]:
    ax.set_ylabel(r'True probability')
plt.savefig(osp.join(OUT_DIR, 'figures', 'AUCE_airfrans_square.pdf'), bbox_inches='tight')

fig, axs = plt.subplots(1,4, figsize=(14,4), sharey=False, sharex=True, layout='constrained')
auce_zz = auce_plot(axs[0], gts[::100], preds_zz[::100], stds_zz[::100], 'ZigZag', resolution=1)
auce_sgld = auce_plot(axs[1], gts[::100], preds_sgld[::100], stds_sgld[::100], 'SGLD', resolution=1)
auce_dp = auce_plot(axs[2], gts[::100], preds_dp[::100], stds_dp[::100], 'Dropout', resolution=1)
auce_ens = auce_plot(axs[3], gts[::100], preds_ens[::100], stds_ens[::100], 'Ensemble', resolution=1)

for ax in axs:
    ax.set_xlabel(r'Predicted probability')
axs[0].set_ylabel(r'True probability')
plt.savefig(osp.join(OUT_DIR, 'figures', 'AUCE_airfrans_line.pdf'), bbox_inches='tight')

# ===========================================
# Figures NeurIPS
# ===========================================

fig, axs = plt.subplots(1,3, figsize=(12, 4), sharey=False, sharex=True, layout='constrained')
auce_zz = auce_plot(axs[0], gts[::10], preds_zz[::10], stds_zz[::10], 'ZigZag', resolution=1)
# auce_sgld = auce_plot(axs[1], gts[::10], preds_sgld[::100], stds_sgld[::100], 'SGLD', resolution=1)
auce_dp = auce_plot(axs[1], gts[::10], preds_dp[::10], stds_dp[::10], 'Dropout', resolution=1)
auce_ens = auce_plot(axs[2], gts[::10], preds_ens[::10], stds_ens[::10], 'Ensemble', resolution=1)
for ax in axs:
    ax.set_xlabel(r'Predicted probability')
    ax.grid(True, zorder=0, linestyle='--')

axs[0].set_ylabel(r'True probability')
plt.savefig(osp.join(OUT_DIR, 'figures', 'AUCE_airfrans_line_neurips.pdf'), bbox_inches='tight')


fig, axs = plt.subplots(2,2, figsize=(10, 8), sharey=False, sharex=False, layout='constrained')
ece_zz, cv_zz = ece_plot(axs[0,0], gts, preds_zz, stds_zz**2, 'ZigZag', binning='quantile',B=10)
ece_sgld, cv_sgld = ece_plot(axs[0,1], gts, preds_sgld, stds_sgld**2, 'SGLD', binning='quantile',B=10)
ece_dp, cv_dp = ece_plot(axs[1,0], gts, preds_dp, stds_dp**2, 'Dropout', binning='quantile',B=10)
ece_ens, cv_ens = ece_plot(axs[1,1], gts, preds_ens, stds_ens**2, 'Ensemble', binning='quantile',B=10)

for ax in axs[1,:]:
    ax.set_xlabel(r'$\mathsf{RMV}$')
for ax in axs[:,0]:
    ax.set_ylabel(r'$\mathsf{RMSE}$')
plt.savefig(osp.join(OUT_DIR, 'figures', 'ECE_airfrans_square.pdf'), bbox_inches='tight')

fig, axs = plt.subplots(1,4, figsize=(14, 4), sharey=False, sharex=False, layout='constrained')
ece_zz, cv_zz = ece_plot(axs[0], gts, preds_zz, stds_zz**2, 'ZigZag', binning='quantile',B=10)
ece_sgld, cv_sgld = ece_plot(axs[1], gts, preds_sgld, stds_sgld**2, 'SGLD', binning='quantile',B=10)
ece_dp, cv_dp = ece_plot(axs[2], gts, preds_dp, stds_dp**2, 'Dropout', binning='quantile',B=10)
ece_ens, cv_ens = ece_plot(axs[3], gts, preds_ens, stds_ens**2, 'Ensemble', binning='quantile',B=10)

for ax in axs:
    ax.set_xlabel(r'$\mathsf{RMV}$')
axs[0].set_ylabel(r'$\mathsf{RMSE}$')
plt.savefig(osp.join(OUT_DIR, 'figures', 'ECE_airfrans_line.pdf'), bbox_inches='tight')

# =======================================================================
# plot AUCE lift

fig, axs = plt.subplots(1,4, figsize=(14,4), sharey=False, sharex=True, layout='constrained')
auce_zz = auce_plot(axs[0], cl_gt, cl_preds_zz, cl_stds_zz, 'ZigZag', resolution=1)
auce_sgld = auce_plot(axs[1], cl_gt, cl_preds_sgld, cl_stds_sgld, 'SGLD', resolution=1)
auce_dp = auce_plot(axs[2], cl_gt, cl_preds_dp, cl_stds_dp, 'Dropout', resolution=1)
auce_ens = auce_plot(axs[3], cl_gt, cl_preds_ens, cl_stds_ens, 'Ensemble', resolution=1)

for ax in axs:
    ax.set_xlabel(r'Predicted probability')
axs[0].set_ylabel(r'True probability')
plt.savefig(osp.join(OUT_DIR, 'figures', 'AUCE_airfrans_line_cl.pdf'), bbox_inches='tight')


plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to move on.")
plt.close('all') # all open plots are correctly closed after each run

# ===== Recalibration =====
print( '+----------------------------+')
print( '| Recalibration              |')
print( '+----------------------------+')

gt_recal = []
preds_ens_recal = []
preds_zz_recal  = []
preds_dp_recal  = []
preds_sgld_recal = []
vars_ens_recal = []
vars_zz_recal  = []
vars_dp_recal  = []
vars_sgld_recal = []
with torch.no_grad():
    for graph in tqdm(recal_dataset, desc='Processing recalibration dataset ...'):
        pred_ens_recal, var_ens_recal = ensemble(graph, return_var=True)
        pred_zz_recal, var_zz_recal = zigzag(graph, return_var=True)
        pred_dp_recal, var_dp_recal = dropout(graph, return_var=True, T = 10)
        pred_sgld_recal, var_sgld_recal = sgld(graph, return_var=True)
        
        gt_recal.append(graph.y.numpy())
        preds_ens_recal.append(pred_ens_recal.numpy())
        preds_zz_recal.append(pred_zz_recal.numpy())
        preds_dp_recal.append(pred_dp_recal.numpy())
        preds_sgld_recal.append(pred_sgld_recal.numpy())

        vars_ens_recal.append(var_ens_recal.numpy())
        vars_zz_recal.append(var_zz_recal.numpy())
        vars_dp_recal.append(var_dp_recal.numpy())
        vars_sgld_recal.append(var_sgld_recal.numpy())
        

preds_ens_recal = np.concatenate(preds_ens_recal).squeeze()
preds_zz_recal  = np.concatenate(preds_zz_recal).squeeze()
preds_dp_recal  = np.concatenate(preds_dp_recal).squeeze()
preds_sgld_recal = np.concatenate(preds_sgld_recal).squeeze()

vars_ens_recal = np.concatenate(vars_ens_recal).squeeze()
vars_zz_recal  = np.concatenate(vars_zz_recal).squeeze()
vars_dp_recal  = np.concatenate(vars_dp_recal).squeeze()
vars_sgld_recal = np.concatenate(vars_sgld_recal).squeeze()
gt_recal = np.concatenate(gt_recal).squeeze()

# Temperature scaling
intercept = False
ts_zz = metrics.TemperatureScaling(intercept)
ts_zz.fit(preds_zz_recal, vars_zz_recal, gt_recal)
ts_ens = metrics.TemperatureScaling(intercept)
ts_ens.fit(preds_ens_recal, vars_ens_recal, gt_recal)
ts_dp = metrics.TemperatureScaling(intercept)
ts_dp.fit(preds_dp_recal, vars_dp_recal, gt_recal)
ts_sgld = metrics.TemperatureScaling(intercept)
ts_sgld.fit(preds_sgld_recal, vars_sgld_recal, gt_recal)

ts_zz_ece = metrics.TemperatureScaling(intercept)
ts_zz_ece.fit(preds_zz_recal, vars_zz_recal, gt_recal, loss='ece')
ts_ens_ece = metrics.TemperatureScaling(intercept)
ts_ens_ece.fit(preds_ens_recal, vars_ens_recal, gt_recal, loss='ece')
ts_dp_ece = metrics.TemperatureScaling(intercept)
ts_dp_ece.fit(preds_dp_recal, vars_dp_recal, gt_recal, loss='ece')
ts_sgld_ece = metrics.TemperatureScaling(intercept)
ts_sgld_ece.fit(preds_sgld_recal, vars_sgld_recal, gt_recal, loss='ece')


fig, axs = plt.subplots(2,2, figsize=(10, 8), sharey=False, sharex=False, layout='constrained')
auce_zz = auce_plot(axs[0,0], gts[::100], preds_zz[::100], np.sqrt(ts_zz.predict(stds_zz[::100]**2)), 'ZigZag', resolution=1)
auce_sgld = auce_plot(axs[0,1], gts[::100], preds_sgld[::100], np.sqrt(ts_sgld.predict(stds_sgld[::100]**2)), 'SGLD', resolution=1)
auce_dp = auce_plot(axs[1,0], gts[::100], preds_dp[::100], np.sqrt(ts_dp.predict(stds_dp[::100]**2)), 'Dropout', resolution=1)
auce_ens = auce_plot(axs[1,1], gts[::100], preds_ens[::100], np.sqrt(ts_ens.predict(stds_ens[::100]**2)), 'Ensemble', resolution=1)
for ax in axs[1,:]:
    ax.set_xlabel(r'Predicted probability')
for ax in axs[:,0]:
    ax.set_ylabel(r'True probability')
plt.savefig(osp.join(OUT_DIR, 'figures', 'recalibrated_AUCE_airfrans_square.pdf'), bbox_inches='tight')

fig, axs = plt.subplots(1,4, figsize=(14,4), sharey=False, sharex=True, layout='constrained')
auce_zz = auce_plot(axs[0], gts[::100], preds_zz[::100], np.sqrt(ts_zz.predict(stds_zz[::100]**2)), 'ZigZag', resolution=1)
auce_sgld = auce_plot(axs[1], gts[::100], preds_sgld[::100], np.sqrt(ts_sgld.predict(stds_sgld[::100]**2)), 'SGLD', resolution=1)
auce_dp = auce_plot(axs[2], gts[::100], preds_dp[::100], np.sqrt(ts_dp.predict(stds_dp[::100]**2)), 'Dropout', resolution=1)
auce_ens = auce_plot(axs[3], gts[::100], preds_ens[::100], np.sqrt(ts_ens.predict(stds_ens[::100]**2)), 'Ensemble', resolution=1)

for ax in axs:
    ax.set_xlabel(r'Predicted probability')
axs[0].set_ylabel(r'True probability')
plt.savefig(osp.join(OUT_DIR, 'figures', 'recalibrated_AUCE_airfrans_line.pdf'), bbox_inches='tight')

fig, axs = plt.subplots(1,3, figsize=(12,4), sharey=False, sharex=True, layout='constrained')
auce_zz = auce_plot(axs[0], gts[::10], preds_zz[::10], np.sqrt(ts_zz_ece.predict(stds_zz[::10]**2)), 'ZigZag', resolution=1)
auce_dp = auce_plot(axs[1], gts[::10], preds_dp[::10], np.sqrt(ts_dp_ece.predict(stds_dp[::10]**2)), 'Dropout', resolution=1)
auce_ens = auce_plot(axs[2], gts[::10], preds_ens[::10], np.sqrt(ts_ens_ece.predict(stds_ens[::10]**2)), 'Ensemble', resolution=1)
auce_zz = auce_plot(axs[0], gts[::10], preds_zz[::10], stds_zz[::10],  resolution=1, color='tab:red', linestyle='-.')
auce_dp = auce_plot(axs[1], gts[::10], preds_dp[::10], stds_dp[::10],  resolution=1, color='tab:red', linestyle='-.')
auce_ens = auce_plot(axs[2], gts[::10], preds_ens[::10], stds_ens[::10], resolution=1, color='tab:red', linestyle='-.')

for ax in axs:
    ax.set_xlabel(r'Predicted probability')
    ax.grid(True, zorder=0, linestyle='--')

axs[0].set_ylabel(r'True probability')
line1 = plt.Line2D([0], [0], color='tab:blue', lw=2)
line2 = plt.Line2D([0], [0], color='tab:red', lw=2, linestyle='-.')
fig.legend([line1, line2], ['Recalibrated', 'Original'] ,
           loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15))
plt.savefig(osp.join(OUT_DIR, 'figures', 'recalibrated_AUCE_airfrans_line_neurips.pdf'), bbox_inches='tight')

fig, axs = plt.subplots(2,2, figsize=(10, 8), sharey=False, sharex=False, layout='constrained')
ece_zz, cv_zz = ece_plot(axs[0,0], gts, preds_zz, ts_zz.predict(stds_zz**2), f'ZigZag, T={ts_zz.T:.2f}', binning='quantile',B=10)
ece_sgld, cv_sgld = ece_plot(axs[0,1], gts, preds_sgld, ts_sgld.predict(stds_sgld**2), f'SGLD, T={ts_sgld.T:.2f}', binning='quantile',B=10)
ece_dp, cv_dp = ece_plot(axs[1,0], gts, preds_dp, ts_dp.predict(stds_dp**2), f'Dropout, T={ts_dp.T:.2f}', binning='quantile',B=10)
ece_ens, cv_ens = ece_plot(axs[1,1], gts, preds_ens, ts_ens.predict(stds_ens**2), f'Ensemble, T={ts_ens.T:.2f}', binning='quantile',B=10)
for ax in axs[1,:]:
    ax.set_xlabel(r'$\mathsf{RMV}$')
for ax in axs[:,0]:
    ax.set_ylabel(r'$\mathsf{RMSE}$')
plt.savefig(osp.join(OUT_DIR, 'figures', 'recalibrated_ECE_airfrans_square.pdf'), bbox_inches='tight')

fig, axs = plt.subplots(1,4, figsize=(14, 4), sharey=False, sharex=False, layout='constrained')
ece_zz, cv_zz = ece_plot(axs[0], gts, preds_zz, ts_zz.predict(stds_zz**2), 'ZigZag', binning='quantile',B=10)
ece_sgld, cv_sgld = ece_plot(axs[1], gts, preds_sgld, ts_sgld.predict(stds_sgld**2), 'SGLD', binning='quantile',B=10)
ece_dp, cv_dp = ece_plot(axs[2], gts, preds_dp, ts_dp.predict(stds_dp**2), 'Dropout', binning='quantile',B=10)
ece_ens, cv_ens = ece_plot(axs[3], gts, preds_ens, ts_ens.predict(stds_ens**2), 'Ensemble', binning='quantile',B=10)

for ax in axs:
    ax.set_xlabel(r'$\mathsf{RMV}$')
axs[0].set_ylabel(r'$\mathsf{RMSE}$')
plt.savefig(osp.join(OUT_DIR, 'figures', 'recalibrated_ECE_airfrans_line.pdf'), bbox_inches='tight')

fig.suptitle('Recalibrated ECE')

# comparisons
fig, axs = plt.subplots(1,4, figsize=(14,4), sharey=False, sharex=False, layout='constrained')
print(f'+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+')
print(f'|{" "*10}| {"ZigZag":<8} | {"SGLD":<8} | {"Dropout":<8} | {"Ensemble":<8} |')
print(f'+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+')
auce_zz = auce_plot(axs[0], gts[::100], preds_zz[::100], stds_zz[::100], resolution=1, color=color1)
auce_sgld = auce_plot(axs[1], gts[::100], preds_sgld[::100], stds_sgld[::100], resolution=1, color=color1)
auce_dp = auce_plot(axs[2], gts[::100], preds_dp[::100], stds_dp[::100], resolution=1, color=color1)
auce_ens = auce_plot(axs[3], gts[::100], preds_ens[::100], stds_ens[::100], resolution=1, color=color1)
print(f'| {"Original":<8} | {auce_zz:8.4f} | {auce_sgld:8.4f} | {auce_dp:8.4f} | {auce_ens:8.4f} |')
auce_zz = auce_plot(axs[0], gts[::100], preds_zz[::100], np.sqrt(ts_zz.predict(stds_zz[::100]**2)), 'ZigZag', resolution=1, color=color2, linestyle='-.')
auce_sgld = auce_plot(axs[1], gts[::100], preds_sgld[::100], np.sqrt(ts_sgld.predict(stds_sgld[::100]**2)), 'SGLD', resolution=1, color=color2, linestyle='-.')
auce_dp = auce_plot(axs[2], gts[::100], preds_dp[::100], np.sqrt(ts_dp.predict(stds_dp[::100]**2)), 'Dropout', resolution=1, color=color2, linestyle='-.')
auce_ens = auce_plot(axs[3], gts[::100], preds_ens[::100], np.sqrt(ts_ens.predict(stds_ens[::100]**2)), 'Ensemble', resolution=1, color=color2, linestyle='-.')
print(f'| {"NLL":<8} | {auce_zz:8.4f} | {auce_sgld:8.4f} | {auce_dp:8.4f} | {auce_ens:8.4f} |')
auce_zz = auce_plot(axs[0], gts[::100], preds_zz[::100], np.sqrt(ts_zz_ece.predict(stds_zz[::100]**2)), 'ZigZag', resolution=1, color=color3, linestyle=':')
auce_sgld = auce_plot(axs[1], gts[::100], preds_sgld[::100], np.sqrt(ts_sgld_ece.predict(stds_sgld[::100]**2)), 'SGLD', resolution=1, color=color3, linestyle=':')
auce_dp = auce_plot(axs[2], gts[::100], preds_dp[::100], np.sqrt(ts_dp_ece.predict(stds_dp[::100]**2)), 'Dropout', resolution=1, color=color3, linestyle=':')
auce_ens = auce_plot(axs[3], gts[::100], preds_ens[::100], np.sqrt(ts_ens_ece.predict(stds_ens[::100]**2)), 'Ensemble', resolution=1, color=color3, linestyle=':')
print(f'| {"ECE":<8} | {auce_zz:8.4f} | {auce_sgld:8.4f} | {auce_dp:8.4f} | {auce_ens:8.4f} |')
print(f'+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+')
for ax in axs:
    ax.set_xlabel(r'Predicted probability')
axs[0].set_ylabel(r'True probability')
line1 = plt.Line2D([0], [0], color=color1, lw=2)
line2 = plt.Line2D([0], [0], color=color2, lw=2, linestyle='-.')
line3 = plt.Line2D([0], [0], color=color3, lw=2, linestyle=':')
fig.legend([line1, line2, line3], ['Original', 'Recalibrated, NLL', 'Recalibrated, ECE'], 
           loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15))
plt.savefig(osp.join(OUT_DIR, 'figures', 'recalibrated_AUCE_airfrans_line_comparison.pdf'), bbox_inches='tight')

fig, axs = plt.subplots(1,4, figsize=(14,4), sharey=False, sharex=False, layout='constrained')
print(f'+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+')
print(f'|{" "*10}| {"ZigZag":<8} | {"SGLD":<8} | {"Dropout":<8} | {"Ensemble":<8} |')
print(f'+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+')
ece_zz, cv_zz = ece_plot(axs[0], gts, preds_zz, stds_zz**2, binning='quantile',B=10, color=color1)
ece_sgld, cv_sgld = ece_plot(axs[1], gts, preds_sgld, stds_sgld**2, binning='quantile',B=10, color=color1)
ece_dp, cv_dp = ece_plot(axs[2], gts, preds_dp, stds_dp**2, binning='quantile',B=10, color=color1)
ece_ens, cv_ens = ece_plot(axs[3], gts, preds_ens, stds_ens**2, binning='quantile',B=10, color=color1)
print(f'| {"Original":<8} | {ece_zz:8.4f} | {ece_sgld:8.4f} | {ece_dp:8.4f} | {ece_ens:8.4f} |')
ece_zz, cv_zz = ece_plot(axs[0], gts, preds_zz, ts_zz.predict(stds_zz**2), 'ZigZag', binning='quantile',B=10,
color=color2, marker='s')
ece_sgld, cv_sgld = ece_plot(axs[1], gts, preds_sgld, ts_sgld.predict(stds_sgld**2), 'SGLD', binning='quantile',B=10,
color=color2, marker='s')
ece_dp, cv_dp = ece_plot(axs[2], gts, preds_dp, ts_dp.predict(stds_dp**2), 'Dropout', binning='quantile',B=10,
color=color2, marker='s')
ece_ens, cv_ens = ece_plot(axs[3], gts, preds_ens, ts_ens.predict(stds_ens**2), 'Ensemble', binning='quantile',B=10,
color=color2, marker='s')
print(f'| {"NLL":<8} | {ece_zz:8.4f} | {ece_sgld:8.4f} | {ece_dp:8.4f} | {ece_ens:8.4f} |')
ece_zz, cv_zz = ece_plot(axs[0], gts, preds_zz, ts_zz_ece.predict(stds_zz**2), 'ZigZag', binning='quantile',B=10,
color=color3, marker='v')
ece_sgld, cv_sgld = ece_plot(axs[1], gts, preds_sgld, ts_sgld_ece.predict(stds_sgld**2), 'SGLD', binning='quantile',B=10,
color=color3, marker='v')
ece_dp, cv_dp = ece_plot(axs[2], gts, preds_dp, ts_dp_ece.predict(stds_dp**2), 'Dropout', binning='quantile',B=10,
color=color3, marker='v')
ece_ens, cv_ens = ece_plot(axs[3], gts, preds_ens, ts_ens_ece.predict(stds_ens**2), 'Ensemble', binning='quantile',B=10,
color=color3, marker='v')
print(f'| {"ECE":<8} | {ece_zz:8.4f} | {ece_sgld:8.4f} | {ece_dp:8.4f} | {ece_ens:8.4f} |')
print(f'+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+{"-"*10}+')
# # Create an inset zoomed plot for the ZigZag plot (axs[0])
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

axins = zoomed_inset_axes(axs[0], zoom=4, loc='upper left')
ece_plot(axins, gts, preds_zz, stds_zz**2, binning='quantile', B=10, color=color1)
ece_plot(axins, gts, preds_zz, ts_zz.predict(stds_zz**2),  binning='quantile', B=10, color=color2, marker='s')
ece_plot(axins, gts, preds_zz, ts_zz_ece.predict(stds_zz**2), binning='quantile', B=10, color=color3, marker='v')

axins.set_xlim(0, 0.2)  # Zoom to RMV between 0 and 0.2
axins.set_ylim(0, 0.2)  # Zoom to RMSE between 0 and 0.2
axins.tick_params(labelleft=False, labelbottom=False)

mark_inset(axs[0], axins, loc1=3, loc2=4, fc="none", ec="0.5")

line1 = plt.Line2D([0], [0], color=color1, lw=2, marker='o', markeredgecolor='k')
line2 = plt.Line2D([0], [0], color=color2, lw=2, marker='s', markeredgecolor='k')
line3 = plt.Line2D([0], [0], color=color3, lw=2, marker='v', markeredgecolor='k')
for ax in axs:
    ax.set_xlabel(r'$\mathsf{RMV}$')

fig.legend([line1, line2, line3], ['Original', 'Recalibrated, NLL', 'Recalibrated, ECE'], 
           loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15))
axs[0].set_ylabel(r'$\mathsf{RMSE}$')
plt.savefig(osp.join(OUT_DIR, 'figures', 'recalibrated_ECE_airfrans_line_comparison.pdf'), bbox_inches='tight')

# get predictions
graph = random.choice(test_dataset)
x = graph.pos[:, 0].numpy()
gt = graph.y.numpy()

with torch.no_grad():
    pred_ens, var_ens = ensemble(graph, return_var=True)
    pred_zz, var_zz = zigzag(graph, return_var=True)
    pred_dp, var_dp = dropout(graph, return_var=True, T = 10)
    pred_sgld, var_sgld = sgld(graph, return_var=True)

fig, axs = plt.subplots(2,2, figsize=(10, 8), sharey=False, sharex=False, layout='constrained')
plot_single_prediction(axs[0,0], x, gt, pred_ens.numpy(), ts_ens_ece.predict(var_ens.numpy()), 'Ensemble')
plot_single_prediction(axs[0,1], x, gt, pred_zz.numpy(), ts_zz_ece.predict(var_zz.numpy()), 'ZigZag')
plot_single_prediction(axs[1,0], x, gt, pred_dp.numpy(), ts_dp_ece.predict(var_dp.numpy()), 'Dropout')
plot_single_prediction(axs[1,1], x, gt, pred_sgld.numpy(), ts_sgld_ece.predict(var_sgld.numpy()), 'SGLD')

fig.suptitle('Recalibrated predictions')
axs[0,1].legend()
axs[0,0].set_ylabel(r'$c_p$')
axs[1,0].set_ylabel(r'$c_p$')
axs[1,0].set_xlabel(r'$x/c$')
axs[1,1].set_xlabel(r'$x/c$')


plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run

# ======================================================================================================
# ||  NeurIPS 2025 statistical significance results                                                   ||
# ======================================================================================================

# set model parameters
hidden = 64
blocks = 4
ens_size = 5
p = 0.1 # dropout probability
z0 = 5.0 # zigzag constant

@dataclass
class Args:
    blocks: int 
    hidden: int
    fourier: int

global_args = Args(blocks=blocks, hidden=hidden, fourier=N) 


ens_params = copy.deepcopy(global_args)
ens_params.ens_size = ens_size
ens_params.model_type = 'ensemble'

zigzag_params = copy.deepcopy(global_args)
zigzag_params.z0 = z0
zigzag_params.model_type = 'zigzag'

dropout_params = copy.deepcopy(global_args)
dropout_params.drop_prob = p
dropout_params.model_type = 'dropout'


# initialize models
ens_list = [utils.ModelFactory.create(ens_params).to('cpu') for _ in range(5)]
zz_list = [utils.ModelFactory.create(zigzag_params).to('cpu') for _ in range(5)]
mcd_list = [utils.ModelFactory.create(dropout_params).to('cpu') for _ in range(5)]

# load weights
ens_dir = osp.join(OUT_DIR, 'trained_models', 'ensemble_wn')
for n, fname in enumerate(os.listdir(ens_dir)):
    ensemble[n].load_state_dict(torch.load(osp.join(ens_dir, fname), map_location='cpu'))

# load repeated zigzag
zz_list[0].load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'wn_zigzag_150_800_64_25_16_0.001_0.33_4_1.0_5.0.pt'), map_location='cpu'))
zz_list[1].load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'repeat_1_zigzag_150_800_64_25_16_0.005_0.33_4_1.0_5.0.pt'), map_location='cpu'))
zz_list[2].load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'repeat_2_zigzag_150_800_64_25_16_0.005_0.33_4_1.0_5.0.pt'), map_location='cpu'))
zz_list[3].load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'repeat_3_zigzag_150_800_64_25_16_0.005_0.33_4_1.0_5.0.pt'), map_location='cpu'))
zz_list[4].load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'repeat_4_zigzag_150_800_64_25_16_0.005_0.33_4_1.0_5.0.pt'), map_location='cpu'))

mcd_list[0].load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'wn_dropout_150_800_64_25_16_0.001_0.33_4_1.0.pt'), map_location='cpu'))
mcd_list[1].load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'repeat_1_dropout_150_800_64_25_16_0.005_0.33_4_1.0_5.0.pt'), map_location='cpu'))
mcd_list[2].load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'repeat_2_dropout_150_800_64_25_16_0.005_0.33_4_1.0_5.0.pt'), map_location='cpu'))
mcd_list[3].load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'repeat_3_dropout_150_800_64_25_16_0.005_0.33_4_1.0_5.0.pt'), map_location='cpu'))
mcd_list[4].load_state_dict(torch.load(osp.join(OUT_DIR, 'trained_models', 'repeat_4_dropout_150_800_64_25_16_0.005_0.33_4_1.0_5.0.pt'), map_location='cpu'))

for i in range(5):
    ens_dir = osp.join(OUT_DIR, 'trained_models', f'repeat_ensemble{i}')   
    for n, fname in enumerate(os.listdir(ens_dir)):
        ens_list[i][n].load_state_dict(torch.load(osp.join(ens_dir, fname), map_location='cpu'))

preds_ens = []
preds_zz  = []
preds_mcd  = []

stds_ens = []
stds_zz  = []
stds_mcd  = []

gt    = []
for graph in test_dataset:
    gt.append(graph.y.numpy())
gts = np.concatenate(gt).squeeze()

with torch.no_grad():
    for zigzag, dropout, ensemble in zip(zz_list, mcd_list, ens_list):
        preds_zz.append([])
        stds_zz.append([])

        preds_mcd.append([])
        stds_mcd.append([])

        preds_ens.append([])
        stds_ens.append([])

        for graph in tqdm(test_dataset, desc='Processing test dataset ...'):

            pred_zz, var_zz = zigzag(graph, return_var=True)
            pred_mcd, var_mcd = dropout(graph, return_var=True, T = 100)
            pred_ens, var_ens = ensemble(graph, return_var=True)

            preds_zz[-1].append(pred_zz.numpy())
            stds_zz[-1].append(torch.sqrt(var_zz).numpy())

            preds_mcd[-1].append(pred_mcd.numpy())
            stds_mcd[-1].append(torch.sqrt(var_mcd).numpy())

            preds_ens[-1].append(pred_ens.numpy())
            stds_ens[-1].append(torch.sqrt(var_ens).numpy())
       
for i in range(len(preds_zz)):
    preds_zz[i] =  np.concatenate(preds_zz[i]).squeeze() 
    stds_zz[i] =  np.concatenate(stds_zz[i]).squeeze() 
    preds_mcd[i] =  np.concatenate(preds_mcd[i]).squeeze() 
    stds_mcd[i] =  np.concatenate(stds_mcd[i]).squeeze() 
    preds_ens[i] =  np.concatenate(preds_ens[i]).squeeze()
    stds_ens[i] =  np.concatenate(stds_ens[i]).squeeze()

fig, axs = plt.subplots(1,3, figsize=(12, 4), sharey=True, sharex=False, layout='constrained')
auce_plot_stats(axs[0], gts, preds_zz, stds_zz, 'ZigZag', resolution=10)
auce_plot_stats(axs[1], gts, preds_mcd, stds_mcd, 'Dropout', resolution=10)
auce_plot_stats(axs[2], gts, preds_ens, stds_ens, 'Ensemble', resolution=10)
for ax in axs:
    ax.set_xlabel(r'Predicted probability')
    ax.grid(True, zorder=0, linestyle='--')

axs[0].set_ylabel(r'True probability')
plt.savefig(osp.join(OUT_DIR, 'figures', 'AUCE_airfrans_neurips_stats.pdf'), bbox_inches='tight')
plt.show()