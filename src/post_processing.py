import random

from tqdm import tqdm

import numpy as np

import torch 

from torch_geometric.nn.aggr import SoftmaxAggregation

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from dataset import XFoilDataset, FourierEpicycles
from model import EncodeProcessDecode, ZigZag, Ensemble, MCDropout
from utils import count_parameters, set_seed
from metrics import auce_plot, ece_plot
# =================================================
# Matplotlib settings
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
# =================================================

set_seed(42)

n = 10
pre_transform = FourierEpicycles(n=n, cat=False)

root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/airfoils/train_shapes'
dataset = XFoilDataset(root, normalize=True, pre_transform=pre_transform,
                       force_reload=True)

# load train-test split as in training
train_idx = torch.load('../out/train_idx.pt')
test_idx = torch.load('../out/test_idx.pt')
train_dataset = dataset[train_idx]
test_dataset  = dataset[test_idx]

avg_data = dataset.avg 
std_data = dataset.std

# model = EncodeProcessDecode(
#             node_features=3+n,
#             edge_features=3,
#             hidden_features=64,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=1
#             )

model = ZigZag(
            node_features=n,
            edge_features=3,
            hidden_features=64,
            n_blocks=6,
            out_nodes=1,
            out_glob=1,
            z0=-10.0
            )

# model = Ensemble(
#             n_models=5,
#             node_features=3+n,
#             edge_features=3,
#             hidden_features=64,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=1,
#             )

# model = MCDropout(
#             node_features=3+n,
#             edge_features=3,
#             hidden_features=64,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=1,
#             dropout=True,
#             p=0.1
#             )

model.load_state_dict(torch.load('../out/zigzag_eigshapes_xonly_nopos.pt'))
# for n,single_model in enumerate(model):
#     single_model.load_state_dict(torch.load(f'out/ensemble/ensemble_{n}.pt'))

n_params = count_parameters(model)
print( '+---------------------------------+')
print(f'| Total parameter count: {n_params:8d} |')
print( '+---------------------------------+')

fig, ax = plt.subplots(2,2, sharex=True, sharey=False, layout='constrained',
                       figsize=(9,6))
indices = random.choices(range(len(test_dataset)),k=4)
print(f'| Selected indices {indices}')
print( '+---------------------------------+')

for i, ind in enumerate(indices):
    row = i//2
    col = i%2
    graph = test_dataset[ind]
    with torch.no_grad():
        if model.kind == 'dropout':
            pred, pred_glob, var, var_glob = model(graph, T=50, return_var=True)
        else: 
            pred, pred_glob, var, var_glob = model(graph, return_var=True)
    pred_glob = pred_glob*std_data + avg_data
    y_glob = graph.y_glob*std_data + avg_data

    std = torch.sqrt(var)
    std_glob = torch.sqrt(var_glob)
    std_glob = std_glob*std_data

    ax[row,col].scatter(graph.pos[:,0], graph.y, c='k', marker='x', label='ground truth')
    ax[row,col].scatter(graph.pos[:,0], pred.squeeze(), c='none', edgecolor='tab:blue', 
                       label='prediction')
    ax[row,col].set_ylim(ax[row,col].get_ylim()[::-1])
    if row == 1:
        ax[row,col].set_xlabel(r'$x/c$ [-]')
    if col == 0:
        ax[row,col].set_ylabel(r'$c_p$ [-]')
    ax[row,col].set_title(f'Efficiency: pred {pred_glob[0,0]:.2f}$\pm${std_glob[0,0]:.2f} | true {y_glob:.2f}')
ax[0,1].legend()

# of the last sample plot uncertainty
fig, ax = plt.subplots(figsize=(14,7))
ax.scatter(graph.pos[:,0], graph.y, c='k', marker='x', label='ground truth',zorder=2)
ax.errorbar(graph.pos[:,0], pred.squeeze(), yerr = std.squeeze(), fmt='o', c='tab:blue', mec='k', 
                        ecolor='k', capsize=5, linewidth=1, label='prediction',zorder=1)
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_xlabel(r'$x/c$ [-]')
ax.set_ylabel(r'$c_p$ [-]')
ax.set_title(f'Efficiency: pred {pred_glob[0,0]:.2f}$\pm${std_glob[0,0]:.2f} | true {y_glob:.2f}')
# add calibration plot
auce, p_err, p_pred = auce_plot(graph.y.numpy(),pred.squeeze().numpy(),
                                std.squeeze().numpy(), plot=False, get_values=True)
left, bottom, width, height = 0.65, 0.55, 0.23, 0.3
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(p_err, p_pred)
ax2.fill_between(p_err, p_pred, p_err, color='tab:blue', alpha=0.3)
ax2.plot([0,1],[0,1],'k--',label='perfect calibration')
ax2.set_ylabel('True probability', fontsize=12, labelpad=1.0)
ax2.set_xlabel('Predicted probability', fontsize=12, labelpad=1.0)
ax2.xaxis.set_tick_params(labelsize=10, direction='in')
ax2.yaxis.set_tick_params(labelsize=10, direction='in')
ax2.set_xlim([0,1])
ax2.set_ylim([0,1])
ax2.text(0.05, 0.85, f'AUCE={auce:.2f}', fontsize=12)
ax.legend(loc='lower right')


preds = []
gt    = []
std_list = []
with torch.no_grad():
    for graph in tqdm(test_dataset, desc='Processing dataset ...'):
        if model.kind == 'dropout':
            _, pred_glob, _, var_glob = model(graph, T=50, return_var=True)
        else: 
            _, pred_glob, _, var_glob = model(graph, return_var=True)
        gt.append(graph.y_glob.item())
        preds.append(pred_glob.item())
        std_list.append(torch.sqrt(var_glob).item())

r2 = r2_score(gt, preds)

fig, ax = plt.subplots()
ax.scatter(preds, gt, alpha=0.7)
ax.plot(gt,gt, 'k--', label='perfect fit')
ax.set_xlabel('predicted')
ax.set_ylabel('true')
ax.set_title(f'Correlation plot; $R^2$ score = {r2:.2f}')

# auce
auce_plot(np.array(gt), np.array(preds), np.array(std_list))
ece_plot(np.array(gt), np.array(preds), np.array(std_list),
         B=8, binning='quantile')

plt.show()

