import random

from tqdm import tqdm

import numpy as np

import torch 
from torchvision import transforms

from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.transforms import Distance

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from dataset import XFoilDataset, FourierEpicycles, TangentVec, UniformSampling, AirfRANSDataset
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

# n = 20
# pre_transform = FourierEpicycles(n=n, cat=False)

# root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/airfoils/train_shapes'
# dataset = XFoilDataset(root, normalize=True, pre_transform=pre_transform,
#                        force_reload=True)

N = 25
n_points = 250
pre_transform = transforms.Compose((UniformSampling(n=n_points), FourierEpicycles(n=N), TangentVec(), Distance()))

# pre_transform = FourierEpicycles(n=N)


# root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/airfoils/train_shapes'
# dataset = XFoilDataset(root, pre_transform=pre_transform, force_reload=True)
# root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/AirfRANS' # pando
root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/AirfRANS' # local

train_dataset = AirfRANSDataset('full', True, root, normalize=True, pre_transform=pre_transform, force_reload=True)
mean = train_dataset.glob_mean
std = train_dataset.glob_std
test_dataset = AirfRANSDataset('scarce', False, root, normalize=(mean,std), pre_transform=pre_transform, force_reload=True)

print(f'len dataset = {len(test_dataset)}')

# model = EncodeProcessDecode(
#             node_features=3+n,
#             edge_features=3,
#             hidden_features=64,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=1
#             )

model = ZigZag(
            node_features=N+2+2+1,
            edge_features=3,
            hidden_features=64,
            n_blocks=6,
            out_nodes=1,
            out_glob=0,
            z0=-2.0, latent =False
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

model.load_state_dict(torch.load('../../out/test_full_airfrans.pt',
                                 map_location=torch.device('cpu')))
# for n,single_model in enumerate(model):
    # single_model.load_state_dict(torch.load(f'../out/ensemble/ensemble_{n}.pt'))

n_params = count_parameters(model)
print( '+---------------------------------+')
print(f'| Total parameter count: {n_params:8d} |')
print( '+---------------------------------+')

fig, ax = plt.subplots(2,2, sharex=True, sharey=False, layout='constrained',
                       figsize=(9,6))

# randomize sample selection
import time 
tic = int(time.time())
random.seed(tic)

indices = random.sample(range(len(test_dataset)),k=4,)
print(f'| Selected indices {indices}')
print( '+---------------------------------+')

indices[0] = 22

for i, ind in enumerate(indices):
    row = i//2
    col = i%2
    graph = test_dataset[ind]
    with torch.no_grad():
        if model.kind == 'dropout':
            pred, var = model(graph, T=50, return_var=True)
        else: 
            pred, var = model(graph, return_var=True)


    std = torch.sqrt(var)


    ax[row,col].plot(graph.pos[:,0], graph.y, color='k', label='ground truth')
    ax[row,col].plot(graph.pos[:,0], pred.squeeze(), color='tab:blue', linestyle='--', label='prediction')
    ax[row, col].fill_between(graph.pos[:,0], pred.squeeze()+std.squeeze(), pred.squeeze()-std.squeeze(), alpha=0.6)
    ax[row,col].set_ylim(ax[row,col].get_ylim()[::-1])
    if row == 1:
        ax[row,col].set_xlabel(r'$x/c$ [-]')
    if col == 0:
        ax[row,col].set_ylabel(r'$c_p$ [-]')
    
ax[0,1].legend()

# for ind, graph in tqdm(enumerate(train_dataset), total=len(train_dataset)):
#     fig, ax = plt.subplots()
#     ax.plot(graph.pos[:,0], graph.pos[:,1], 'o-')
#     ax.set_title(f'index {ind}')
#     plt.savefig(f'../out/sample{ind}.png')
#     plt.close()

# single plot with uncertainty
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(graph.pos[:,0], graph.y, color='k', label='ground truth')
ax.plot(graph.pos[:,0], pred.squeeze(), color='tab:blue', linestyle='--', label='prediction')
ax.fill_between(graph.pos[:,0], pred.squeeze()+std.squeeze(), pred.squeeze()-std.squeeze(), alpha=0.6)
ax.set_xlabel(r'$x/c$ [-]')
ax.set_ylabel(r'$c_p$ [-]')
ax.set_ylim(ax.get_ylim()[::-1])

# add calibration plot
auce, p_err, p_pred = auce_plot(graph.y.numpy(),pred.squeeze().numpy(),
                                std.squeeze().numpy(), plot=False, get_values=True)
left, bottom, width, height = 0.65, 0.55, 0.23, 0.3
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(p_err, p_pred)
ax2.fill_between(p_err, p_pred, p_err, color='tab:blue', alpha=0.3)
ax2.plot([0,1],[0,1],'k--')
ax2.set_ylabel('True probability', fontsize=12, labelpad=1.0)
ax2.set_xlabel('Predicted probability', fontsize=12, labelpad=1.0)
ax2.xaxis.set_tick_params(labelsize=10, direction='in')
ax2.yaxis.set_tick_params(labelsize=10, direction='in')
ax2.set_xlim([0,1])
ax2.set_ylim([0,1])
ax2.text(0.05, 0.85, f'AUCE={auce:.2f}', fontsize=12)


preds = []
gt    = []
std_list = []
with torch.no_grad():
    for graph in tqdm(test_dataset, desc='Processing dataset ...'):
        if model.kind == 'dropout':
            pred, var = model(graph, T=50, return_var=True)
        else: 
            pred, var = model(graph, return_var=True)
        gt.append(graph.y.numpy())
        preds.append(pred.numpy())
        std_list.append(torch.sqrt(var).numpy())

preds = np.concatenate(preds).squeeze()
gt = np.concatenate(gt).squeeze()
std = np.concatenate(std_list).squeeze()

r2 = r2_score(gt, preds)
print( '-------------------')
print(f' R2 score : {r2:>5.3f}')
print( '-------------------')

fig, ax = plt.subplots()
ax.scatter(preds[::100], gt[::100], alpha=0.5, s=10)
ax.plot(gt[::100],gt[::100], 'k--', label='perfect fit')
ax.set_xlabel('predicted')
ax.set_ylabel('true')
ax.set_title(f'Correlation plot; $R^2$ score = {r2:.2f}')

# auce
auce_plot(gt[::10], preds[::10], std[::10])
ece_plot(gt[::10], preds[::10], std[::10], B=8, binning='quantile')

plt.show()

