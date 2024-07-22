import random

from tqdm import tqdm

import torch 

from torch_geometric.nn.aggr import SoftmaxAggregation

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from dataset import XFoilDataset, FourierEpicycles
from model import EncodeProcessDecode, ZigZag, Ensemble
from utils import count_parameters
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
    'ytick.labelsize' : 14
})
# =================================================
n = 30
pre_transform = FourierEpicycles(n=n)

root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/airfoils/train_shapes'
dataset = XFoilDataset(root, normalize=True, pre_transform=pre_transform)

avg = dataset.avg 
std = dataset.std

# model = EncodeProcessDecode(
#             node_features=3+n,
#             edge_features=3,
#             hidden_features=64,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=1
#             )

# model = ZigZag(
#             node_features=3+n,
#             edge_features=3,
#             hidden_features=64,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=1,
#             z0=-3.0
#             )

model = Ensemble(
            n_models=5,
            node_features=3+n,
            edge_features=3,
            hidden_features=64,
            n_blocks=6,
            out_nodes=1,
            out_glob=1,
            )

# model.load_state_dict(torch.load('out/zigzag.pt'))
for n,single_model in enumerate(model):
    single_model.load_state_dict(torch.load(f'out/ensemble/ensemble_{n}.pt'))

n_params = count_parameters(model)
print( '+---------------------------------+')
print(f'| Total parameter count: {n_params:8d} |')
print( '+---------------------------------+')

fig, ax = plt.subplots(2,2, sharex=True, sharey=False, layout='constrained')
indices = random.choices(range(len(dataset)),k=4)

for i, ind in enumerate(indices):
    row = i//2
    col = i%2
    graph = dataset[ind]
    with torch.no_grad():
        pred, pred_glob = model(graph)
    pred_glob = pred_glob*std + avg
    y_glob = graph.y_glob*std + avg


    ax[row,col].scatter(graph.pos[:,0], graph.y, c='none', edgecolors='tab:blue',
            label='ground truth')
    ax[row,col].scatter(graph.pos[:,0], pred, c='k', marker='x', label='prediction')
    ax[row,col].set_ylim(ax[row,col].get_ylim()[::-1])
    if row == 1:
        ax[row,col].set_xlabel(r'$x/c$ [-]')
    if col == 0:
        ax[row,col].set_ylabel(r'$c_p$ [-]')
    ax[row,col].set_title(f'Efficiency: pred {pred_glob[0,0]:.2f} | true {y_glob:.2f}')
ax[0,1].legend()

preds = []
gt    = []
with torch.no_grad():
    for graph in tqdm(dataset, desc='Processing dataset ...'):
        _, pred_glob = model(graph)
        gt.append(graph.y_glob.item())
        preds.append(pred_glob.item())

r2 = r2_score(gt, preds)

fig, ax = plt.subplots()
ax.scatter(preds, gt, alpha=0.7)
ax.plot(gt,gt, 'k--', label='perfect fit')
ax.set_xlabel('predicted')
ax.set_ylabel('true')
ax.set_title(f'Correlation plot; $R^2$ score = {r2:.2f}')
plt.show()

