import os
import argparse

import torch 
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ExponentialLR

from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.loader import DataLoader

from torch_geometric.transforms import Distance
from torchvision import transforms

from sklearn.model_selection import train_test_split

from dataset import UniformSampling, FourierEpicycles, TangentVec, AirfRANSDataset, XFoilDataset
from model import EncodeProcessDecode, ZigZag, Ensemble, MCDropout
from training import Trainer, EnsembleTrainer
from utils import set_seed, Parser

parser = Parser(print=True)
args = parser.args
# set seed for reproducibility
set_seed(42)

# debug: track down anomaly
# torch.autograd.set_detect_anomaly(True)
# n = 20
# pre_transform = FourierEpicycles(n=n, cat=False)

# root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/airfoils/train_shapes'
# dataset = XFoilDataset(root, normalize=True, pre_transform=pre_transform,
#                        force_reload=True)

N = args.fourier
n_points = 250
pre_transform = transforms.Compose((UniformSampling(n=n_points), FourierEpicycles(n=N), TangentVec(), Distance()))

root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/AirfRANS' # local
# root = '/home/daep/e.foglia/Documents/02_UQ/01_airfrans/01_data/' # pando
train_dataset = AirfRANSDataset('full',train=True, root=root, normalize=True, pre_transform=pre_transform, force_reload=False)
mean = train_dataset.glob_mean
std = train_dataset.glob_std
test_dataset = AirfRANSDataset('full',train=False, root=root, normalize=(mean, std), pre_transform=pre_transform, force_reload=False)
# random train-test split
# train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
# train_set = dataset[train_idx]
# test_set = dataset[test_idx]

n_samples = args.samples

# create loaders
train_loader = DataLoader(train_dataset[:n_samples], batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

# total number of features N + 2 + 2 + 1 (Fourier+global params+pos+curvature)
n = N+2+2+1
# model = EncodeProcessDecode(
#             node_features=3+n,
#             edge_features=3,
#             hidden_features=64,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=1
#             )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print( '----------------------------')
print(f' Available device: {device}')
print( '----------------------------')

model = ZigZag(
            node_features=n,
            edge_features=3,
            hidden_features=args.hidden,
            n_blocks=6,
            out_nodes=1,
            out_glob=0,
            z0=-2.0, latent=False
            ).to(device)

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
# loss = MSELoss()
loss = lambda y, pred: torch.mean((y-pred)**2)

initial_lr = 5e-3
final_lr = 1e-4
epochs = args.epochs
gamma = (final_lr/initial_lr)**(1/epochs)

trainer = Trainer(
    epochs=epochs,
    model=model,
    optimizer=Adam,
    optim_kwargs={'lr':initial_lr},
    loss_fn=loss,
    scheduler=ExponentialLR,
    scheduler_kwargs={'gamma':gamma},
    device=device
)

# trainer = EnsembleTrainer(
#     epochs=epochs,
#     ensemble=model,
#     optimizer='adam',
#     optim_kwargs={'lr':initial_lr},
#     loss_fn=loss,
#     scheduler='exponential',
#     scheduler_kwargs={'gamma':gamma}
# )

# out_dir = '/home/daep/e.foglia/Documents/02_UQ/01_airfrans/03_results' # pando
out_dir = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/scripts/paper/UQ_airfoil/out'
trainer.fit(train_loader, test_loader, os.path.join(out_dir,'trained_models/test_full_airfrans.pt'))
# torch.save(train_idx, os.path.join(out_dir,'train_idx.pt'))
# torch.save(test_idx, os.path.join(out_dir,'test_idx.pt'))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.semilogy(trainer.training_history, label='training')
ax.semilogy(trainer.test_history, '--', label='testing')
ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel(r'loss $\mathcal{L}(\theta)$')
ax.set_title('Training history')
plt.savefig(os.path.join(out_dir,'training_history.png'), dpi=300)

fig, ax = plt.subplots()
ax.semilogy(trainer.lr_history)
ax.set_xlabel('epoch')
ax.set_ylabel('learning rate $l_r$')
ax.set_title('Learing rate history')
plt.savefig(os.path.join(out_dir,'lr_history.png'), dpi=300)
