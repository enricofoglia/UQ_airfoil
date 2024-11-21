import os
import argparse

import time

import torch 
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ExponentialLR

from torch_geometric.loader import DataLoader

from torch_geometric.transforms import Distance
from torchvision import transforms

from sklearn.model_selection import train_test_split

from dataset import UniformSampling, FourierEpicycles, TangentVec, AirfRANSDataset, XFoilDataset
from model import EncodeProcessDecode, ZigZag, Ensemble, MCDropout
from training import Trainer, EnsembleTrainer, SGLD, PowerDecayLR
from utils import set_seed, Parser, ModelFactory

parser = Parser(print=True)
args = parser.args
# set seed for reproducibility
# set_seed(42)

# debug: track down anomaly
# torch.autograd.set_detect_anomaly(True)

N = args.fourier
n_points = 250
pre_transform = transforms.Compose((UniformSampling(n=n_points), FourierEpicycles(n=N), TangentVec(), Distance()))

root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/AirfRANS' # local
# root = '/home/daep/e.foglia/Documents/02_UQ/01_airfrans/01_data/' # pando
train_dataset = AirfRANSDataset('full',train=True, root=root, normalize=True, pre_transform=pre_transform, force_reload=False)
mean = train_dataset.glob_mean
std = train_dataset.glob_std
test_dataset = AirfRANSDataset('full',train=False, root=root, normalize=(mean, std), pre_transform=pre_transform, force_reload=False)

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

model = ModelFactory.create(args).to(device)

# model = ZigZag(
#             node_features=n,
#             edge_features=3,
#             hidden_features=args.hidden,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=0,
#             z0=-1.0, latent=True
#             ).to(device)

# model = Ensemble(
#             n_models=5,
#             node_features=n,
#             edge_features=3,
#             hidden_features=args.hidden,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=0,
#             ).to(device)

# model = MCDropout(
#             node_features=n,
#             edge_features=3,
#             hidden_features=args.hidden,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=0,
#             dropout=True,
#             p=0.1
#             ).to(device)
# loss = MSELoss()
loss = lambda y, pred: n_samples/args.batch*torch.mean((y-pred)**2) # !!!

initial_lr = 5e-3
final_lr = 1e-4
epochs = args.epochs
gamma = (final_lr/initial_lr)**(1/epochs)

if args.model_type ==  'ensemble':
    trainer = EnsembleTrainer(
        epochs=epochs,
        ensemble=model,
        optimizer='adam',
        optim_kwargs={'lr':initial_lr},
        loss_fn=loss,
        scheduler='exponential',
        scheduler_kwargs={'gamma':1/2,
                          'a':1e-5, 'b':1}
    )

else:
    trainer = Trainer(
        epochs=epochs,
        model=model,
        optimizer=SGLD,
        optim_kwargs={'lr':initial_lr,
                      'weight_decay': 1.0},
        loss_fn=loss,
        scheduler=PowerDecayLR,
        scheduler_kwargs={'gamma':1/2,
                          'a':1e-5, 'b':1},
        device=device,
        mcmc=True,
        save_start=3,
        save_rate=1
    )


# out_dir = '/home/daep/e.foglia/Documents/02_UQ/01_airfrans/03_results' # pando
out_dir = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/scripts/paper/UQ_airfoil/out'

model_name = f"{args.identifier}_{args.model_type}_{args.epochs}_{args.samples}_{args.hidden}_{args.fourier}_{args.batch}"


tic = time.time()
trainer.fit(train_loader, test_loader, os.path.join(out_dir,'trained_models',f'{model_name}.pt'))
toc = time.time()

formatted_time = time.strftime("%H:%M:%S", time.gmtime(toc-tic))
print( '----------------------------')
print(f' Elapsed time | {formatted_time}')
print( '----------------------------')

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
plt.savefig(os.path.join(out_dir,f'training_history_{model_name}.png'), dpi=300)

fig, ax = plt.subplots()
ax.semilogy(trainer.lr_history)
ax.set_xlabel('epoch')
ax.set_ylabel('learning rate $l_r$')
ax.set_title('Learing rate history')
plt.savefig(os.path.join(out_dir,f'lr_history_{model_name}.png'), dpi=300)
