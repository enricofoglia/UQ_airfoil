import torch 
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ExponentialLR

from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split

from dataset import XFoilDataset, FourierEpicycles
from model import EncodeProcessDecode, ZigZag, Ensemble, MCDropout
from training import Trainer, EnsembleTrainer

# debug: track down anomaly
# torch.autograd.set_detect_anomaly(True)
n = 30
pre_transform = FourierEpicycles(n=n)

root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/airfoils/train_shapes'
dataset = XFoilDataset(root, normalize=True, pre_transform=pre_transform,
                       force_reload=True)

# random train-test split
train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_set = dataset[train_idx]
test_set = dataset[test_idx]

# create loaders
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

model = EncodeProcessDecode(
            node_features=3+n,
            edge_features=3,
            hidden_features=64,
            n_blocks=6,
            out_nodes=1,
            out_glob=1
            )

# model = ZigZag(
#             node_features=3+n,
#             edge_features=3,
#             hidden_features=64,
#             n_blocks=6,
#             out_nodes=1,
#             out_glob=1,
#             z0=-3.0
#             )

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
epochs = 10
gamma = (final_lr/initial_lr)**(1/epochs)

trainer = Trainer(
    epochs=epochs,
    model=model,
    optimizer=Adam,
    optim_kwargs={'lr':initial_lr},
    loss_fn=loss,
    scheduler=ExponentialLR,
    scheduler_kwargs={'gamma':gamma}
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

trainer.fit(train_loader, test_loader, 'out/simple_mlp.pt')

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.semilogy(trainer.training_history, label='training')
ax.semilogy(trainer.test_history, '--', label='testing')
ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel(r'loss $\mathcal{L}(\theta)$')
ax.set_title('Training history')
plt.savefig('out/training_history.png', dpi=300)

fig, ax = plt.subplots()
ax.semilogy(trainer.lr_history)
ax.set_xlabel('epoch')
ax.set_ylabel('learning rate $l_r$')
ax.set_title('Learing rate history')
plt.savefig('out/lr_history.png', dpi=300)
