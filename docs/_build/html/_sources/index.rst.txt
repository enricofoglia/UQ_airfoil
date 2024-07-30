.. UQ_airfoil documentation master file, created by
   sphinx-quickstart on Tue Jul 30 10:55:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

UQ_airfoil documentation
========================

**UQ_airfoil** is a python package that allows to build various kinds of Deep Learning models build on graphs to learn the pressure distribution and the aerodynamic efficiency of a dataset of 2D airfoils. Since no model is perfect, we provide various uncertainty quantification (UQ) methodologies to estimate the level of epistemic uncertainty of the neural network.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   src

Use example
-----------

Here is how you can use this package::

   from dataset import XFoilDataset, FourierEpicycles
   from model import ZigZag
   from training import Trainer, 

   pre_transform = FourierEpicycles(n=10, cat=False)

   root = '../data'
   dataset = XFoilDataset(root, normalize=True, pre_transform=pre_transform,
                        force_reload=True)

   # random train-test split
   train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2)
   train_set = dataset[train_idx]
   test_set = dataset[test_idx]

   # create loaders
   train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
   test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

   # initialize model
   model = ZigZag(
               node_features=n,
               edge_features=3,
               hidden_features=64,
               n_blocks=6,
               out_nodes=1,
               out_glob=1,
               z0=-2.0, latent=True
               )

   # initialize trainer
   loss = lambda y, pred: torch.mean((y-pred)**2)

   initial_lr = 5e-3
   final_lr = 1e-4
   epochs = 200
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

   # fit model
   trainer.fit(train_loader, test_loader, '../out/best_model.pt')
