import os
import copy

from typing import (
    Optional,
    Callable,
    Dict,
    Any
)

import torch 
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import (
    LRScheduler,
    ExponentialLR,
    ReduceLROnPlateau
    )
 
from torch.nn import MSELoss, Module

from torch_geometric.loader import DataLoader

from tqdm import tqdm


class Trainer():
    def __init__(
            self,
            epochs:int,
            model:Module,
            optimizer:Optional[Optimizer]=Adam,
            loss_fn:Optional[Callable]=MSELoss(),
            device:Optional[str]='cpu',
            scheduler:Optional[LRScheduler]=None,
            optim_kwargs:Optional[Dict[str,Any]]=None,
            scheduler_kwargs:Optional[Dict[str,Any]]=None,
            ) -> None:
        
        self.epochs = epochs
        self.model = model
        if optim_kwargs is not None:
            self.optimizer = optimizer(self.model.parameters(), **optim_kwargs)
        else: 
            self.optimizer = optimizer(self.model.parameters())
        self.loss_fn = loss_fn
        self.device = device

        if scheduler is not None:
            if scheduler_kwargs is not None:
                self.scheduler = scheduler(self.optimizer, **scheduler_kwargs)
            else:
                self.scheduler = scheduler(self.optimizer)
        else:
            self.scheduler = None

        self.weight = 0.01

    def fit(self, train_loader:DataLoader, test_loader:DataLoader, savefile:Optional[str]='out/best_model.pt'):
        print( '+----------------------------------+')
        print( '| Training started ...             |')
        print( '+----------------------------------+')
        print(f'| Total number of epochs : {self.epochs:<8d}|')
        print( '+----------------------------------+')

        self.training_history = []
        self.test_history = []
        self.best_loss = torch.inf
        self.lr_history = []
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.training_history.append(self._train_epoch(train_loader, self.model))
            self.test_history.append(self._test_epoch(test_loader, self.model))
            print(f' Current loss = {self.training_history[-1]}')

            if self.scheduler is not None:
                self.scheduler.step()

            self.lr_history.append(self.optimizer.param_groups[0]['lr'])

            if self.test_history[-1] < self.best_loss:
                self.best_loss = self.test_history[-1]
                torch.save(self.model.state_dict(), savefile)
            

        print( '| Training ended                   |')
        print( '+----------------------------------+')
        print(f'| Final Loss : {self.training_history[-1]:<19.3f} |')
        print(f'| Bets Loss  : {self.best_loss:<19.3f} |')
        print( '+----------------------------------+')

    def _train_epoch(self, loader, model):
        batch_count = 0
        current_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            # reset gradients
            self.optimizer.zero_grad()

            # compute loss
            y = batch.y
            y_glob = batch.y_glob

            if model.kind == 'simple_gnn' or model.kind == 'dropout':
                pred, pred_glob  = model(batch)
                loss = (self.loss_fn(pred.squeeze(), y.squeeze()) 
                        + self.weight*self.loss_fn(pred_glob.squeeze(), y_glob.squeeze()))
            elif model.kind == 'zigzag':
                pred1, pred_glob1  = model(batch)
                pred2, pred_glob2  = model(batch, pred1)

                # the 0.5 helps comparing the loss of zigzag and the simple gnn
                loss = 0.5*(self.loss_fn(pred1.squeeze(), y.squeeze()) 
                        + self.weight*self.loss_fn(pred_glob1.squeeze(), y_glob.squeeze())
                        + self.loss_fn(pred2.squeeze(), y.squeeze()) 
                        + self.weight*self.loss_fn(pred_glob2.squeeze(), y_glob.squeeze()))
            else:
                raise ValueError(f'Unrecognized kind of model "{model.kind}"')
            # backpropagate and step
            loss.backward()
            self.optimizer.step()

            batch_count += 1
            current_loss += loss.item()

        return current_loss/batch_count
    
    def _test_epoch(self, loader, model):
        batch_count = 0
        current_loss = 0
        model.eval()
        for batch in loader:
            batch = batch.to(self.device)
            y = batch.y
            y_glob = batch.y_glob
            pred, pred_glob  = model(batch)
            loss = (self.loss_fn(pred.squeeze(), y.squeeze()) 
                    + self.weight*self.loss_fn(pred_glob.squeeze(), y_glob.squeeze()))

            batch_count += 1
            current_loss += loss.item()

        return current_loss/batch_count
    
class EnsembleTrainer(Trainer):
    def __init__(self,
                 epochs: int,
                 ensemble: Module,
                 optimizer: str | None = 'adam',
                 loss_fn: Callable[..., Any] | None = MSELoss(),
                 device: str | None = 'cpu',
                 scheduler: LRScheduler | None = None,
                 optim_kwargs: Dict[str, Any] | None = None,
                 scheduler_kwargs: Dict[str, Any] | None = None) -> None:
        
        self.epochs = epochs
        self.ensemble = ensemble
        
        self._optimizer = optimizer
        self.optim_kwargs = optim_kwargs
        self._scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        self.loss_fn = loss_fn
        self.device = device

        
        self.weight = 0.01

    # TODO: find a way to properly reinitialize optimizer and scheduler
    def _optimizer_init(self, model)->None:
        if self._optimizer == 'adam':
            self.optimizer = Adam
        else:
            raise ValueError(f"Chosen optimizer '{self._optimizer}' is not currently supported")

        if self.optim_kwargs is not None:
            self.optimizer = self.optimizer(model.parameters(), **self.optim_kwargs)
        else: 
            self.base_optimizer = self.optimizer(model.parameters())

    def _scheduler_init(self)->None:
        if self._scheduler == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau
        elif self._scheduler == 'exponential':
            self.scheduler = ExponentialLR
        else:
            raise ValueError(f"Chosen scheduler '{self._scheduler}' is not currently supported")
        
        if self.scheduler is not None:
            if self.scheduler_kwargs is not None:
                self.scheduler = self.scheduler(self.optimizer, **self.scheduler_kwargs)
            else:
                self.scheduler = self.scheduler(self.optimizer)
        else:
            self.base_scheduler = None


    def fit(self, train_loader:DataLoader, test_loader:DataLoader, savefile:Optional[str]='out/best_model.pt'):
        print( '+----------------------------------+')
        print( '| Training started ...             |')
        print( '+----------------------------------+')
        print(f'| Number of epochs : {self.epochs:<13d} |')
        print(f'| Number of models : {self.n_models:<13d} |')
        print(f'| Total number of epochs : {self.epochs*self.n_models:<8d}|')
        print( '+----------------------------------+')

        self.training_history = []
        self.test_history = []
        self.lr_history = []

        for n, model in enumerate(self.ensemble):
            print( '+----------------------------------+')
            print(f'| Model {n+1:>3d}/{self.n_models:<3d} ...{" "*16}|')
            print( '+----------------------------------+')
            
            training_history = []
            test_history = []
            lr_history = []
            self.best_loss = torch.inf
            
            self._optimizer_init(model)
            self._scheduler_init()

            for epoch in tqdm(range(self.epochs)):
                model.train()
                training_history.append(self._train_epoch(train_loader, model))
                test_history.append(self._test_epoch(test_loader, model))
                # print(f' Current loss = {self.training_history[-1]}')

                if self.scheduler is not None:
                    self.scheduler.step()

                lr_history.append(self.optimizer.param_groups[0]['lr'])

                if test_history[-1] < self.best_loss:
                    self.best_loss = test_history[-1]
                    self.save_checkpoint(model, savefile, n)

            print(f'| Final Loss : {training_history[-1]:<19.3f} |')
            print(f'| Best Loss  : {self.best_loss:<19.3f} |')
            print( '+----------------------------------+')

            self.training_history.append(training_history)
            self.test_history.append(test_history)
            self.lr_history.append(lr_history)

        print( '| Training ended                   |')
        print( '+----------------------------------+')
        

    @property
    def n_models(self):
        return len(self.ensemble)
    
    def save_checkpoint(self, model: Module, filename:str='out/best_model.pt', ensemble_num:int=-1)-> None:
        if ensemble_num > -1:
            dir_path = os.path.dirname(filename)
            file_name = os.path.basename(filename)

            # Create the new path by adding the folder
            new_dir_path = os.path.join(dir_path, 'ensemble')
            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)   

            base_name = f"{file_name.split('.')[0]}_{ensemble_num}.pt"

            # add file name to path
            filename = os.path.join(new_dir_path, base_name)
            
        torch.save(model.state_dict(), filename)


