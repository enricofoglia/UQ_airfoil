'''
Train single neural networks using :py:class:`Trainer` and ensembles using :py:class:`EnsembleTrainer`. 

.. warning:: To use these classes on new models, you need to include a new :obj:`kind` in :obj:`_train_epoch` and :obj:`_test_epoch`.

To do: 
^^^^^^
* Differentiate loss function for node and global features
* Make the requirement for :obj:`kind` softer: throws a warning but tries something anyway
'''

import os
import copy

from pathlib import Path

from typing import (
    Optional,
    Callable,
    Dict,
    Any,
    Union
)

import torch 
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import (
    LRScheduler,
    ExponentialLR,
    ReduceLROnPlateau
    )
 
from torch.nn import MSELoss, Module
from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import _LRScheduler

from torch_geometric.loader import DataLoader

from tqdm import tqdm

 
class Trainer():
    r'''Class for training single neural networks.

    Args:
        epochs (int): numper of training epochs
        model (torch.nn.Module): model to be trained. To distringuish different 
            architectures, :obj:`model` needs to have the attribute :obj:`kind`.
        optimizer (torch.optim.Optimizer, optional): an instance of an optimizer
            yet to be initialized (default :obj:`Adam`).
        loss_fn (Callable, optional): loss function :math:`\ell(\widehat{y},y)`
            (default :obj:`MSELoss()`)
        device (str or torch.device, optional): type of device (default :obj:`"cpu"`)
        scheduler (torch.optim.lr_scheduler.LRScheduler or None, optional):
            learning rate scheduler, to be initialized (default :obj:`None`)
        optim_kwargs (dict or None, optional): keyword argument to initialize
            the optimizer (default, :obj:`None`)
        scheduler_kwargs (dict or None, optional): keyword argument to initialize
            the scheduler (default, :obj:`None`)
        weight (float, optional): ratio between the loss of the global and
            the node level loss (default :obj:`0.01`)
        mcmc (bool, optional): save weights during optimization (default :obj:`False`)
        save_rate (int, optional): save weigths every :obj:`save_rate` epochs (default :obj:`1`)
        save_start (int, optional): start saving from :obj:`save_start` epoch (defcault :obj:`0`)
    '''
    def __init__(
            self,
            epochs:int,
            model:Module,
            optimizer:Optional[Optimizer]=Adam,
            loss_fn:Optional[Callable[..., Any]]=MSELoss(),
            device:Optional[Union[torch.device, str]]='cpu',
            scheduler:Optional[LRScheduler]=None,
            optim_kwargs:Optional[Dict[str,Any]]=None,
            scheduler_kwargs:Optional[Dict[str,Any]]=None,
            weight:Optional[float]=0.01,
            mcmc:Optional[bool]=False,
            save_rate:Optional[int]=1,
            save_start:Optional[int]=0
            ) -> None:
        
        self.epochs = epochs
        self.model = model
        if optim_kwargs is not None:
            self.optimizer = optimizer(self.model.parameters(), **optim_kwargs)
        else: 
            self.optimizer = optimizer(self.model.parameters())
        self.loss_fn = loss_fn

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if scheduler is not None:
            if scheduler_kwargs is not None:
                self.scheduler = scheduler(self.optimizer, **scheduler_kwargs)
            else:
                self.scheduler = scheduler(self.optimizer)
        else:
            self.scheduler = None

        self.weight = weight
        self.mcmc = mcmc 
        self.save_start = save_start
        self.save_rate = save_rate

    def fit(self, train_loader:DataLoader, test_loader:DataLoader, 
            savefile:Optional[str]='out/best_model.pt')-> None:
        r'''Optimize the model. Save the best model in terms of test 
        performances in :obj:`"savefile"`.
        '''
        print( '+----------------------------------+')
        print( '| Training started ...             |')
        print( '+----------------------------------+')
        print(f'| Total number of epochs : {self.epochs:<8d}|')
        print( '+----------------------------------+')

        self.training_history = []
        self.test_history = []
        self.best_loss = torch.inf
        self.lr_history = []

        save_path = Path(savefile)
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.training_history.append(self._train_epoch(train_loader, self.model))
            self.test_history.append(self._test_epoch(test_loader, self.model))
            # print(f' Current loss = {self.training_history[-1]}')

            if self.scheduler is not None:
                self.scheduler.step()

            if self.test_history[-1] < self.best_loss:
                self.best_loss = self.test_history[-1]
                torch.save(self.model.state_dict(), save_path)

            if self.mcmc:
                if epoch >= self.save_start and epoch%self.save_rate == 0:
                    sgld_path = save_path.with_name(save_path.stem 
                                                    + f'SGLD_{(epoch-self.save_start)//self.save_rate}'
                                                    + save_path.suffix)
                    torch.save(self.model.state_dict(), sgld_path)
                    self.lr_history.append(self.optimizer.param_groups[0]['lr'])

                    
        if self.mcmc:
            torch.save(torch.tensor(self.lr_history), 'learning_rate_SGLD.pt')

        print( '| Training ended                   |')
        print( '+----------------------------------+')
        print(f'| Final Loss : {self.training_history[-1]:<19.3f} |')
        print(f'| Best Loss  : {self.best_loss:<19.3f} |')
        print( '+----------------------------------+')

    def _train_epoch(self, loader, model):
        batch_count = 0
        current_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            # reset gradients
            self.optimizer.zero_grad()

            if model.kind == 'simple_gnn' or model.kind == 'dropout':
                loss = self._loss_feedforward(model, batch)

            elif model.kind == 'zigzag' or model.kind == 'latent_zigzag':
                loss = self._loss_zigzag(model, batch)
            else:
                raise ValueError(f'Unrecognized kind of model "{model.kind}"')
            # backpropagate and step
            loss.backward()
            self.optimizer.step()

            batch_count += 1
            current_loss += loss.item()

        return current_loss/batch_count         

    def _loss_zigzag(self, model, batch):
        y = batch.y
        if hasattr(batch, 'y_glob'):
            y_glob = batch.y_glob
            if 'latent' not in model.kind.split('_'):
                pred1, pred_glob1 = model(batch)
                feedback = y.unsqueeze(1)
            else:
                pred1, pred_glob1, feedback = model(batch, return_hidden=True)
            pred2, pred_glob2 = model(batch, feedback.detach())
            loss = 0.5*(self.loss_fn(pred1.squeeze(), y.squeeze()) 
                        + self.weight*self.loss_fn(pred_glob1.squeeze(), y_glob.squeeze())
                        + self.loss_fn(pred2.squeeze(), y.squeeze()) 
                        + self.weight*self.loss_fn(pred_glob2.squeeze(), y_glob.squeeze()))
        else:
            if 'latent' not in model.kind.split('_'):
                pred1 = model(batch)
                feedback = y.unsqueeze(1)
            else:
                pred1, feedback = model(batch, return_hidden=True)
            pred2 = model(batch, feedback.detach())
            loss = 0.5*(self.loss_fn(pred1.squeeze(), y.squeeze()) 
                        + self.loss_fn(pred2.squeeze(), y.squeeze()))
            
        return loss


    def _loss_feedforward(self, model, batch):
        y = batch.y
        if hasattr(batch, 'y_glob'):
            y_glob = batch.y_glob
            pred, pred_glob = model(batch)
            loss = (self.loss_fn(pred.squeeze(), y.squeeze()) 
                        + self.weight*self.loss_fn(pred_glob.squeeze(), y_glob.squeeze()))
        else:
            pred = model(batch)
            loss = (self.loss_fn(pred.squeeze(), y.squeeze()))
                       
        return loss
    
    def _test_epoch(self, loader, model):
        batch_count = 0
        current_loss = 0
        model.eval()
        for batch in loader:
            batch = batch.to(self.device)
            loss = self._loss_feedforward(model, batch)

            batch_count += 1
            current_loss += loss.item()

        return current_loss/batch_count
    
    def get_train_history(self):
        r'''Get the training history.'''
        try: return self.training_history
        except AttributeError:
            print('The model has not been trained yet.')
    
    def get_test_history(self):
        r'''Get the test history.'''
        try: return self.test_history
        except AttributeError:
            print('The model has not been trained yet.')
    
class EnsembleTrainer(Trainer):
    r'''Class to train Ensemble models. Since it is based on the :obj:`Trainer` class,
    the constituents of the ensemble can be any model supported by the base trainer.

    Args:
        epochs (int): numper of training epochs
        ensemble (Module): ensemble to be trained. 
        optimizer (str, optional): name of the optimizer to be used. Supported 
            optimizers: :obj:`"adam"`.
        loss_fn (Callable, optional): loss function :math:`\ell(\widehat{y},y)`
            (default :obj:`MSELoss()`)
        device (str, optional); type of device (default :obj:`"cpu"`)
        scheduler (LRScheduler or None, optional):
            learning rate scheduler, to be initialized (default :obj:`None`)
        optim_kwargs (dict or None, optional): keyword argument to initialize
            the optimizer (default, :obj:`None`)
        scheduler_kwargs (dict or None, optional): keyword argument to initialize
            the scheduler (default, :obj:`None`)
        weight (float, optional): ratio between the loss of the global and
            the node level loss (default :obj:`0.01`)

    '''
    def __init__(self,
                 epochs: int,
                 ensemble: Module,
                 optimizer: Optional[str] = 'adam',
                 loss_fn: Optional[Callable[..., Any]] = MSELoss(),
                 device: Optional[Union[torch.device,str]] = 'cpu',
                 scheduler: Optional[LRScheduler] = None,
                 optim_kwargs: Optional[Dict[str, Any]] = None,
                 scheduler_kwargs: Optional[Dict[str, Any]] = None,
                 weight: Optional[float] = 0.01) -> None:
        
        self.epochs = epochs
        self.ensemble = ensemble
        
        self._optimizer = optimizer
        self.optim_kwargs = optim_kwargs
        self._scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        self.loss_fn = loss_fn
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.weight = weight

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


    def fit(self, train_loader:DataLoader, test_loader:DataLoader,
             savefile:Optional[str]='out/best_model.pt')->None:
        r'''Optimize the model. Save the best model in terms of test 
        performances in :obj:`"savefile"`, adding an 'ensemble' directory.
        '''
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
                    self._save_checkpoint(model, savefile, n)

            print(f'| Final Loss : {training_history[-1]:<19.3f} |')
            print(f'| Best Loss  : {self.best_loss:<19.3f} |')
            print( '+----------------------------------+')

            self.training_history.append(training_history)
            self.test_history.append(test_history)
            self.lr_history.append(lr_history)

        print( '| Training ended                   |')
        print( '+----------------------------------+')
        

    @property
    def n_models(self)->int:
        return len(self.ensemble)
    
    def _save_checkpoint(self, model: Module, filename:str='out/best_model.pt', ensemble_num:int=-1)-> None:
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


class SGLD(Optimizer):
    """Implements SGLD algorithm based on
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

    Built on the PyTorch SGD implementation
    (https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/sgd.py)

    copied from:
    https://github.com/alisiahkoohi/Langevin-dynamics/blob/master/src/langevin_sampling/SGLD.py
    """

    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])
                noise_std = torch.tensor([2 * group['lr']])
                noise_std = noise_std.sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0,
                                                          std=1) * noise_std
                p.data.add_(noise)

        return 1.0
    
class PowerDecayLR(_LRScheduler):
    """
    Custom learning rate scheduler that implements the formula:
    lr = a * (b + epoch)^(-gamma)
    
    Args:
        optimizer: Wrapped optimizer
        a (float): Scale factor
        b (float): Offset factor
        gamma (float): Power decay factor
        last_epoch (int): The index of last epoch. Default: -1
    """
    def __init__(self,
                 optimizer:Optimizer,
                 a:float,
                 b:float,
                 gamma:float,
                 last_epoch:Optional[int]=-1):
        self.a = a
        self.b = b
        self.gamma = gamma
        super(PowerDecayLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Calculate the new learning rate using the formula
        factor = self.a * pow(self.b + self.last_epoch, -self.gamma)
        # Apply the factor to all parameter groups
        return [factor for base_lr in self.base_lrs]