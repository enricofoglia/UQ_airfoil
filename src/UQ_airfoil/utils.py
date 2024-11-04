import argparse

import random

import numpy as np

import torch

from model import ZigZag, Ensemble, MCDropout


def count_parameters(model):
    if model.kind == 'ensemble':
         return sum(p.numel() for p in model[0].parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def set_seed(seed_value:int) -> None:
    """ Set seed for reproducibility. """
    random.seed(seed_value)       # Python random module.
    np.random.seed(seed_value)    # Numpy module.
    torch.manual_seed(seed_value)

class Parser:
    def __init__(self, print=True) -> None:
        # parse inputs for batch training
        self.parser = argparse.ArgumentParser(description='Train graph networks on the airfrans dataset')
        self.parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
        self.parser.add_argument('--samples', '-s', type=int, default=800, help='number of samples in train dataset')
        self.parser.add_argument('--hidden', '-i', type=int, default=64, help='number of hidden units')
        self.parser.add_argument('--fourier', '-f', type=int, default=25, help='number of Fourier modes')
        self.parser.add_argument('--batch', '-b', type=int, default=16, help='batch size')
        self.parser.add_argument('--model_type', type=str, default='zigzag', help='which type of model to train', choices=['zigzag', 'latent_zigzag', 'ensembles', 'dropout'])
        self.parser.add_argument('--ens_size', '-n', type=int, default=5, help='number of particles in ensemble')
        self.parser.add_argument('--z0', type=float, help='uninformative value for zigzag', default=-1.0)
        self.parser.add_argument('--drop_prob', '-p', type=float, default=0.1, help='dropout probability')

        self.args = self.parser.parse_args()
        if print: self.message()

    def message(self):
        print( '+----------------------------+')
        print( '| Parsed args   | value      |')
        print( '+---------------+------------+')
        print(f'| Model type    | {self.args.model_type:>10}')
        print( '+---------------+------------+')
        print(f'| Epochs        | {self.args.epochs:>10d} |')
        print(f'| Samples       | {self.args.samples:>10d} |')
        print(f'| Hidden units  | {self.args.hidden:>10d} |')
        print(f'| Fourier modes | {self.args.fourier:>10d} |')
        print(f'| Batch size    | {self.args.batch:>10d} |')
        if self.args.model_type == 'ensemble':
            print(f'| Ensemble size | {self.args.ens_size:>10d} |')
        elif self.args.model_type == 'dropout':
            print(f'| Dropout prob. | {self.args.drop_prob:>10.2f} |')
        elif 'zigzag' in self.args.model_type.split('_'):
            print(f'| z0            | {self.args.z0:>10.2f} |')
            if 'latent' in self.args.model_type.split('_'):
                print(f'| Latent        | {'True':>10} |')
            else:
                print(f'| Latent        | {'False':>10} |')
        print( '+----------------------------+')

class ModelFactory:

    @staticmethod
    def create(self, args:dict):
        models = {
            'ensemble': Ensemble,
            'zigzag': ZigZag,
            'latent_zigzag': ZigZag,
            'dropout': MCDropout
        }
        
        model_class = models.get(args.model_type)
        if model_class is None:
            raise ValueError(f"Model type '{args.model_type}' not found. Available types: {list(models.keys())}")
        
        return model_class(**self._model_parameters(args))
    
    @property
    def _model_parameters(self, args:dict=None)->dict:
        if args == None:
            return self.model_dict
        
        # fixed parameters
        self.model_dict = {'edge_features': 3,
                           'n_blocks': 6,
                           'out_nodes': 1,
                           'out_glob': 0}
        # common parsed parameters
        self.model_dict['node_features'] = args.fourier+5
        self.model_dict['hidden_features'] =  args.hidden

        # type-specific parsed arguments
        if args.model_type == 'ensembles':
            self.model_dict['n_models'] = args.ens_size
        elif args.model_type == 'dropout':
            self.model_dict['p'] = args.drop_prob
        elif 'zigzag' in args.model_type.split('_'):
            self.model_dict['z0'] = args.z0
            if 'latent' in args.model_type.split('_'):
                self.model_dict['latent'] = True
            else:
                self.model_dict['latent'] = False 

        return self.model_dict
    
   