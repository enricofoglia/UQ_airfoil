import argparse

import random

import numpy as np

import torch

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

        self.args = self.parser.parse_args()
        if print: self.message()

    def message(self):
        print( '+----------------------------+')
        print( '| Parsed args   | value      |')
        print( '+---------------+------------+')
        print(f'| Epochs        | {self.args.epochs:>10d} |')
        print(f'| Samples       | {self.args.samples:>10d} |')
        print(f'| Hidden units  | {self.args.hidden:>10d} |')
        print(f'| Fourier modes | {self.args.fourier:>10d} |')
        print(f'| Batch size    | {self.args.batch:>10d} |')
        print( '+----------------------------+')