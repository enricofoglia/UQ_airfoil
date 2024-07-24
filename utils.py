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
