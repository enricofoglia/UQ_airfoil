'''
The base model for predictions is the :py:class:`EncodeProcessDecode` graph convolutional network as defined by Battaglia (2018).

Uncertainty Quantification
^^^^^^^^^^^^^^^^^^^^^^^^^^

The estimation of the epistemic uncertainty (EU), i.e. the level of confidence of the model in its own prediction, has been implemented in three different ways:

* :py:class:`Ensemble` : trains :math:`n` models. The EU is taken to be the variance between the predictions of the single models.
* :py:class:`MCDropout` : train one model with dropout layers. At inference time, perform :math:`T` forward passes *with the dropout layers active* and use the variance of the predictions as the EU.
* :py:class:`ZigZag` : train one model to recognize correct input-output pairs by feeding it back its own predictions. Use the squared difference of the two predictions as the EU. Comes in two flavors: the classic, where the feedback is done with the output, and the "latent" where the feedback is done with the hidden features of the last layer. 
'''

import copy

import torch
from torch import nn
from torch import Tensor

import torch_scatter

import torch_geometric as pyg
from torch_geometric.nn.aggr import Aggregation, SoftmaxAggregation
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv

from typing import (
    List, 
    Tuple,
    Union, 
    Optional,
    Callable,
    Dict,
    Any
    )

class MiniMLP(nn.Module):
    r'''Simple multilayer perceptron. 

    Args:
        inputs (int): dimensionality of the input
        targets (int): dimensionality of the output
        hidden (List[int]): dimensionaly of the hidden layers
    '''
    def __init__(self, inputs: int, targets: int, hidden: List[int]) -> None:
        
        super().__init__()

        layers = []
        
        # Add the input layer
        prev_dim = inputs
        for h_dim in hidden:
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        
        # Add the output layer
        layers.append(nn.Linear(prev_dim, targets))
        
        # Convert the list of layers into a ModuleList
        self.layers = nn.ModuleList(layers)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, inputs)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, targets)
        '''
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
class DropoutMLP(MiniMLP):
    r'''Simple multilayer perceptron with dropout layers. 

    Args:
        inputs (int): dimensionality of the input
        targets (int): dimensionality of the output
        hidden (List[int]): dimensionaly of the hidden layers
        p (float): probability of dropout
    '''
    def __init__(self, inputs: int, targets: int, hidden: List[int],p:float) -> None:
        
        super().__init__(inputs, targets, hidden)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, inputs)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, targets)
        '''
        for layer in self.layers[:-1]:
            x = self.dropout(self.activation(layer(x)))
        return self.layers[-1](x)
    
    
class GNNBlock(pyg.nn.conv.MessagePassing):
    r'''A graph-convolutional block "à la Google". Start by updating edges
    using a MLP (with a single hidden layer), then propagate the values to 
    the neighborhood of every node using the aggregation function. Finally,
    update the node features using another MLP. For the complete documentation
    of the inputs, see :obj:`MessagePassing` class.

    Args:
        node_features (int): dimensionality of the node features
        edge_features (int): dimensionality of the edge features
        aggr (str, List[str] or Aggregation, optional): aggregation function.
            See documentation of :obj:`MessagePassing` class (default :obj:`"sum"`)
        dropout (bool, optional): whether to include dropout layers in the 
            mini-MLPs (default :obj:`False`)
        p (float, optional): probability of dropout (default :obj:`0.1`)
    '''
    def __init__(
            self,
            node_features:int,
            edge_features:int,
            aggr: Optional[Union[str, List[str], Aggregation]] = 'sum',
            dropout:Optional[bool]=False,
            p:Optional[float]=0.1,
            *,
            aggr_kwargs: Optional[Dict[str, Any]] = None,
            flow: str = "source_to_target",
            node_dim: int = -2,
            decomposed_layers: int = 1):
        
        super().__init__(aggr=aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim,decomposed_layers=decomposed_layers)

        self.node_features = node_features
        if dropout:
            self.udpate_edge = DropoutMLP(edge_features+2*node_features, edge_features, [edge_features],p)
            self.udpate_node = DropoutMLP(edge_features+node_features, node_features, [node_features],p)
        else:
            self.udpate_edge = MiniMLP(edge_features+2*node_features, edge_features, [edge_features])
            self.udpate_node = MiniMLP(edge_features+node_features, node_features, [node_features])

    def message(self, x_i, x_j, edge_attr):
        x = torch.cat((x_i, x_j, edge_attr), dim=-1)
        x = self.udpate_edge(x)
        return x
    
    def aggregate(self, inputs, index,dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim,dim_size=dim_size, reduce='mean')
        return (inputs, out)
    
    def forward(self, x, edge_index, edge_attr)->Tuple[Tensor, Tensor]:
        edge_out, aggr = self.propagate(edge_index, x=(x,x), edge_attr = edge_attr)
        node_out = self.udpate_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_out + edge_attr
        node_out = node_out + x 
        return node_out, edge_out


class EncodeProcessDecode(nn.Module):
    r'''Complete convolutional graph neural network "à la Google". The data
    flow is composed of three stages:
    1. Encode the features using an MLP
    2. Process the latent features using various :obj:`GNNBlock` in series
    3. Global aggregate and process global features
    4. Decode node features using an MLP

    Args:
        node_features (int): dimensionality of the input node features
        edge_features (int): dimensionality of the input edge features
        hidden_features (int): dimensionality of the hidden features
        n_blocks (int): number of :obj:`GNNBlock` in the processor
        out_nodes (int): dimensionality of the output node features
        out_glob (int): dimensionality of the output global features
        dropout (bool, optional): whether to include dropout layers in the 
            mini-MLPs (default :obj:`False`)
        p (float, optional): probability of dropout (default :obj:`0.1`)

    TODO: maybe separate in two classes for dropout and not dropout and then use a super class.
    '''
    def __init__(
            self,
            node_features:int,
            edge_features:int,
            hidden_features:int,
            n_blocks:int,
            out_nodes:int,
            out_glob:int,
            dropout:Optional[bool]=False,
            p:Optional[float]=0.1
           )->None:
        
        super().__init__()

        self.kind = 'simple_gnn'

        if dropout:
            self.encoder_nodes = DropoutMLP(node_features, hidden_features, [hidden_features],p)
            self.encoder_edges = DropoutMLP(edge_features, hidden_features, [hidden_features],p)
            blocks = [GNNBlock(hidden_features, hidden_features,dropout=dropout,p=p) for _ in range(n_blocks)]
            self.processor = nn.ModuleList(blocks)
            self.decoder_nodes = DropoutMLP(hidden_features, out_nodes, [hidden_features],p)
            if out_glob > 0:
                self.node2glob = SoftmaxAggregation(learn=True)
                self.decoder_glob = DropoutMLP(hidden_features, out_glob, [hidden_features, hidden_features],p)

        else:
            self.encoder_nodes = MiniMLP(node_features, hidden_features, [hidden_features])
            self.encoder_edges = MiniMLP(edge_features, hidden_features, [hidden_features])
            blocks = [GNNBlock(hidden_features, hidden_features,dropout=dropout,p=p) for _ in range(n_blocks)]
            self.processor = nn.ModuleList(blocks)
            self.decoder_nodes = MiniMLP(hidden_features, out_nodes, [hidden_features])
            if out_glob > 0:
                self.node2glob = SoftmaxAggregation(learn=True)
                self.decoder_glob = MiniMLP(hidden_features, out_glob, [hidden_features, hidden_features])


    def forward(self, data:Data, return_hidden:bool=False)->Tuple[Tensor, Tensor]:
        '''Forward pass through the network.

        Args:
            data (Data): Input graph
            return_hidden (bool, optional): if :obj:`True`, returns also the activations of the last layer (default :obj:`False`)

        Returns:
            Tuple[Tensor, Tensor]: Output tensors for node features, global features and activations of the last layer (if :obj:`return_hidden=True`)
        '''
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # encode node and edge features
        node_feature = self.encoder_nodes(x)

        edge_feature = self.encoder_edges(edge_attr)

        # processing
        for block in self.processor:
            node_feature, edge_feature = block(node_feature, edge_index, edge_feature)

        # decode node features
        y = self.decoder_nodes(node_feature)

        # global aggregation
        if hasattr(self, 'node2glob'):
            batch = data.batch
            glob_in = self.node2glob(node_feature, batch,dim=-2)
            glob_out = self.decoder_glob(glob_in)

            if return_hidden: return y, glob_out, node_feature
            return y, glob_out
        else:
            if return_hidden: return y, node_feature
            return y
    
class ZigZag(EncodeProcessDecode):
    r'''Implementation of ZigZag to estimate the epistemic uncertainty of a graph network. 
    ZigZag is based in the intuition that a neural network can be trained
    to recognize its own outputs, so that if it commits a mistake it should
    be able to detect it. To see a complete list of inputs, see :obj:`EncodeProcessDecode`.

    Args:
        z0 (float, optional): constant for first pass in the network (default :obj:`0.0`)
        latent (bool, optional): if :obj:`True`, loop the last latent feature instead of 
            the output (default :obj:`False`)
    '''
    def __init__(self,
                node_features: int,
                edge_features: int,
                hidden_features: int,
                n_blocks: int,
                out_nodes: int,
                out_glob: int,
                z0:Optional[float]=0.0,
                latent:Optional[bool]=False) -> None:
        
        if latent:
            super().__init__(node_features+hidden_features, edge_features, 
                             hidden_features, n_blocks, out_nodes, out_glob)
            self.kind = 'latent_zigzag'
            self.dim_reentrant = hidden_features
        else:
            super().__init__(node_features+out_nodes, edge_features, 
                             hidden_features, n_blocks, out_nodes, out_glob)
            self.kind = 'zigzag'
            self.dim_reentrant = out_nodes
        
        self.z0 = z0

    def forward(self, data:Data, y:Optional[Tensor]=None,
                 return_var:bool=False, return_hidden:bool=False)->Tuple[Tensor, Tensor]:
        r'''
        Performs a forward pass in the network.

        Args:
            data (Data): input graph
            y (Tensor, optional): feedback loop. If :obj:`None`, uses :obj:`self.z0` (default :obj:`None`)
            return_var (bool, optional): if :obj:`True`, return prediction and variance by calling the model twice.
            return_hidden (bool, optional): if :obj:`True`, returns also the activations of the last layer (default :obj:`False`)
        '''
        if return_var: return self._call_recursively(data)
        datain = copy.deepcopy(data)
        if y is None:
            batch = data.x.shape[0]
            y = self.z0*torch.ones((batch, self.dim_reentrant),device=data.x.device)

        datain.x = torch.cat([data.x, y], dim=1)
        return super().forward(datain, return_hidden=return_hidden)
    
    def _call_recursively(self, data:Data):
        r'''Call ZigZag twice to return the epistemic uncertainty on the 
        node and global features.
        '''
        if hasattr(self, 'node2glob'):
            if self.kind == 'latent_zigzag':
                y1, y_glob1, h = self.forward(data, return_hidden=True)
                y2, y_glob2 = self.forward(data, y=h.detach())
            else:
                y1, y_glob1 = self.forward(data)
                y2, y_glob2 = self.forward(data, y=y1.detach())
            return 0.5*(y1+y2), 0.5*(y_glob1+y_glob2), 0.5*(y1-y2)**2,  0.5*(y_glob1-y_glob2)**2
        else:
            if self.kind == 'latent_zigzag':
                y1, h = self.forward(data, return_hidden=True)
                y2 = self.forward(data, y=h.detach())
            else:
                y1 = self.forward(data)
                y2 = self.forward(data, y=y1.detach())
            return 0.5*(y1+y2), 0.5*(y1-y2)**2


class Ensemble(nn.Module):
    r'''Ensemble class. See it as a list of models, to be trained independently.
    The base model is the :obj:`EncodeProcessDecode` GNN, see its documentation 
    for a complete list of inputs.

    Args:
        n_models (int): number of particles in the ensemble.
    '''
    def __init__(self,
                 n_models:int,
                node_features:int,
                edge_features:int,
                hidden_features:int,
                n_blocks:int,
                out_nodes:int,
                out_glob:int,
            )->None:
        
        super().__init__()

        self.kind = 'ensemble'
        self.models_list = [EncodeProcessDecode(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            n_blocks=n_blocks,
            out_nodes=out_nodes,
            out_glob=out_glob
        ) for _ in range(n_models)]

    def forward(self, data:Data, return_var:bool=False)->Tuple[Tensor, Tensor]:
        r'''
        Performs a forward pass in the network.

        Args:
            data (Data): input graph
            return_var (bool, optional): if :obj:`True`, return prediction and variance by computing the variance of the predictions of the single models.
        '''
        y_list = []
        glob_list = []
        for model in self.models_list:
            y, glob = model(data)
            y_list.append(y)
            glob_list.append(glob)
        y_mean = torch.stack(y_list,dim=-1).mean(dim=-1)
        glob_mean = torch.stack(glob_list,dim=-1).mean(dim=-1)
    
        if return_var:
            return y_mean, glob_mean, torch.stack(y_list,dim=-1).var(dim=-1), torch.stack(glob_list,dim=-1).var(dim=-1)
        else:
            return y_mean, glob_mean
        
    def __len__(self):
        return len(self.models_list)
    
    def __iter__(self):
        # Use a generator function to yield elements
        for model in self.models_list:
            yield model

    def __getitem__(self, idx):
        return self.models_list[idx]
    
class MCDropout(EncodeProcessDecode):
    r'''Class for graph-net MC Dropout method for epistemic uncertainty evaluation.
    Based on the :obj:`EncodeProcessDecode` class, see its documentation for 
    complete list of inputs.
    '''
    def __init__(self, node_features: int,
                 edge_features: int,
                 hidden_features: int,
                 n_blocks: int,
                 out_nodes: int,
                 out_glob: int,
                 dropout: bool | None = True,
                 p:Optional[float]=0.1):
        
        
        super().__init__(node_features,
                         edge_features,
                         hidden_features,
                         n_blocks,
                         out_nodes, 
                         out_glob,
                         dropout, 
                         p)
        self.kind= 'dropout'
        
    def forward(self,  data:Data, T:Optional[int]=1, return_var:bool=False):
        r'''Performs a forward pass in the network.

        Args:
            data (Data): input graph
            T (int, optional): how many forward passes are used in the network (default :obj:`1`)
            return_var (bool, optional): if :obj:`True`, return prediction and variance by calling the model twice.
        '''
        if T == 1:
            return super().forward(data)
        
        self.train()

        y_list = []
        glob_list = []
        for _ in range(T):
            y, glob = super().forward(data)
            y_list.append(y)
            glob_list.append(glob)
        y_mean = torch.stack(y_list,dim=-1).mean(dim=-1)
        glob_mean = torch.stack(glob_list,dim=-1).mean(dim=-1)
    
        if return_var:
            return y_mean, glob_mean, torch.stack(y_list,dim=-1).var(dim=-1), torch.stack(glob_list,dim=-1).var(dim=-1)
        else:
            return y_mean, glob_mean


if __name__ == '__main__':
    from dataset import XFoilDataset

    root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/airfoils/train_shapes'
    dataset = XFoilDataset(root)
    graph = dataset[0]

    block = GNNBlock(edge_features=3, node_features=3)
    # print(torch_scatter.scatter(graph.edge_attr, graph.edge_index, dim=-2, reduce='sum'))
    print(block(graph.x, graph.edge_index, graph.edge_attr))

    model = EncodeProcessDecode(
            node_features=3,
            edge_features=3,
            hidden_edge_features=32,
            hidden_node_features=32,
            n_blocks=4,
            out_nodes=1,
            )
    
    print(model)
    
    for _ in range(10):
        print(model(graph))