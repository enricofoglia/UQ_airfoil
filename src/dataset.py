'''This module includes all functions and classes to take care of the data preprocessing. The main class to be used is :py:class:`XFoilDataset`, which takes care of everything. 
'''

import os.path as osp
import os

from typing import (
    List, 
    Tuple,
    Union, 
    Optional,
    Callable,
    Any
    )

import json

from tqdm import tqdm

import torch 

import torch_geometric as pyg
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import Distance, BaseTransform

import numpy as np

import airfrans as af

class GeometricData(Data):
    r'''Class representing a 2D geometric graph. Includes :obj:`tangents` and 
    :obj:`curvature` methods, provided that the :obj:`TangentVec` transform 
    has been previously called on an object of this class. 
    '''
    def __init__(self, x: torch.Tensor | None = None, edge_index: torch.Tensor | None = None, edge_attr: torch.Tensor | None = None, y: torch.Tensor | int | float | None = None, pos: torch.Tensor | None = None, time: torch.Tensor | None = None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

    @property
    def curvature(self):
        '''Compute the curvature as the turning angle between two consecutive
        edges. Store it as a scalar vertex feature.'''
        # add curvature by arcos(dot(tangents))
        if hasattr(self, '_curvature'):
            return self._curvature
        else:
            angles = []
    
            # Iterate through each node
            # TODO: is there a way to do it without looping over every node?
            for node in range(self.num_nodes):
                # Find edges connected to the current node
                connected_edges = (self.edge_index[0] == node).nonzero(as_tuple=True)[0]
                
                if len(connected_edges) < 2:
                    continue
                
                for i in range(len(connected_edges)):
                    for j in range(i + 1, len(connected_edges)):
                        # Get tangents connected to the vertex
                        v1 = self.tangents[connected_edges[i],:]
                        v2 = self.tangents[connected_edges[j],:]
                        
                        # Compute the angle between the vectors
                        cos_theta = torch.dot(v1, -v2) /  (torch.norm(v1) * torch.norm(v2))
                        angle = torch.acos(cos_theta).item()
                        angles.append(angle)
                        if not np.isfinite(angle):
                            # torch.acos(1.0) = nan 
                            angles[-1] = 0.
            
            self._curvature = torch.tensor(angles, dtype=torch.float32)
            return self._curvature
        

    @property
    def tangents(self):
        '''The normalized tangent vectors to all edges of the graph.'''
        if not hasattr(self, '_tangents'):
            raise RuntimeError("You must call 'TangentVec' "
                               "on this class before accessing the tangents.")
        return self._tangents

class XFoilDataset(Dataset):
    '''
    This implementation of the class is made more complicated then necessary
    in order to be ready when I'll have to make a real dataset.
    Class for graphs representing 2D airfoils processed in XFloil. In particular
    the :obj:`y` attribute is the pressure distribution, and the :obj:`y_glob` one
    is the aerodynamic efficiency :math:`(c_L/c_D)`. The nodes are supposed to be
    saved in order starting from the trailing edge, in counter-clockwise order.
    If :obj:`normalize=True`, standardize the global feature (default :obj:`False`).

    Args:
        root (str, optional): root directoty (default :obj:`None`)
        normalize (bool, optional): if :obj:`True`, standardize global outputs (default :obj:`False`)
        transform (Callable, optional): function to call when retrieving a
            sample (default :obj:`None`)
        pre_transform (Callable, optional): function to call when initializing
            the class (default :obj:`None`)    
        pre_filter (Callable): how to discard samples at initialization (default :obj:`None`)
        log (bool, optional): whether to print to console while performing actions (default :obj:`True`)
        force_reload (bool, optional): whether to re-process the dataset. (default :obj:`False`)
    '''
    def __init__(
        self,
        root: Optional[str] = None,
        normalize:Optional[bool]=False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: Optional[bool] = True,
        force_reload: Optional[bool]=True
        ) -> None:
        
        # set up raw directory
        self.root = root
        self._raw_file_names = [fname for fname in os.listdir(self.raw_dir)]
        self._processed = False 
        self._normalized = False 
        self.normalize = normalize

        if normalize:
            self._compute_norm()

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

       
    @property
    def raw_dir(self) -> str:
        if osp.isdir(osp.join(self.root, 'raw')):
            return osp.join(self.root, 'raw')
        else:
            raise FileNotFoundError(f"Directory '{osp.join(self.root, 'raw')}' does not exist")
        
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return self._raw_file_names
    
    @property
    def processed_dir(self) -> str:
        os.makedirs(osp.join(self.root, 'processed'), exist_ok=True)
        return osp.join(self.root, 'processed')
    
    def process(self) -> None:
        '''
        Read raw data, construct graphs, add tangents and edge lengths as edge features 
        and curvature as vertex features. Save all processed data.
        '''
        if self._processed:
            return
        idx = 0
        for raw_path in tqdm(self.raw_paths):
            # Read data from `raw_path`.
            data = self._generate_sample(raw_path)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1
        self._processed = True

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        skip_names = ['pre_filter.pt', 'pre_transform.pt']
        if self._processed:
            return [fname for fname in os.listdir(self.processed_dir) 
                    if fname not in skip_names]
        else:
            return []

    def _generate_sample(self, filename) -> GeometricData:
        with open(filename, 'r') as file:
            sample = json.load(file)

        # 2D coords of the vertices
        pos = torch.tensor(sample["coords"], dtype=torch.float32)

        # edges indexes, edge with (index_1, index_2) connects vertices index_1 and index_2
        edge_index_forward = torch.tensor([[i, (i+1) % len(pos)] for i in range(len(pos))], dtype=torch.long).T
        edge_index_backwards = torch.tensor([[(i+1)% len(pos), i ] for i in range(len(pos))], dtype=torch.long).T
        edge_index = torch.cat([edge_index_forward, edge_index_backwards],dim=1)
        # global objective is lift to drag ratio
        y_glob = sample["CL"] / sample["CD"]
        if self.normalize:
            y_glob = (y_glob-self.avg)/self.std

        # pressure coefficent is the node level objective
        y = torch.tensor(sample['Cp'], dtype=torch.float32)    

        graph = GeometricData(x=pos, pos=pos, edge_index=edge_index, y=y, y_glob=y_glob)
        
        # add tangent vectors and length to edges
        graph = TangentVec()(graph)  
        graph = Distance()(graph)

        # add curvature to nodes' features
        graph.x = torch.cat([graph.x, graph.curvature.unsqueeze(1)], dim=1)
        
        return graph
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx:int)-> GeometricData:
        data = torch.load(self.processed_paths[idx])
        return data 
    
    def _compute_norm(self):
        '''Compute the average and standard deviation of the global feature
        to perform data normalization. Save them as attributes :obj:`avg`
        and :obj:`std`, respectively.'''

        y_list = []
        for raw_path in self.raw_paths:
            with open(raw_path, 'r') as f:
                sample = json.load(f)
            y_list.append(sample["CL"] / sample["CD"])

        self.avg = torch.mean(torch.tensor(y_list))
        self.std = torch.std(torch.tensor(y_list))

class AirfRANSDataset(Dataset):
    def __init__(self,
                 task: str,
                 train: bool | None = True,
                 root: str | None = None,
                 normalize: bool | None = True,
                 transform: Callable[..., Any] | None = None,
                 pre_transform: Callable[..., Any] | None = None,
                 pre_filter: Callable[..., Any] | None = None,
                 log: bool = True,
                 force_reload: bool = False) -> None:
        
        # set up raw directory
        self.root = root
        self._raw_file_names = [fname for fname in os.listdir(self.raw_dir)]
        if not force_reload and os.listdir(self.processed_dir):
            self._processed = True
        else: self._processed = False 
        self._normalized = False 
        self.normalize = normalize

        if normalize:
            self._compute_norm()

        self.task = task
        self.train = train
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

    @property
    def raw_dir(self) -> str:
        if osp.isdir(osp.join(self.root, 'raw')):
            return osp.join(self.root, 'raw')
        else:
            raise FileNotFoundError(f"Directory '{osp.join(self.root, 'raw')}' does not exist")
        
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return self._raw_file_names
    
    @property
    def processed_dir(self) -> str:
        os.makedirs(osp.join(self.root, 'processed'), exist_ok=True)
        return osp.join(self.root, 'processed')
    
    def process(self) -> None:
        '''
        Read raw data, construct graphs, add tangents and edge lengths as edge features 
        and curvature as vertex features. Save all processed data.
        '''
        if self._processed:
            return
        idx = 0
        raw_data, _ = af.dataset.load(self.raw_dir, task=self.task, train= self.train)
        for airfoil in tqdm(raw_data):
            # extract skin data
            ordered_data = self._order_points(self._extract_skin(airfoil))

            # Read data from `raw_path`.
            data = self._generate_sample(ordered_data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1
        self._processed = True

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        skip_names = ['pre_filter.pt', 'pre_transform.pt']
        if self._processed:
            return [fname for fname in os.listdir(self.processed_dir) 
                    if fname not in skip_names]
        else:
            return []

    def _generate_sample(self, skin_data) -> GeometricData:

        # 2D coords of the vertices
        pos = torch.tensor(skin_data[:,0:2], dtype=torch.float32)

        # edges indexes, edge with (index_1, index_2) connects vertices index_1 and index_2
        edge_index_forward = torch.tensor([[i, (i+1) % len(pos)] for i in range(len(pos))], dtype=torch.long).T
        edge_index_backwards = torch.tensor([[(i+1)% len(pos), i ] for i in range(len(pos))], dtype=torch.long).T
        edge_index = torch.cat([edge_index_forward, edge_index_backwards],dim=1)
       
        # pressure is the node level objective
        y = torch.tensor(skin_data[:,9], dtype=torch.float32)    

        graph = GeometricData(x=pos, pos=pos, edge_index=edge_index, y=y)
        
        # add tangent vectors and length to edges
        graph = TangentVec()(graph)  
        graph = Distance()(graph)

        # add curvature to nodes' features
        graph.x = torch.cat([graph.x, graph.curvature.unsqueeze(1)], dim=1)
        
        return graph
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx:int)-> GeometricData:
        data = torch.load(self.processed_paths[idx])
        return data 
    
    def _compute_norm(self):
        # TODO: implement normalization using Cp (sample wise) and dataset normalization (standardization)
        pass

    def _extract_skin(self, data):
        # extract airfoil skin
        skin_idx = np.nonzero(data[:,4]==0)
        return data[skin_idx]

    def _order_points(self, skin_data):
        '''
        Orders 2D airfoil points counter-clockwise, starting from the trailing edge.

        Identifies the trailing edge as the point with the largest x-coordinate, then orders 
        all points counter-clockwise based on their angles relative to the airfoil's centroid.
        '''

        trailing_edge_idx = np.argmax(skin_data[:, 0])

        centroid = np.mean(skin_data, axis=0)[0:2]

        angles = np.arctan2(skin_data[:, 1] - centroid[1], skin_data[:, 0] - centroid[0])

        sorted_indices = np.argsort(angles)
        sorted_skin_data = skin_data[sorted_indices]

        trailing_edge_sorted_idx = np.where(sorted_indices == trailing_edge_idx)[0][0]
        ordered_skin_data = np.roll(sorted_skin_data, -trailing_edge_sorted_idx, axis=0)

        return ordered_skin_data

class TangentVec(BaseTransform):
    def __init__(self, norm: bool = True, cat: bool = True) -> None:
        super().__init__()
        self.norm = norm
        self.cat = cat 

    def forward(self, data:Data) -> Data:
        assert data.pos is not None
        assert data.edge_index is not None
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[row] - pos[col]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if self.norm and cart.numel() > 0:
            length = torch.linalg.vector_norm(cart, dim=1)
            cart = cart/length.view(-1, 1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        # add tangent as attribute for other routines
        data._tangents = data.edge_attr[:,-2:]
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(norm={self.norm})'
    
class FourierEpicycles(BaseTransform):
    r'''Compute the Fourier transform of the airfoil seen as a function
    of a complex variable. Save the first :obj:`n` modes as node features
    (only the real part is sufficient). If :obj:`cat=False`, overwrite the
    node features already present. 
    '''
    def __init__(self, n:int, cat:Optional[bool]=True) -> None:
        
        super().__init__()
        self.n = n
        self.cat = cat 

    def forward(self, data: Data) -> Data:
        n_points = data.pos.shape[0]
        assert data.pos is not None 
        assert self.n < n_points

        x,y = data.pos.T 
        pseudo = data.x

        # make complex
        z = torch.complex(x,y)

        # compute FFT
        Z = torch.fft.fft(z) # points are not equispaced # TODO: implement NFFT (non-equispaced FFT)
        # retain only amplitude (real part)
        ampl = torch.sqrt(Z.real[:self.n]**2 + Z.imag[:self.n]**2)
        phase = torch.atan2(Z.imag[:self.n],Z.real[:self.n])
        
        # compute eignshapes
        theta = torch.linspace(0, 2*torch.pi, n_points)
        eigx = [ampl[m]/ampl[0]*torch.cos(m*theta + phase[m]) for m in range(self.n)]
        eigy = [ampl[m]/ampl[0]*torch.sin(m*theta + phase[m]) for m in range(self.n)]
        eigx, eigy = torch.stack(eigx,dim=-1), torch.stack(eigy,dim=-1)

        ampl_repl = ampl.repeat(n_points, 1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            # data.x = torch.cat([pseudo, ampl_repl.type_as(pseudo)], dim=-1)
            data.x = torch.cat([pseudo, eigx.type_as(pseudo)], dim=-1)
            # pseudo = data.x 
            # data.x = torch.cat([pseudo, eigy.type_as(pseudo)], dim=-1)
        else:
            data.x = eigx 
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n={self.n})'

                
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # =================================================
    # Matplotlib settings
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Sans Serif"],
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts}\usepackage[cm]{sfmath}",
        'axes.linewidth' : 2,
        'lines.linewidth' : 2,
        'axes.labelsize' : 16,
        'xtick.labelsize' : 14,
        'ytick.labelsize' : 14
    })
    # =================================================
    N = 100
    pre_transform = FourierEpicycles(n=N)

    # root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/airfoils/train_shapes'
    # dataset = XFoilDataset(root, pre_transform=pre_transform)
    root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/AirfRANS'
    dataset = AirfRANSDataset('scarce', True, root, normalize=False, pre_transform=pre_transform)
    index = 0
    for point in dataset:
        print(f'Checking point {index} ...')
        if torch.isfinite(point.x).all() and torch.isfinite(point.pos).all() and torch.isfinite(point.edge_attr).all():
            print('All good :)')
        else: 
            print(f'Check more closely point {index}')
        index += 1

    # print(dataset.processed_paths[0])
    # graph = dataset._generate_sample(dataset.raw_paths[0])
    # print(graph)
    graph = dataset[5]
    print(graph)
    print('x',torch.isfinite(graph.x).all())
    print('pos',torch.isfinite(graph.pos).all())
    print('edge_attr',torch.isfinite(graph.edge_attr).all())

    def plot_graph_curvature(graph):
        fig, ax = plt.subplots(layout='constrained')
        lim = torch.max(graph.curvature[graph.curvature != torch.max(graph.curvature)])
        sc = ax.scatter(graph.pos[:,0], graph.pos[:,1], c=graph.curvature, cmap='RdYlBu_r',
                        vmax=lim, edgecolors=None, zorder=2.5)
        num_edges = graph.edge_index.shape[1]
        for i in range(num_edges):
            start_idx = graph.edge_index[0, i].item()
            end_idx = graph.edge_index[1, i].item()
            start_coords = graph.pos[start_idx]
            end_coords = graph.pos[end_idx]
            ax.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], c='black', 
                     alpha=0.6)
        plt.colorbar(sc, ax=ax, label=r'turning angle $\varphi_i$')
        ax.set_xlabel(r'$x/c$ [-]')
        ax.set_ylabel(r'$y/c$ [-]')
        ax.axis('equal')

    def plot_graph_fourier(graph):
        fig, ax = plt.subplots(layout='constrained')
        ax.stem(graph.x[0,3:]/max(graph.x[0,3:]))
        ax.set_xlabel(r'index $n$')
        ax.set_ylabel(r'relative amplitude $\vert \hat{z}_i\vert/\vert \hat{z}_{max}\vert$')
        ax.set_title('Fourier transform')
        ax.set_yscale('log')
        
    def plot_graph_eigenshapes(graph,n):
        eigx = graph.x[:,3+n]
        fig, ax = plt.subplots(layout='constrained')
        scx = ax.scatter(graph.pos[:,0], graph.pos[:,1], c=eigx, cmap='RdYlBu_r', edgecolors=None, zorder=2.5)

        num_edges = graph.edge_index.shape[1]
        for i in range(num_edges):
            start_idx = graph.edge_index[0, i].item()
            end_idx = graph.edge_index[1, i].item()
            start_coords = graph.pos[start_idx]
            end_coords = graph.pos[end_idx]
            ax.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], c='black', alpha=0.6)
            
        plt.colorbar(scx, ax=ax, label=r'amplitude, x')
        ax.set_xlabel(r'$x/c$ [-]')
        ax.set_ylabel(r'$y/c$ [-]')
        ax.axis('equal')

        ax.set_title(r'Eigenshape $\phi_{{n}}(\vartheta)$, $n$={0}'.format(n))

    def plot_graph_subsample(graph, skip=3):
        fig, ax = plt.subplots(layout='constrained')
        v_us = graph.pos[::skip,:]
        ax.plot(v_us[:,0],v_us[:,1], 'ko-', mfc='w', ms=10)
        ax.set_xlabel(r'$x/c$ [-]')
        ax.set_ylabel(r'$y/c$ [-]')
        ax.axis('equal')
        ax.set_axis_off()
        

    plot_graph_subsample(graph, skip=5)
    plt.savefig('../out/figures/airfoil_graph.png', dpi=300, transparent=True)
    plot_graph_curvature(graph)
    # plt.show()

    plot_graph_fourier(graph)
    # plt.show()

    plot_graph_eigenshapes(graph, 0)
    plot_graph_eigenshapes(graph, 1)
    plot_graph_eigenshapes(graph, 5)
    plot_graph_eigenshapes(graph, 10)
    plt.show()




