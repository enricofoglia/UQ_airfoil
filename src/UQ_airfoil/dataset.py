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
from scipy.interpolate import CubicSpline

import airfrans as af

class GeometricData(Data):
    r'''Class representing a 2D geometric graph. Includes :obj:`tangents` and 
    :obj:`curvature` methods, provided that the :obj:`TangentVec` transform 
    has been previously called on an object of this class. 
    '''
    def __init__(
            self,
            x: Optional[torch.Tensor] = None,
            edge_index: Optional[torch.Tensor] = None,
            edge_attr: Optional[torch.Tensor] = None,
            y: Optional[Union[torch.Tensor, int, float]] = None,
            pos: Optional[torch.Tensor] = None,
            time: Optional[torch.Tensor] = None,
            **kwargs
            ):
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
        
        self._normalized = False 
        self.normalize = normalize
        if not force_reload and os.listdir(self.processed_dir):
            self._processed = True
        else: 
            self._processed = False 

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

        # add curvature to nodes' features
        graph = TangentVec()(graph)
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
    r'''Handling of the AirfRANS dataset (https://github.com/Extrality/AirfRANS.git). 
    Of the 2D flow field only the pressure on the surface is retained. The points of
    the skin are extracted, ordered and connected into a graph. Because of the 
    extreme inhomogeneity of the RANS mesh (e.g. refinement at the leading edge),
    it is advisable to use this dataset in conjuction with the UniformSampling 
    transformation.

    Args: 
        task (str): task to determine which version of the dataset to load. 
            Possible values are :obj:`'full'`, :obj:`'scarce'`, :obj:`'reynolds'` or :obj:`'aoa'`.
        train (bool, optional): wether to load the train or test split (default :obj:`True`)
        root (str, optional): root directoty (default :obj:`None`)
        normalize (bool or tuple, optional): if :obj:`True`, standardize global outputs (default :obj:`False`) and normalize pressure using :math:`1/2u^2` sample-wise. It is also possible to pass a tuple with precomputed mean and std, as :math:`((\mu_u,\mu_{\alpha}),(\sigma_u, \sigma_{\alpha}))`.
        transform (Callable, optional): function to call when retrieving a
            sample (default :obj:`None`)
        pre_transform (Callable, optional): function to call when initializing
            the class (default :obj:`None`)    
        pre_filter (Callable): how to discard samples at initialization (default :obj:`None`)
        log (bool, optional): whether to print to console while performing actions (default :obj:`True`)
        force_reload (bool, optional): whether to re-process the dataset. (default :obj:`False`)
    '''
    def __init__(self,
                 task: str,
                 train: Optional[bool] = True,
                 root: Optional[str] = None,
                 normalize: Optional[Union[bool,Tuple[Tuple]]] = True,
                 transform: Optional[Callable[..., Any]] = None,
                 pre_transform: Optional[Callable[..., Any]] = None,
                 pre_filter: Optional[Callable[..., Any]] = None,
                 log: Optional[bool] = True,
                 force_reload: Optional[bool] = False) -> None:
        
        # set up raw directory
        self.root = root
        self._skip_names = ['manifest.json']
        self._raw_file_names = [fname for fname in os.listdir(self.raw_dir) 
                                if fname not in self._skip_names]
        if not force_reload and os.listdir(self.processed_dir):
            self._processed = True
        else: self._processed = False 

        if isinstance(normalize, bool):
            self._normalized = not normalize 
            self.normalize = normalize
            self._glob_mean = (0.0, 0.0)
            self._glob_std  = (1.0,1.0)
        else:
            self._normalized = True 
            self._glob_mean = normalize[0]
            self._glob_std  = normalize[1]
            self.normalize = True

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
        train = 'train' if self.train else 'test'
        os.makedirs(osp.join(self.root, f'processed_{self.task}'), exist_ok=True)
        return osp.join(self.root, f'processed_{self.task}_{train}')
    
    @property
    def glob_mean(self):
        if self._normalized:
            return self._glob_mean
        
        u_mean, alpha_mean = 0.0, 0.0
        for name in self._raw_file_names:
            u, a = self._process_name(name) 
            u_mean += u
            alpha_mean += a 
        self._glob_mean = (u_mean / len(self._raw_file_names), 
                           alpha_mean / len(self._raw_file_names))
        return self._glob_mean
    
    @property
    def glob_std(self):
        if self._normalized:
            return self._glob_std
        
        u_mean, alpha_mean = self.glob_mean
        u_var, alpha_var = 0.0, 0.0
        for name in self._raw_file_names:
            u, a = self._process_name(name) 
            u_var += (u - u_mean)**2
            alpha_var += (a - alpha_mean)**2 
        self._glob_std =  (np.sqrt(u_var / (len(self._raw_file_names)-1)), 
                np.sqrt(alpha_var / (len(self._raw_file_names)-1)))
        self._normalized = True
        return self._glob_std

    def process(self) -> None:
        '''
        Read raw data, construct graphs, add tangents and edge lengths as edge features 
        and curvature as vertex features. Save all processed data.
        '''
        if self._processed:
            return
        idx = 0
        raw_data, names = af.dataset.load(self.raw_dir, task=self.task, train= self.train)
        for airfoil, name in tqdm(zip(raw_data, names), total=len(names)):
            # get global params
            u, alpha = self._process_name(name)
            if self.normalize:
                dyn_pressure = 0.5*u**2 # the pressure is already given divided by rho, chord = 1
                u = (u-self.glob_mean[0]) / self.glob_std[0]
                alpha = (alpha-self.glob_mean[1]) / self.glob_std[1]
            else: dyn_pressure = 1.0

            # extract skin data
            ordered_data = self._order_points(self._extract_skin(airfoil))

            # Read data from `raw_path`.
            data = self._generate_sample(ordered_data, dyn_pressure, u, alpha)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1
        self._processed = True

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        skip_names = ['pre_filter.pt', 'pre_transform.pt', 'manifest.json']
        if self._processed:
            return [fname for fname in os.listdir(self.processed_dir) 
                    if fname not in skip_names]
        else:
            return []
        
    def _process_name(self, string):
        '''Read name and return tuple (U, alpha) with velocity U in m/s and 
        angle of attack alpha in degrees'''
        splitted = string.split('_')
        return float(splitted[2]), float(splitted[3]) 

    def _generate_sample(self, skin_data, dyn_pressure, u, alpha) -> GeometricData:

        # 2D coords of the vertices
        pos = self._make_periodic(torch.tensor(skin_data[:,0:2], dtype=torch.float32))

        edge_index = self._construct_edges(pos)
       
        # pressure is the node level objective
        y = self._make_periodic(torch.tensor(skin_data[:,9], dtype=torch.float32)) / dyn_pressure 

        x = torch.stack((u*torch.ones_like(y),alpha*torch.ones_like(y),pos[:,0],pos[:,1]), dim=-1)
        graph = GeometricData(x=x, pos=pos, edge_index=edge_index, y=y)

        # add curvature to nodes' features
        graph = TangentVec()(graph)
        graph.x = torch.cat([graph.x, graph.curvature.unsqueeze(1)], dim=1)
        
        return graph
    
    def _construct_edges(self, pos):
        """Construct edge connectivity of an ordered point cloud."""
        edge_index_forward = torch.tensor([[i, (i+1) % len(pos)] for i in range(len(pos))], dtype=torch.long).T
        edge_index_backwards = torch.tensor([[(i+1)% len(pos), i ] for i in range(len(pos))], dtype=torch.long).T
        edge_index = torch.cat([edge_index_forward, edge_index_backwards],dim=1)
        return edge_index
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx:int)-> GeometricData:
        data = torch.load(self.processed_paths[idx])
        return data 

    def _extract_skin(self, data):
        # extract airfoil skin
        skin_idx = np.nonzero(data[:,4]==0)
        return data[skin_idx]
    
    def _order_points(self, points):
        '''Order the points by splitting in suction and pressure side using 
        the orientation of the normals.
        '''

        suction_side = points[points[:,6]<0]
        pressure_side = points[points[:,6]>=0]

    
        indices_suction = np.argsort(suction_side[:,0])[::-1]
        indices_pressure = np.argsort(pressure_side[:,0])

        return np.concatenate((suction_side[indices_suction],
                                pressure_side[indices_pressure]))




    def _make_periodic(self, x):
        try:
            return torch.concatenate((x, x[0].unsqueeze(0)))
        except RuntimeError:
            return torch.concatenate((x, x[0]))

class TangentVec(BaseTransform):
    r''' Add tangent vectors as edge features. If :obj:`norm=True`, normalize all lengths to 1.
    '''
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

    .. warning:: The FFT assumes that the input data is equispaced. If this is not the case, use in combination with :obj:`UniformSampling`.
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
        Z = torch.fft.fft(z) # make the hypothesis that points are equispaced
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
            data.x = torch.cat([pseudo, eigx.type_as(pseudo)], dim=-1)
            
        else:
            data.x = eigx 
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n={self.n})'
    

class UniformSampling(BaseTransform):
    r'''Resamples a curve uniformly into :obj:`n` points.
    '''
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n

    def forward(self, data: Data) -> Data:
        """Interpolate the curve uniformly."""
        assert data.pos is not None 

        x,y = data.pos.T 

        # make data periodic
        x = x
        y = y
        # Compute cumulative chord lengths for parameterization
        dx = torch.diff(x)
        dy = torch.diff(y)
        chord_lengths = torch.sqrt(dx**2 + dy**2)
        cum_chord = torch.concatenate((torch.tensor([0]), torch.cumsum(chord_lengths, dim=0)))
        
        # Normalize to [0, 1]
        t_param = cum_chord / cum_chord[-1]
        # print(t_param)
        
        # Create uniform sampling
        t_uniform = torch.linspace(0, 1, self.n+1)
        
        # Interpolate x and y separately
        x_spline = CubicSpline(t_param, x, bc_type='periodic')
        y_spline = CubicSpline(t_param, y, bc_type='periodic')

        # Interpolate remaining data
        feat_spline_list = []
        for feature in data.x.T:
            try:
                feat_spline_list.append(CubicSpline(t_param, feature, bc_type='periodic')) # TODO: check periodicity
            except ValueError:
                feat_spline_list.append(CubicSpline(t_param, feature, bc_type='not-a-knot'))

        try:
            out_spline = CubicSpline(t_param, data.y, bc_type='periodic')
        except ValueError:
            out_spline = CubicSpline(t_param, data.y, bc_type='not-a-knot')

        # rebuild the graph
        data = self._build_graph(t_uniform[:-1], x_spline, y_spline, 
                     feat_spline_list, out_spline, dtype=data.x.dtype)
        
        return data
    
    
    def _construct_edges(self, pos):
        """Construct edge connectivity of an ordered point cloud."""
        edge_index_forward = torch.tensor([[i, (i+1) % len(pos)] for i in range(len(pos))], dtype=torch.long).T
        edge_index_backwards = torch.tensor([[(i+1)% len(pos), i ] for i in range(len(pos))], dtype=torch.long).T
        edge_index = torch.cat([edge_index_forward, edge_index_backwards],dim=1)
        return edge_index
    
    def _build_graph(self, t_sample, x_spline, y_spline, 
                     feat_spline_list, out_spline, dtype):
        '''Construct new graph for resampled shape'''
        
        # sample splines
        x      = torch.tensor(x_spline(t_sample), dtype=dtype)
        y      = torch.tensor(y_spline(t_sample), dtype=dtype)
        output = torch.tensor(out_spline(t_sample), dtype=dtype)
        features = torch.zeros(len(t_sample), len(feat_spline_list), dtype=dtype)
        for col, spline in enumerate(feat_spline_list):
            features[:,col] = torch.tensor(spline(t_sample), dtype=dtype)

        # make all periodic
       
        # concatenate the coordinates
        pos = torch.stack((x, y), dim=1)

        # build edge connectivity
        edge_index = self._construct_edges(pos)

        return GeometricData(x=features, pos=pos, edge_index=edge_index, y=output)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n={self.n})'

                
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from torchvision import transforms

    from cmcrameri import cm

    CMAP = cm.managua_r

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
    N = 25
    n_points = 250
    pre_transform = transforms.Compose(( UniformSampling(n_points), FourierEpicycles(n=N),
                                        TangentVec(), Distance()))

    # pre_transform = FourierEpicycles(n=N)


    # root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/airfoils/train_shapes'
    # dataset = XFoilDataset(root, pre_transform=pre_transform, force_reload=True)
    root = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/AirfRANS'
    dataset = AirfRANSDataset('scarce', True, root, normalize=True, pre_transform=pre_transform, force_reload=True)
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
    graph = dataset[0]
    print(graph)
    print('x',torch.isfinite(graph.x).all())
    print('pos',torch.isfinite(graph.pos).all())
    print('edge_attr',torch.isfinite(graph.edge_attr).all())
    print('y',torch.isfinite(graph.y).all())

    # for edge, (feat, index )in enumerate(zip(graph.edge_attr,graph.edge_index.T)):
    #     print(f'Edge {edge}')
    #     print(f'Connectivity {index}')
    #     print(f'Feature {feat}')
    def plot_graph_pressure(graph, dataset, ax =None):
        u = graph.x[0,0]*dataset.glob_std[0]+dataset.glob_mean[0]
        alpha = graph.x[0,1]*dataset.glob_std[1]+dataset.glob_mean[1]

        p = graph.y
        if not dataset.normalize:
            p /= 0.5*u**2

        # fig, ax = plt.subplots(figsize=(6,3))
        if ax is None: 
            _, ax = plt.subplots()
        sc = ax.scatter(graph.pos[:,0], graph.pos[:,1], c=p, cmap=CMAP,
                        edgecolors=None, zorder=2.5)
        num_edges = graph.edge_index.shape[1]
        for i in range(num_edges):
            start_idx = graph.edge_index[0, i].item()
            end_idx = graph.edge_index[1, i].item()
            start_coords = graph.pos[start_idx]
            end_coords = graph.pos[end_idx]
            ax.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], c='black', 
                     alpha=0.6)
        plt.colorbar(sc, ax=ax, label=r'$C_p$ [-]')
        ax.set_xlabel(r'$x/c$ [-]')
        ax.set_ylabel(r'$y/c$ [-]')
        ax.axis('equal')
        
        ax.set_title(r'Pressure coefficient at $U$={0:.1f} m/s and $\alpha$={1:.2f}$^\circ$'.format(
            u,alpha) )

    def plot_graph_curvature(graph):
        fig, ax = plt.subplots(figsize=(6,3),layout='constrained')
        lim = torch.max(graph.curvature[graph.curvature != torch.max(graph.curvature)])
        sc = ax.scatter(graph.pos[:,0], graph.pos[:,1], c=graph.curvature, cmap=CMAP,
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
        max_amplitude = graph.x[0,5]
        for freq in range(graph.x.shape[1]-5):
            ax.stem(freq, max(graph.x[:,freq+5])/max_amplitude)
        ax.set_xlabel(r'index $n$')
        ax.set_ylabel(r'relative amplitude $\vert \hat{z}_i\vert/\vert \hat{z}_{max}\vert$')
        ax.set_title('Fourier transform')
        ax.set_yscale('log')
        
    def plot_graph_eigenshapes(graph,n):
        eigx = graph.x[:,5+n]
        fig, ax = plt.subplots(figsize=(6,3),layout='constrained')
        scx = ax.scatter(graph.pos[:,0], graph.pos[:,1], c=eigx, cmap=CMAP, edgecolors=None, zorder=2.5)

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
        fig, ax = plt.subplots(figsize=(6,3),layout='constrained')
        v_us = graph.pos[::skip,:]
        last_edge = torch.stack((v_us[-1],v_us[0]),dim=0)
        ax.plot(v_us[:,0],v_us[:,1], 'ko-', mfc='w', ms=10)
        ax.plot(last_edge[:,0],last_edge[:,1], 'ko-', mfc='w', ms=10)
        ax.set_xlabel(r'$x/c$ [-]')
        ax.set_ylabel(r'$y/c$ [-]')
        ax.axis('equal')
        ax.set_axis_off()
        
    for ind, graph in tqdm(enumerate(dataset), total=len(dataset)):
        fig, ax = plt.subplots(2,1, layout='constrained', sharex=True)
        ax[0].plot(graph.pos[:,0], graph.pos[:,1], 'o-', alpha=0.75)
        ax[0].set_title('Position')
        ax[0].set_xlabel(r'$x/c$')
        ax[0].set_ylabel(r'$y/c$')

        plot_graph_pressure(graph, dataset, ax = ax[1])
        fig.suptitle(f'index {ind}')
        plt.savefig(f'../out/sample{ind}.png')
        plt.close()

    plot_graph_subsample(graph, skip=5)
    plt.savefig('../out/figures/airfoil_graph.png', dpi=300, transparent=True)
    plot_graph_curvature(graph)
    # plt.show()

    plot_graph_fourier(graph)
    # plt.show()

    plot_graph_pressure(graph, dataset)

    plot_graph_eigenshapes(graph, 0)
    plot_graph_eigenshapes(graph, 1)
    plot_graph_eigenshapes(graph, 5)
    plot_graph_eigenshapes(graph, 10)
    plt.show()




