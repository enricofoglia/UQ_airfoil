import os 
import logging

import torch 

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import airfrans as af

from cmcrameri import cm

PATH = '/home/daep/e.foglia/Documents/1A/05_uncertainty_quantification/data/AirfRANS/Dataset'

# set up logger
import logging

# Custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File handler with rotating logs (max size 1 MB, 3 backups)
file_handler = logging.FileHandler('out.logs')
file_handler.setLevel(logging.DEBUG)

# Console handler (optional)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"],
    "text.latex.preamble": r"\usepackage{amsmath,amsfonts}\usepackage[cm]{sfmath}",
    'axes.linewidth' : 2,
    'lines.linewidth' : 2,
    'axes.labelsize' : 14,
    'xtick.labelsize' : 12,
    'ytick.labelsize' : 12,
    'axes.titlesize': 'large'
})

def plot_airfoil_field(data, field_name='u', near_field=True):
    '''
    Plot an instance of the airfrans dataset.

    Args:
        data (np.array): datapoint
        field_name (str, optional): name of the plotted field. Possible options are "u" (axial velocity),  "v" (vertical velocity), "p" (pressure :math:`-p_\infty`), "nut" (turbulent viscosity) (default :obj:`u`)
        near_field (bool, optional): if :obj:`True`, plot airfoil within the rectangle :math:`[-0.5,1.5]\times[-0.5,0.5]` (default :obj:`True`)
    '''
    x,y = data[:,0:2].T

    if near_field:
        max_x = 1.5; min_x = -0.5
        max_y = 0.5; min_y = -0.5
    else:
        max_x = max(x); min_x = min(x)
        max_y = max(y); min_y = min(y)

    if field_name == 'u':
        field = data[:,7]
        label = r'$u$ [ms$^{-1}$]'
    elif field_name == 'v':
        field = data[:,8]
        label = r'$v$ [ms$^{-1}$]'
    elif field_name == 'p':
        field = data[:,9]
        label = r'$p$ [Pa]'
    elif field_name == 'nut':
        field = data[:,10]
        label = r'$\nu_t$ [m$^2$s$^{-1}$]'
    else:
        raise ValueError(f'No field {field_name} in data.' 
                        'Allowed options are "u", "v", "p", "nut"')
    
    min_field = min(field); max_field = max(field)
    levels = np.linspace(min_field, max_field, 16, endpoint=True)
    
    # extract airfoil skin
    skin_idx = np.argwhere(data[:,4]==0)
    skin_x = x[skin_idx]; skin_y = y[skin_idx] 
    
    grid_x, grid_y = np.mgrid[min_x:max_x:200j, min_y:max_y:200j]
    grid_field = griddata((x, y), field, (grid_x, grid_y), method='cubic')

    fig,ax = plt.subplots(figsize=(8,3.5), layout='tight')
    cs = ax.contourf(grid_x, grid_y, grid_field, levels=levels,
                    vmin=min_field, vmax=max_field, cmap=cm.lipari, extend='both')
    ax.contour(grid_x, grid_y, grid_field, levels=levels, colors='k', linewidths=0.5)
                
    ax.scatter(skin_x, skin_y, c='k', s=10)

    # Adjust the axis to wrap tightly around the plot
    ax.axis('tight')       # Automatically sets limits to data range
    ax.margins(0)          # Removes extra margins
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(r'$x/c$ [-]')
    ax.set_ylabel(r'$y/c$ [-]')

    cbar = fig.colorbar(cs, ax=ax, )

    cbar.ax.set_ylabel(label)

    # plt.show()

def extract_skin(data):
    # extract airfoil skin
    skin_idx = np.argwhere(data[:,4]==0)
    return data[skin_idx,:]

def order_points(skin_data):
    '''
    Orders 2D airfoil points counter-clockwise, starting from the trailing edge.

    Identifies the trailing edge as the point with the largest x-coordinate, then orders 
    all points counter-clockwise based on their angles relative to the airfoil's centroid.

    Args:
        skin_data (np.ndarray): A (N, K) array of airfoil points where the first two columns are the (x, y) coordinates.

    Returns:
        np.ndarray: A (N, K) array of points ordered counter-clockwise from the trailing edge.
    '''

    trailing_edge_idx = np.argmax(skin_data[:, 0])

    # Step 3: Calculate the centroid (mean point)
    centroid = np.mean(skin_data, axis=0)[0:2]

    # Step 4: Compute angles from the centroid to each point relative to the trailing edge
    # Use atan2 to get the angle in the range (-pi, pi)
    angles = np.arctan2(skin_data[:, 1] - centroid[1], skin_data[:, 0] - centroid[0])

    # Step 5: Sort the points based on the angle
    sorted_indices = np.argsort(angles)
    sorted_skin_data = skin_data[sorted_indices]

    # Step 6: Ensure the trailing edge is the starting point
    # Find where the trailing edge is in the sorted list and reorder such that it is the first point
    trailing_edge_sorted_idx = np.where(sorted_indices == trailing_edge_idx)[0][0]
    ordered_skin_data = np.roll(sorted_skin_data, -trailing_edge_sorted_idx, axis=0)

    return ordered_skin_data

if __name__ == '__main__':

    dataset_list, dataset_name = af.dataset.load(root = PATH, task = 'scarce', train = False)

    logger.debug( 'dataset_list : some prints')
    logger.debug(f'>> type   : {type(dataset_list)}')
    logger.debug(f'>> length : {len(dataset_list)}')
    logger.debug(f'>> first element type   : {type(dataset_list[0])}')
    logger.debug(f'>> first element length : {len(dataset_list[0])}')
    logger.debug(f'>> first element shape  : {dataset_list[0].shape}')
    logger.debug('')
    logger.debug( 'dataset name : some prints')
    logger.debug(f'First element name: {dataset_name[0]}')


    plot_airfoil_field(dataset_list[1], 'u', near_field=True)
    plot_airfoil_field(dataset_list[1], 'v', near_field=True)
    plot_airfoil_field(dataset_list[1], 'p')
    plot_airfoil_field(dataset_list[1], 'nut')
    # plt.show()

    # extract skin
    skin_data = extract_skin(dataset_list[1])
    skin_data = order_points(skin_data)
    fig, ax = plt.subplots()
    sc = ax.scatter(skin_data[:,0], skin_data[:,1], c=skin_data[:,9],
                    cmap=cm.glasgow)
    fig.colorbar(sc, ax=ax)
    ax.axis('scaled')
    plt.show()