import matplotlib.pyplot as plt 
import numpy as np
import math
import random
import swmmio
import pyswmm
from pyswmm import Simulation,Subcatchments
import math
import pickle
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import Point, box
import tifffile
import rasterio
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import os   
import re
import argparse
import random
import torch
from systems_pbc import *
import torch.backends.cudnn as cudnn
from utils import *
from visualize import *
import matplotlib.pyplot as plt
from models import FNN3d
from train_utils import Adam
from tqdm import tqdm
from train_utils.losses import GeoPC_loss
import matplotlib.image as pm
from matplotlib.colors import ListedColormap
import torch.nn as nn
# from mpl_toolkits.basemap import Basemap
import tifffile
from data_ import utils
import scipy.ndimage
import torch.nn.functional as F
import boundary
import boundary_gt
from PIL import Image
from pyMesh import visualize2D, setAxisLabel
import train_utils.tensorboard as tb
from AWL import AutomaticWeightedLoss
import torch.nn.functional as F
from hydraulics import saint_venant
from scipy import interpolate
# from skimage.transform import resize
from train_utils.losses import *
from torch.utils.data import DataLoader
from pyMesh import visualize2D
import imageio
import io
import tifffile as tiff
def sum_data_5min(data):
    t=len(data)
    timestep = 10
    output = np.zeros((math.ceil(t/timestep)))
    for i in range(0,math.ceil(t/timestep)):
        output[i] = np.sum(data[i*timestep:(i+1)*timestep])
    return output
def rasterize(path,subs,extent):
  data = {}
  for file in os.listdir(path):
    if file.endswith(".txt"):
        path_txt = os.path.join(path, file)
        match = re.search(r'(\d+mm)', file)
        if match:
            name = match.group(1)
        data_txt = []
        data_txt_sum = []
        for point_id in subs.index:
            data_txt.append(np.loadtxt(path_txt, skiprows=1,usecols=point_id+1))
            data_txt_sum.append(sum_data_5min(data_txt[point_id]))
        data_txt = np.array(data_txt)
        data_txt_sum = np.array(data_txt_sum)[:,:25]
        print(data_txt_sum.shape)
        fig=plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('plots')
        ax.set_xlabel('Time')
        ax.set_ylabel('Runoff')
        for i in range(0,subs.shape[0]):
            ax.plot(data_txt_sum[i])
        plt.show()
        width = extent[2] - extent[0]-1
        height = extent[3] - extent[1]
        print(width,height)
        new_map = np.zeros((math.ceil(height),math.ceil(width),data_txt_sum.shape[1]))   
        points = subs['center_point']
        picxy = points.apply(lambda p: (p.x-extent[0], height-(p.y-extent[1])))
        new_map[picxy.apply(lambda p: math.ceil(p[1])), picxy.apply(lambda p: math.ceil(p[0]))] = data_txt_sum
        print(len(np.where(new_map[:,:,:]!=0)[0]),new_map.shape)
        data[name] = new_map
  return data
def process(runoff_txt_path,inp_file):
    extent1 =[391688.1352,5816723.5617,394900.3042,5819450.5491] 
    # Import the SWMM model
    Bln_1_model=swmmio.Model(inp_file)
    subs=Bln_1_model.subcatchments.dataframe
    # Reset the index to add the Name of the subcatchments as a column to the dataframe
    subs = subs.reset_index()
    # convert the coordinates to a polygon coordination that can be used by geopandas
    subs['geometry']=subs['coords'].apply(lambda coords: Polygon(coords))

    # Create a GeoDataFrame with the specified CRS
    subs = gpd.GeoDataFrame(subs, geometry='geometry', crs='EPSG:25833')

    # drop unnecessary data
    subs.drop(columns=['Raingage', 'Outlet', 'PercImperv', 'Width',
       'PercSlope', 'CurbLength', 'N-Imperv', 'N-Perv', 'S-Imperv', 'S-Perv',
       'PctZero', 'RouteTo', 'PctRouted', 
        'coords', 'Area'],inplace=True)

    # Create a new column with the center points of the polygons
    subs['center_point'] = subs['geometry'].centroid
    # Plot the GeoDataFrame
    ax = subs.plot()
    # Plot the center points on the same plot
    subs['center_point'].plot(ax=ax, color='red', markersize=10)
    extent2 = subs.total_bounds
    extent = [min(extent1[0],extent2[0]), min(extent1[1],extent2[1]), max(extent1[2],extent2[2]), max(extent1[3],extent2[3])]
    runoff=rasterize(runoff_txt_path,subs,extent1)
    return runoff

############################
# Process data
###########################
def inter(array, size):
    h, w = array.shape
    new_h, new_w = np.floor_divide((h, w), size)
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    new_x = np.linspace(0, w - 1, new_w)
    new_y = np.linspace(0, h - 1, new_h)
    f = interpolate.interp2d(x, y, array, kind='linear')
    array_down = f(new_x, new_y)
    # array_down = resize(array, (new_h, new_w), order=1, anti_aliasing=True)
    return array_down

def maxpool_with_nearest_non_zero_torch(array, downsample_factor):
    """
    Perform max pooling on the array using non-overlapping blocks of size `downsample_factor` x `downsample_factor`.
    Each downsampled pixel is assigned the maximum non-zero element from its corresponding block using PyTorch.
    """
    # Convert the numpy array to a PyTorch tensor
    array_tensor = torch.from_numpy(array).float()
    
    # Replace all zeros with a very small negative number to ignore them during max pooling
    array_tensor[array_tensor == 0] = -float('inf')
    
    # Reshape the array to 4D tensor: (batch_size=1, channels=1, height, width)
    array_tensor = array_tensor.unsqueeze(0).unsqueeze(0)
    
    # Apply 2D max pooling with kernel size and stride equal to the downsample factor
    pooled_tensor = torch.nn.functional.max_pool2d(array_tensor, kernel_size=downsample_factor, stride=downsample_factor)
    
    # Replace negative infinity back with zeros
    pooled_tensor[pooled_tensor == -float('inf')] = 0
    
    # Convert back to a NumPy array and remove the extra dimensions
    pooled_array = pooled_tensor.squeeze().numpy()
    
    return pooled_array

def data(path, name, files,runoff):
    t0 = 25
    dt0 = 300
    t00, tfinal = 0, (t0) * dt0
    dx = 30.0 * 16
    # # data
    h_gt = []
    qx_gt = []
    qy_gt = []
    pre=[]
    for i in range(t00,tfinal,dt0):    
        current_time = str(i)
        print('current_time', current_time)
        path_h = os.path.join(path,'r2d_bln1_swmm_b_'+ '%s_%s'%(name,current_time)+'H'+".tif")
        path_u = os.path.join(path, 'r2d_bln1_swmm_b_'+ '%s_%s'%(name,current_time)+'U'+".tif")
        path_v = os.path.join(path, 'r2d_bln1_swmm_b_'+ '%s_%s'%(name,current_time)+'V'+".tif")
        pre_current = np.array(runoff[name][:,:,int(i/300)])
        pre_current = np.nan_to_num(pre_current, nan=0.0)
        pre_current = maxpool_with_nearest_non_zero_torch(pre_current, scale)
        print('pre_current', np.max(pre_current))
        pre_current = torch.from_numpy(pre_current)
        pre_current = pre_current.float()
        h_current = tiff.imread(path_h)
        h_current = np.array(h_current)
        h_current = np.nan_to_num(h_current, nan=0.0)
        h_current = inter(h_current, scale)
        print('h_current', np.max(h_current))
        h_current = torch.from_numpy(h_current)
        h_current = h_current.float()
        qx_current = tiff.imread(path_u)
        qx_current = np.array(qx_current)
        qx_current = np.nan_to_num(qx_current, nan=0.0)
        qx_current = inter(qx_current, scale)
        print('qx_current', np.max(qx_current))
        qx_current = torch.from_numpy(qx_current)
        qx_current = qx_current.float()
        qy_current = tiff.imread(path_v)
        qy_current = np.array(qy_current)
        qy_current = np.nan_to_num(qy_current, nan=0.0)
        qy_current = inter(qy_current, scale)
        print('qy_current', np.max(qy_current))
        qy_current = torch.from_numpy(qy_current)
        qy_current = qy_current.float()
        h_gt.append(h_current)
        qx_gt.append(qx_current)
        qy_gt.append(qy_current)
        pre.append(pre_current)
    h_gt = torch.stack(h_gt, 0)
    u_gt = torch.stack(qx_gt, 0)
    v_gt = torch.stack(qy_gt, 0)
    pre = torch.stack(pre, 0)
    print('len_supervised', h_gt.size())
            
    print('pre',pre.size(),len(np.where(pre!=0)[0]),pre[np.where(pre!=0)],max(pre[np.where(pre!=0)]))
    return h_gt, u_gt, v_gt, pre

class load_data():
    def __init__(self, runoff, path_gt):
        super().__init__()
        self.runoff = runoff
        self.imput_path_val=path_gt
        self.load()

    def load(self):
            runoff=self.runoff
            t0 = 25
            first_value = next(iter(runoff.values()))
            m = int(first_value.shape[0]/scale)
            n = int(first_value.shape[1]/scale)
            lmbdleft, lmbdright = 0, (m - 1)
            thtlower, thtupper = 0, (n - 1)
            dt = 300
            t00, tfinal = 0, (t0 - 1)
            t = np.linspace(t00, tfinal, t0)
            x = np.linspace(lmbdleft, lmbdright, m)
            y = np.linspace(thtlower, thtupper, n)
            data_star = np.hstack((x.flatten(), y.flatten(), t.flatten()))
            lb = data_star.min(0)
            ub = data_star.max(0)
            input_data_list = []
            h_gt_list = []
            u_gt_list = []
            v_gt_list = []
            z_list = []
            for root, directories, files in os.walk(self.imput_path_val):
                for subdirectory in directories:
                  path = os.path.join(root, subdirectory)
                  name = os.path.basename(path)
                  if name and name[0].isdigit():###
                    print(path)
                    print('Name', name)
                    h_gt, u_gt, v_gt, pre = data(path, name, files,self.runoff)
                    gridx = torch.from_numpy(x)
                    gridx = gridx.reshape(1, m, 1, 1, 1).repeat([1, 1, n, t0, 1])
                    gridy = torch.from_numpy(y)
                    gridy = gridy.reshape(1, 1, n, 1, 1).repeat([1, m, 1, t0, 1])
                    gridt = torch.from_numpy(t)
                    gridt = gridt.reshape(1, 1, 1, t0, 1).repeat([1, m, n, 1, 1])
                    gridpre = pre#torch.from_numpy(pre)
                    gridpre = gridpre.reshape(1, m, n, t0, 1)#.repeat([1, m, n, 1, 1])
                    h_init = h_gt[0, :, :]
                    h_init = h_init.reshape(1, m, n, 1, 1).repeat([1, 1, 1, t0, 1])
                    input_data = torch.cat((gridx, gridy, gridt, gridpre), dim=-1)
                    input_data = torch.cat((input_data, h_init.cpu()), dim=-1)
                    input_data = input_data.float()
                    h_gt = torch.unsqueeze(h_gt, dim=0)
                    u_gt = torch.unsqueeze(u_gt, dim=0)
                    v_gt = torch.unsqueeze(v_gt, dim=0)
                    input_data_list.append(input_data)
                    h_gt_list.append(h_gt)
                    u_gt_list.append(u_gt)
                    v_gt_list.append(v_gt)
            data_input = torch.cat(input_data_list, dim=0)
            gt_h = torch.cat(h_gt_list, dim=0)
            gt_u = torch.cat(u_gt_list, dim=0)
            gt_v = torch.cat(v_gt_list, dim=0)
            self.data_input = data_input
            self.gt_h = gt_h
            self.gt_u = gt_u
            self.gt_v = gt_v

    def __getitem__(self, idx):
        return self.data_input[idx], self.gt_h[idx], self.gt_u[idx], self.gt_v[idx]

    def __len__(self, ):
        return self.data_input.shape[0]
#eval
def generate_movie_2D(key, test_x, test_y, preds_y, plot_title='', field=0, val_cbar_index=-1, err_cbar_index=-1,
                      val_clim=None, err_clim=None, font_size=None, movie_dir='', movie_name='movie.gif',
                      frame_basename='movie', frame_ext='jpg', remove_frames=True):
    frame_files = []

    if movie_dir:
        os.makedirs(movie_dir, exist_ok=True)

    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    if len(preds_y.shape) == 4:
        Nsamples, Nx, Ny, Nt = preds_y.shape
        preds_y = preds_y.reshape(Nsamples, Nx, Ny, Nt, 1)
        test_y = test_y.reshape(Nsamples, Nx, Ny, Nt, 1)
    Nsamples, Nx, Ny, Nt, Nfields = preds_y.shape
    print('preds_y', preds_y.shape)

    pred = preds_y[key, ..., field]
    true = test_y[key, ..., field]
    error = torch.abs(pred - true)
    print(error.shape)
    RMSE= torch.sqrt(F.mse_loss(pred,true))
    print("RMSE:",RMSE.shape,RMSE)

    a = test_x[key]
    x = torch.linspace(0, 1, Nx + 1)[:-1]
    y = torch.linspace(0, 1, Ny + 1)[:-1]
    X, Y = torch.meshgrid(x, y)
    t = a[0, 0, :, 2]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    colors = plt.cm.viridis(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    cmap = ListedColormap(colors)

    pcm1 = ax1.pcolormesh(X, Y, true[..., val_cbar_index], cmap=cmap, label='true', shading='gouraud')
    pcm2 = ax2.pcolormesh(X, Y, pred[..., val_cbar_index], cmap=cmap, label='pred', shading='gouraud')
    pcm3 = ax3.pcolormesh(X, Y, error[..., err_cbar_index], cmap=cmap, label='error', shading='gouraud')

    if val_clim is None:
        val_clim = pcm1.get_clim()
    if err_clim is None:
        err_clim = pcm3.get_clim()

    pcm1.set_clim(val_clim)
    plt.colorbar(pcm1, ax=ax1)
    ax1.axis('square')

    pcm2.set_clim(val_clim)
    plt.colorbar(pcm2, ax=ax2)
    ax2.axis('square')

    pcm3.set_clim(err_clim)
    plt.colorbar(pcm3, ax=ax3)
    ax3.axis('square')

    plt.tight_layout()

    for i in range(Nt):
        # Exact
        ax1.clear()
        pcm1 = ax1.pcolormesh(X, Y, true[..., i], cmap=cmap, label='true', shading='gouraud')
        pcm1.set_clim(val_clim)
        ax1.set_title(f'Hydraulic Model {plot_title}(m): Maximum')
        ax1.axis('square')

        # Predictions
        ax2.clear()
        pcm2 = ax2.pcolormesh(X, Y, pred[..., i], cmap=cmap, label='pred', shading='gouraud')
        pcm2.set_clim(val_clim)
        ax2.set_title(f'KI-Tool {plot_title}(m): Maximum')
        ax2.axis('square')

        # Error
        ax3.clear()
        pcm3 = ax3.pcolormesh(X, Y, error[..., i], cmap=cmap, label='error', shading='gouraud')
        pcm3.set_clim(err_clim)
        ax3.set_title(f'Error {plot_title}(m): Maximum')
        ax3.axis('square')

        #         plt.tight_layout()
        fig.canvas.draw()

        if movie_dir:
            frame_path = os.path.join(movie_dir, f'{frame_basename}-{i:03}.{frame_ext}')
            frame_files.append(frame_path)
            plt.savefig(frame_path)

    if movie_dir:
        movie_path = os.path.join(movie_dir, movie_name)
        with imageio.get_writer(movie_path, mode='I') as writer:
            for frame in frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)