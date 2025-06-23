import torch
import os
import numpy as np
import datetime
import os
import random
import time

from models.FNO import FNO2d, FNO3d
from models.Unet import UNet2d, UNet3d
# from models.GFNO_steerable import GFNO2d_steer
# from models.Unet import Unet_Rot, Unet_Rot_M, Unet_Rot_3D
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
import pandas as pd
from models.DNO import DNO

from utils25 import  LpLoss, nse, corr, critical_success_index

import scipy
import numpy as np
from timeit import default_timer
import argparse
from torch.utils.tensorboard import SummaryWriter as writer
import torch
import h5py
import xarray as xr
from tqdm import tqdm
from openpyxl import load_workbook
import io
import os
import tifffile as tiff
import torch.nn.functional as F
zoom_factor=1
width=int(3213/10)
height=int(2727/10)
################################################################
# Dataset class
################################################################
class flood_data(torch.utils.data.Dataset):
    def __init__(self, path_root, T_in, T_out=None, train=True, strategy="markov", std=0.0):
        self.markov = strategy == "markov"
        self.teacher_forcing = strategy == "teacher_forcing"
        self.one_shot = strategy == "oneshot"
        self.path_root = path_root
        # self.data = data[..., :(T_in + T_out)] if self.one_shot else data[..., :(T_in + T_out), :]
        self.data = []
        pt_files = [name for name in os.listdir(self.path_root) if name.endswith('.pt')]
        n = len(pt_files)
        print('Number of .pt files:', n)
        self.data = [os.path.join(self.path_root, name) for name in pt_files]
        self.nt = T_in + T_out
        self.T_in = T_in
        self.T_out = T_out
        self.num_hist = 1 if self.markov else self.T_in
        self.train = train
        self.noise_std = std

    def log_transform(self, data, eps=1e-2):
        return torch.log(1 + data/eps)

    def __len__(self):
        if self.train:
            if self.markov:
                return len(self.data) * (self.nt - 1)
            if self.teacher_forcing:
                return len(self.data) * (self.nt - self.T_in)
        return len(self.data)
    def downsample_data(self, data, scale_factor=0.125):
        data = data.permute(2, 3, 0, 1)  # (3213, 2727, 25, 5) -> (5, 25, 3213, 2727)
        data = F.interpolate(data, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        data = data.permute(2, 3, 0, 1)  
        return data

    def __getitem__(self, idx):
        if not self.train or not (self.markov or self.teacher_forcing): # full target: return all future steps
            pde_path = self.data[idx]
            # path_idx = os.path.join(path_root, str(idx) + ".pt")
            pde = torch.load(pde_path)
            pde = self.downsample_data(pde,scale_factor=1/zoom_factor)
            # pde = pde.permute(1, 2, 0, 3)
            if self.one_shot:
                x = pde[..., :self.T_in, :3]
                mask = (x[..., 0:1] == 0.0)
                x[..., 1:2][mask] = 0.0
                x[..., 2:3][mask] = 0.0
                x[..., :3] = torch.nan_to_num(x[..., :3], nan=0.0)
                x1 = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                # x_n = x.numpy()
                # x_n = np.nan_to_num(x_n, nan=-99999)
                # x_n = np.ma.masked_array(x_n, mask=(x_n < -2000))
                # x = torch.from_numpy(x_n).float()
                # nan_mask = torch.isnan(x)
                # contains_nan = torch.any(nan_mask)
                # print('contains_nan', contains_nan)
                p = pde[..., self.T_in:(self.T_in + self.T_out), 3:4]
                x2 = self.log_transform(p) / 10.0
                x2 = torch.unsqueeze(x2, dim = -1)
                z = pde[..., self.T_in:(self.T_in + self.T_out), 4:5]
                max_z = z.max()
                z_value = max_z + 30.0
                z = torch.nan_to_num(z, nan=z_value)
                x3 = torch.nn.functional.normalize(z)
                x3 = torch.unsqueeze(x3, dim=-1)
                x = torch.cat((x1, x2), dim=-1)
                x = torch.cat((x, x3), dim=-1)
                # x = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                y = pde[..., self.T_in:(self.T_in + self.T_out), :3]
                mask_y = (y[..., 0:1] == 0.0)
                y[..., 1:2][mask_y] = 0.0
                y[..., 2:3][mask_y] = 0.0
                mask_tensor = ~torch.isnan(y)
                y = torch.nan_to_num(y, nan=0.0)
            else:
                x = pde[..., (self.T_in - self.num_hist):self.T_in, :3]
                mask = (x[..., 0:1] == 0.0)
                x[..., 1:2][mask] = 0.0
                x[..., 2:3][mask] = 0.0
                x[..., :3] = torch.nan_to_num(x[..., :3], nan=0.0)
                x1 = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                # x_n = x.numpy()
                # x_n = np.nan_to_num(x_n, nan=-99999)
                # x_n = np.ma.masked_array(x_n, mask=(x_n < -2000))
                # x = torch.from_numpy(x_n).float()
                # nan_mask = torch.isnan(x)
                # contains_nan = torch.any(nan_mask)
                # print('contains_nan', contains_nan)
                p = pde[..., self.T_in:(self.T_in + self.T_out), 3:4]
                x2 = self.log_transform(p) / 10.0
                z = pde[..., self.T_in:(self.T_in + self.T_out), 4:5]
                max_z = z.max()
                z_value = max_z + 30.0
                z = torch.nan_to_num(z, nan=z_value)
                x3 = torch.nn.functional.normalize(z)
                x = torch.cat((x1, x2), dim=-1)
                x = torch.cat((x, x3), dim=-1)
                # x[..., :4] = self.log_transform(x[..., :4])
                y = pde[..., self.T_in:(self.T_in + self.T_out), :3]
                mask_y = (y[..., 0:1] == 0.0)
                y[..., 1:2][mask_y] = 0.0
                y[..., 2:3][mask_y] = 0.0
                mask_tensor = ~torch.isnan(y)
                y = torch.nan_to_num(y, nan=0.0)
            return x, y, mask_tensor
        pde_idx = idx // (self.nt - self.num_hist) # Markov / teacher forcing: only return one future step
        t_idx = idx % (self.nt - self.num_hist) + self.num_hist
        pde_path = self.data[pde_idx]
        # path_idx = os.path.join(path_root, str(pde_idx) + ".pt")
        # pde = torch.load(path_idx)
        pde = torch.load(pde_path)
        pde = pde.permute(1, 2, 0, 3)
        x = pde[..., (t_idx - self.num_hist):t_idx, :]
        mask = (x[..., 0:1] == 0.0)
        x[..., 1:2][mask] = 0.0
        x[..., 2:3][mask] = 0.0
        # mask_tensor = torch.isnan(x[..., :3])
        x[..., :3] = torch.nan_to_num(x[..., :3], nan=0.0)
        max_z = x[..., -1].max()
        z_value = max_z + 30.0
        x[..., -1] = torch.nan_to_num(x[..., -1], nan=z_value)
        x[..., -1] = torch.nn.functional.normalize(x[..., -1])
        x[..., 3:4] = self.log_transform(x[..., 3:4]) / 10.0
        # x[..., :4] = self.log_transform(x[..., :4])
        y = pde[..., t_idx, :3]
        mask_y = (y[..., 0:1] == 0.0)
        y[..., 1:2][mask_y] = 0.0
        y[..., 2:3][mask_y] = 0.0
        mask_tensor = ~torch.isnan(y)
        y = torch.nan_to_num(y, nan=0.0)
        if self.noise_std > 0:
            x += torch.randn(*x.shape, device=x.device) * self.noise_std

        return x, y, mask_tensor
