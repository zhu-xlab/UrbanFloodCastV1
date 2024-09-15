import argparse
import numpy as np
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
import torch.nn as nn
# from mpl_toolkits.basemap import Basemap
import tifffile
from data import utils
import scipy.ndimage
import boundary
import boundary_gt
from PIL import Image
from pyMesh import visualize2D, setAxisLabel
import train_utils.tensorboard as tb
from AWL import AutomaticWeightedLoss
import torch.nn.functional as F
from hydraulics import saint_venant
from scipy import interpolate
from matplotlib.colors import ListedColormap
# from skimage.transform import resize
from train_utils.losses import *
from torch.utils.data import DataLoader
from pyMesh import visualize2D
import imageio
import io
import os
import tifffile as tiff
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
gpu_ids = [0]
output_device = gpu_ids[0]



################
# Arguments
################
parser = argparse.ArgumentParser(description='GeoPINS')
parser.add_argument('--loss_style', default='mean', help='Loss for the network (MSE, vs. summing).')

parser.add_argument('--visualize', default=True, help='Visualize the solution.')
parser.add_argument('--save_model', default=True, help='Save the model for analysis later.')
# PINO_model
parser.add_argument('--layers', nargs='+', type=int, default=[16, 24, 24, 32, 32], help='Dimensions/layers of the NN')
parser.add_argument('--modes1', nargs='+', type=int, default=[32, 32, 32, 32], help='')
parser.add_argument('--modes2', nargs='+', type=int, default=[32, 32, 32, 32], help='')
parser.add_argument('--modes3', nargs='+', type=int, default=[8, 8, 8, 8], help='')
parser.add_argument('--fc_dim', type=int, default=128, help='')
parser.add_argument('--epochs', type=int, default=15000)
parser.add_argument('--activation', default='gelu', help='Activation to use in the network.')
#train
parser.add_argument('--base_lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--milestones', nargs='+', type=int, default=[500, 1000, 2000, 3000, 4000, 5000], help='')
parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='')
parser.add_argument('--theta', type=float, default=0.7, help='q centered weighting. [0,1].')

args = parser.parse_args()
# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device)
else:
    device = torch.device('cpu')
    print(device)


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

# Parameters
g = torch.tensor(9.80616, dtype=torch.float64)
dem_tif_path = '/mnt/SSD2/qingsong/qinqsong/data_Berlin2/input/moa_bottom.tif'
input_path = '/mnt/SSD2/qingsong/qinqsong/data_Berlin2/Moabit/Moa_Randm/val'
imput_path_val = '/mnt/SSD2/qingsong/qinqsong/data_Berlin2/Moabit/Moa_Randm/val'
man_path = '/mnt/SSD2/qingsong/qinqsong/data_Berlin2/input/moa_rough.tif'
dem_map = tifffile.imread(dem_tif_path)
dem_map = inter(dem_map, 8)
print('dem_map', dem_map.shape)
TILE_SIZE_X = 2000
TILE_SIZE_Y = 2000
ALLOWED_MASKED_PERCENTAGE = 0
MAX_TOPOGRAPHY_DIFFERENCE = 2000

# DEM z
def process_dem(dem_map):
    np_ma_map = np.ma.masked_array(dem_map, mask=(dem_map < -2000))
    np_ma_map = utils.fix_missing_values(np_ma_map)
    dem = torch.from_numpy(np_ma_map)
    return dem.float()
# Precipitation

def cfl(dx: float, max_h: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * dx / (g + max_h)


def data(path, name, files):
    t0 = 25
    dt0 = 300
    t00, tfinal = 0, (t0) * dt0
    dx = 30.0 * 16
    # # data
    h_gt = []
    qx_gt = []
    qy_gt = []
    for i in range(t00,tfinal,dt0):
        current_time = str(i)
        print('current_time', current_time)
        path_h = os.path.join(path, '%s_%s'%(name,current_time)+'H'+".tif")
        path_u = os.path.join(path, '%s_%s'%(name,current_time)+'U'+".tif")
        path_v = os.path.join(path, '%s_%s'%(name,current_time)+'V'+".tif")
        # path_h = path_h.replace("'", "\"")
        # print('path_h', path_h)
        h_current = tiff.imread(path_h)
        h_current = np.array(h_current)
        h_current = np.nan_to_num(h_current, nan=0.0)
        h_current = inter(h_current, 8)
        print('h_current', np.max(h_current))
        h_current = torch.from_numpy(h_current)
        h_current = h_current.float()
        qx_current = tiff.imread(path_u)
        qx_current = np.array(qx_current)
        qx_current = np.nan_to_num(qx_current, nan=0.0)
        qx_current = inter(qx_current, 8)
        print('qx_current', np.max(qx_current))
        qx_current = torch.from_numpy(qx_current)
        qx_current = qx_current.float()
        qy_current = tiff.imread(path_v)
        qy_current = np.array(qy_current)
        qy_current = np.nan_to_num(qy_current, nan=0.0)
        qy_current = inter(qy_current, 8)
        print('qy_current', np.max(qy_current))
        qy_current = torch.from_numpy(qy_current)
        qy_current = qy_current.float()
        h_gt.append(h_current)
        qx_gt.append(qx_current)
        qy_gt.append(qy_current)
    h_gt = torch.stack(h_gt, 0)
    u_gt = torch.stack(qx_gt, 0)
    v_gt = torch.stack(qy_gt, 0)
    print('len_supervised', h_gt.size())
    # pre
    pre = np.zeros((t0))
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".txt"):
                pre_path = os.path.join(root, file)
                print('pre_path',pre_path)
                with open(pre_path, 'r') as fil:
                    for line in fil:
                        condition, value = line.strip().split()
                        m = int(int(condition)/dt0)
                        if m <= t0:
                            pre[m] = value
    print('pre',pre)
    return h_gt, u_gt, v_gt, pre


class train_data():
    def __init__(self, train=True):
        super().__init__()
        # self.data_res = data_res
        # self.pde_res = pde_res
        # self.t_duration = t_duration
        # self.paths = paths
        # self.offset = offset
        # self.n_samples = n_samples
        self.load(train=train)

    def load(self, train=True):
        if train:
            t0 = 25
            # days_train = 12
            # T = 86400
            lmbdleft, lmbdright = 0, (dem_map.shape[0] - 1)
            thtlower, thtupper = 0, (dem_map.shape[1] - 1)
            dt = 300
            t00, tfinal = 0, (t0 - 1)
            m = dem_map.shape[0]
            n = dem_map.shape[1]
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
            for root, directories, files in os.walk(input_path):
                for subdirectory in directories:
                    path = os.path.join(root, subdirectory)
                    name = os.path.basename(path)
                    print(path)
                    print('Name',name)
                    h_gt, u_gt, v_gt, pre = data(path, name, files)
                    gridx = torch.from_numpy(x)
                    gridx = gridx.reshape(1, m, 1, 1, 1).repeat([1, 1, n, t0, 1])
                    gridy = torch.from_numpy(y)
                    gridy = gridy.reshape(1, 1, n, 1, 1).repeat([1, m, 1, t0, 1])
                    gridt = torch.from_numpy(t)
                    gridt = gridt.reshape(1, 1, 1, t0, 1).repeat([1, m, n, 1, 1])
                    gridpre = torch.from_numpy(pre)
                    gridpre = gridpre.reshape(1, 1, 1, t0, 1).repeat([1, m, n, 1, 1])
                    h_init = h_gt[0, :, :]
                    h_init = h_init.reshape(1, m, n, 1, 1).repeat([1, 1, 1, t0, 1])
                    input_data = torch.cat((gridx, gridy, gridt, gridpre), dim=-1)
                    # input_data = 2.0 * (input_data - lb) / (ub - lb) - 1.0
                    input_data = torch.cat((input_data, h_init.cpu()), dim=-1)
                    input_data = input_data.float()
                    h_gt = torch.unsqueeze(h_gt, dim=0)
                    u_gt = torch.unsqueeze(u_gt, dim=0)
                    v_gt = torch.unsqueeze(v_gt, dim=0)
                    # h_init = gen_init(ini_height, ini_discharge, downsampling=True)
                    # data_condition = [h_gt, qx_gt, qy_gt]
                    # data_condition0 = data_condition\
                    z = process_dem(dem_map)
                    z = torch.unsqueeze(z, dim=0)
                    input_data_list.append(input_data)
                    h_gt_list.append(h_gt)
                    u_gt_list.append(u_gt)
                    v_gt_list.append(v_gt)
                    z_list.append(z)
            data_input = torch.cat(input_data_list, dim=0)
            gt_h = torch.cat(h_gt_list, dim=0)
            gt_u = torch.cat(u_gt_list, dim=0)
            gt_v = torch.cat(v_gt_list, dim=0)
            data_z = torch.cat(z_list, dim=0)
            self.data_input = data_input
            self.gt_h = gt_h
            self.gt_u = gt_u
            self.gt_v = gt_v
            self.data_z = data_z
        else:
            t0 = 25
            # days_train = 12
            # T = 86400
            lmbdleft, lmbdright = 0, (dem_map.shape[0] - 1)
            thtlower, thtupper = 0, (dem_map.shape[1] - 1)
            dt = 300
            t00, tfinal = 0, (t0 - 1)
            m = dem_map.shape[0]
            n = dem_map.shape[1]
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
            for root, directories, files in os.walk(imput_path_val):
                for subdirectory in directories:
                    path = os.path.join(root, subdirectory)
                    name = os.path.basename(path)
                    print(path)
                    print('Name', name)
                    h_gt, u_gt, v_gt, pre = data(path, name, files)
                    gridx = torch.from_numpy(x)
                    gridx = gridx.reshape(1, m, 1, 1, 1).repeat([1, 1, n, t0, 1])
                    gridy = torch.from_numpy(y)
                    gridy = gridy.reshape(1, 1, n, 1, 1).repeat([1, m, 1, t0, 1])
                    gridt = torch.from_numpy(t)
                    gridt = gridt.reshape(1, 1, 1, t0, 1).repeat([1, m, n, 1, 1])
                    gridpre = torch.from_numpy(pre)
                    gridpre = gridpre.reshape(1, 1, 1, t0, 1).repeat([1, m, n, 1, 1])
                    h_init = h_gt[0, :, :]
                    h_init = h_init.reshape(1, m, n, 1, 1).repeat([1, 1, 1, t0, 1])
                    input_data = torch.cat((gridx, gridy, gridt, gridpre), dim=-1)
                    # input_data = 2.0 * (input_data - lb) / (ub - lb) - 1.0
                    input_data = torch.cat((input_data, h_init.cpu()), dim=-1)
                    input_data = input_data.float()
                    h_gt = torch.unsqueeze(h_gt, dim=0)
                    u_gt = torch.unsqueeze(u_gt, dim=0)
                    v_gt = torch.unsqueeze(v_gt, dim=0)
                    # h_init = gen_init(ini_height, ini_discharge, downsampling=True)
                    # data_condition = [h_gt, qx_gt, qy_gt]
                    # data_condition0 = data_condition\
                    z = process_dem(dem_map)
                    z = torch.unsqueeze(z, dim=0)
                    input_data_list.append(input_data)
                    h_gt_list.append(h_gt)
                    u_gt_list.append(u_gt)
                    v_gt_list.append(v_gt)
                    z_list.append(z)
            data_input = torch.cat(input_data_list, dim=0)
            gt_h = torch.cat(h_gt_list, dim=0)
            gt_u = torch.cat(u_gt_list, dim=0)
            gt_v = torch.cat(v_gt_list, dim=0)
            data_z = torch.cat(z_list, dim=0)
            self.data_input = data_input
            self.gt_h = gt_h
            self.gt_u = gt_u
            self.gt_v = gt_v
            self.data_z = data_z

    def __getitem__(self, idx):
        return self.data_input[idx], self.gt_h[idx], self.gt_u[idx], self.gt_v[idx], self.data_z[idx]

    def __len__(self, ):
        return self.data_input.shape[0]



def train():
    # # model
    model = FNN3d(modes1=args.modes1, modes2=args.modes2, modes3=args.modes3, fc_dim=args.fc_dim,
                  layers=args.layers).to(device)
    model = nn.DataParallel(model, device_ids=gpu_ids, output_device=output_device)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)
    # PATH = '/mnt/SSD2/qingsong/qinqsong/Berlin_flood/GeoPINS_FD_supervised_2/results_rand/pretrain/checkpoint_4500.pth'
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['state_dict'])
    # # print(checkpoint['state_dict'].keys())
    # print('load model sucessfully epoch', checkpoint['epoch'])
    # log_dir = '/mnt/SSD2/qingsong/qinqsong/data_Berlin2/Moabit/results_rand/'
    # eval(model, log_dir)

    model.train()
    # input data
    epochs = 500
    # else:
    #     epochs = 2500
    pbar = range(epochs)
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
    train_pino = 0.0
    train_loss = 0.0
    model.train()
    trainset = train_data(train=True)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    for e in range(epochs):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            input_data, gt_h, gt_u, gt_v, z = data
            input_data, gt_h, gt_u, gt_v, z = input_data.to(device), gt_h.to(device), gt_u.to(device), gt_v.to(device), z.to(device)
            h_init = input_data[..., 0, -1]
            init_condition = [h_init]
            data_condition = [gt_h, gt_u, gt_v]
            out = model(input_data)
            # print(out.shape)
            # boundary
            output = out.permute(0, 3, 1, 2, 4)
            outputH = output[:, :, :, :, 0].clone()
            # torch.where(outputU > 0.0, outputU, 0.0)
            outputU = output[:, :, :, :, 1].clone()
            outputV = output[:, :, :, :, 2].clone()
            # outputH = F.threshold(outputH, threshold=0, value=0)
            # outputU = F.threshold(outputU, threshold=0, value=0)
            # outputV = F.threshold(outputV, threshold=0, value=0)
            loss_d, loss_c = GeoPC_loss(input_data, outputH, outputU, outputV, data_condition, init_condition)
            total_loss = loss_c + loss_d
            total_loss.backward(retain_graph=True)
            optimizer.step()

            total_loss = total_loss.item()
            # train_pino += loss_f.item()
            # train_loss += total_loss.item()
            # if e % 50 == 0:
            scheduler.step()
            pbar.set_description(
                (
                    f'Epoch {e} '
                    f'loss_d: {loss_d:.5f} '
                    f'loss_c: {loss_c:.5f} '
                )
            )
            # loss
            tb.log_scalars(e, write_hparams=True,
                           loss_d=loss_d)
        if (e+1)%100 == 0:
            log_dir = '/mnt/SSD2/qingsong/qinqsong/data_Berlin2/Moabit/results_rand/'
            eval(model, log_dir)
            # eval_high(model, dem_map, log_dir, t0, i, nt)
            # if args.save_model == True:
            #     state_dict = model.state_dict()
            #     torch.save({'epoch': e, 'state_dict': state_dict},
            #                log_dir + f'pretrain/checkpoint_%d.pth'%(e))
            # torch.cuda.empty_cache()
    print('Done!')


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
        ax1.set_title(f'Hydraulic Model {plot_title}: Maximum')
        ax1.axis('square')

        # Predictions
        ax2.clear()
        pcm2 = ax2.pcolormesh(X, Y, pred[..., i], cmap=cmap, label='pred', shading='gouraud')
        pcm2.set_clim(val_clim)
        ax2.set_title(f'KI-Tool {plot_title}: Maximum')
        ax2.axis('square')

        # Error
        ax3.clear()
        pcm3 = ax3.pcolormesh(X, Y, error[..., i], cmap=cmap, label='error', shading='gouraud')
        pcm3.set_clim(err_clim)
        ax3.set_title(f'Error {plot_title}: Maximum')
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

    # if movie_dir and remove_frames:
    #     for frame in frame_files:
    #         try:
    #             os.remove(frame)
    #         except:
    #             pass

def eval(model, log_dir):
    model.eval()
    avg_err_hr = []
    avg_err_ha = []
    avg_err_ur = []
    avg_err_ua = []
    avg_err_vr = []
    avg_err_va = []
    t0 = 25
    valset = train_data(train=False)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
    # lmbdleft, lmbdright = 0, (dem_map.shape[0] - 1)
    # thtlower, thtupper = 0, (dem_map.shape[1] - 1)
    # m = dem_map.shape[0]
    # n = dem_map.shape[1]
    # x = np.linspace(lmbdleft, lmbdright, m)
    # y = np.linspace(thtlower, thtupper, n)
    # X, Y = torch.meshgrid(x, y, indexing='ij')
    key = 0
    for i, data in enumerate(val_loader):
        input_data, gt_h, gt_u, gt_v, z = data
        input_data, gt_h, gt_u, gt_v, z = input_data.to(device), gt_h.to(device), gt_u.to(device), gt_v.to(device), z.to(device)
        gt_hm, gt_um, gt_vm = gt_h.permute(0, 2, 3, 1), gt_u.permute(0, 2, 3, 1), gt_v.permute(0, 2, 3, 1)
        gt_hm, gt_um, gt_vm = torch.unsqueeze(gt_hm, dim=-1), torch.unsqueeze(gt_um, dim=-1), torch.unsqueeze(gt_vm, dim=-1)
        gt_m = torch.cat((gt_hm, gt_um, gt_vm), dim=-1)
        gt_m, _ = torch.max(gt_m, dim=3, keepdim=True)
        gt_m = torch.rot90(gt_m, k=-1, dims=[1,2])
        print('gt_m', gt_m.shape)
        with torch.no_grad():
            out = model(input_data)
            outm, _ = torch.max(out, dim=3, keepdim=True)
            outm = torch.rot90(outm, k=-1, dims=[1,2])
            outm = torch.where(gt_m>0, outm, gt_m)
            print('outm', outm.shape)
        #MOIVE
        movie_dir = '/mnt/SSD2/qingsong/qinqsong/data_Berlin2/Moabit/results_rand/movie/%s/'%(str(i))
        os.makedirs(movie_dir, exist_ok=True)
        #H
        movie_name = 'H.gif'
        frame_basename = 'H_frame'
        frame_ext = 'jpg'
        plot_title = "$H$"
        field = 0
        val_cbar_index = -1
        err_cbar_index = -1
        font_size = 12
        remove_frames = True
        generate_movie_2D(key, input_data.cpu(), gt_m.cpu(), outm.cpu(),
                          plot_title=plot_title,
                          field=field,
                          val_cbar_index=val_cbar_index,
                          err_cbar_index=err_cbar_index,
                          movie_dir=movie_dir,
                          movie_name=movie_name,
                          frame_basename=frame_basename,
                          frame_ext=frame_ext,
                          remove_frames=remove_frames,
                          font_size=font_size)

        # #U
        movie_name = 'U.gif'
        frame_basename = 'U_frame'
        frame_ext = 'jpg'
        plot_title = "$U$"
        field = 1
        val_cbar_index = -1
        err_cbar_index = -1
        font_size = 12
        remove_frames = True
        generate_movie_2D(key, input_data.cpu(), gt_m.cpu(), outm.cpu(),
                          plot_title=plot_title,
                          field=field,
                          val_cbar_index=val_cbar_index,
                          err_cbar_index=err_cbar_index,
                          movie_dir=movie_dir,
                          movie_name=movie_name,
                          frame_basename=frame_basename,
                          frame_ext=frame_ext,
                          remove_frames=remove_frames,
                          font_size=font_size)
        #
        # #V
        movie_name = 'V.gif'
        frame_basename = 'V_frame'
        frame_ext = 'jpg'
        plot_title = "$V$"
        field = 2
        val_cbar_index = -1
        err_cbar_index = -1
        font_size = 12
        remove_frames = True

        generate_movie_2D(key, input_data.cpu(), gt_m.cpu(), outm.cpu(),
                          plot_title=plot_title,
                          field=field,
                          val_cbar_index=val_cbar_index,
                          err_cbar_index=err_cbar_index,
                          movie_dir=movie_dir,
                          movie_name=movie_name,
                          frame_basename=frame_basename,
                          frame_ext=frame_ext,
                          remove_frames=remove_frames,
                          font_size=font_size)

        # h, h05, qx, qy = F.threshold(h, threshold=0, value=0), F.threshold(h, threshold=0.05, value=0), F.threshold(qx, threshold=0, value=0), F.threshold(qy, threshold=0, value=0)
        # h_gt, qx_gt, qy_gt = data(i)
        # h, u, v = out[:, :, :, :, :1], out[:, :, :, :, 1:2], out[:, :, :, :, 2:3]
        # h, u, v = torch.squeeze(h), torch.squeeze(u), torch.squeeze(v)
        # h, u, v = h.permute(2, 0, 1), u.permute(2, 0, 1), v.permute(2, 0, 1)
        # h_g = torch.squeeze(gt_h)
        # u_g = torch.squeeze(gt_u)
        # v_g = torch.squeeze(gt_v)
        # h_p = h.detach().cpu().numpy()
        # # h_p = h_p.reshape(-1, 1)
        # u_p = u.detach().cpu().numpy()
        # # u_p = u_p.reshape(-1, 1)
        # v_p = v.detach().cpu().numpy()
        # # v_p = v_p.reshape(-1, 1)
        #
        # h_gt = h_g.detach().cpu().numpy()
        # print('h_gt', h_gt.shape)
        # print('h_p', h_p.shape)
        # # h_gt = h_gt.reshape(-1, 1)
        # u_gt = u_g.detach().cpu().numpy()
        # # u_gt = u_gt.reshape(-1, 1)
        # v_gt = v_g.detach().cpu().numpy()
        # # v_gt = v_gt.reshape(-1, 1)
        # mask_h = np.where(h_gt != 0, 1, 0)
        # mask_u = np.where(u_gt != 0, 1, 0)
        # mask_v = np.where(v_gt != 0, 1, 0)
        # h_p = mask_h * h_p
        # u_p = mask_u * u_p
        # v_p = mask_v * v_p
        #
        # # error_h_relative = np.linalg.norm(h_gt - h_p, 2) / np.linalg.norm(h_gt, 2)
        # # error_u_relative = np.linalg.norm(u_gt - u_p, 2) / np.linalg.norm(u_gt, 2)
        # # error_v_relative = np.linalg.norm(v_gt - v_p, 2) / np.linalg.norm(v_gt, 2)
        # error_h_abs = np.mean(np.abs(h_gt - h_p))
        # error_u_abs = np.mean(np.abs(u_gt - u_p))
        # error_v_abs = np.mean(np.abs(v_gt - v_p))
        # # avg_err_hr.append(error_h_relative)
        # # avg_err_ha.append(error_h_abs)
        # # avg_err_ur.append(error_u_relative)
        # # avg_err_ua.append(error_u_abs)
        # # avg_err_vr.append(error_v_relative)
        # # avg_err_va.append(error_v_abs)
        # #Max MAE
        # error_hmax_abs = np.mean(np.abs(np.max(h_gt, axis=0) - np.max(h_p, axis=0)))
        # error_umax_abs = np.mean(np.abs(np.max(u_gt, axis=0) - np.max(u_p, axis=0)))
        # error_vmax_abs = np.mean(np.abs(np.max(v_gt, axis=0) - np.max(v_p, axis=0)))
        # # MAE for every time step
        # hh = np.abs(h_gt - h_p)
        # error_hh = np.mean(hh, axis=1)
        # error_h_everytime = np.mean(error_hh, axis=1)
        #
        # uu = np.abs(u_gt - u_p)
        # error_uu = np.mean(uu, axis=1)
        # error_u_everytime = np.mean(error_uu, axis=1)
        #
        # vv = np.abs(v_gt - v_p)
        # error_vv = np.mean(vv, axis=1)
        # error_v_everytime = np.mean(error_vv, axis=1)
        #
        # #MAE  bigger than 1 cm
        # h_gt1 = h_gt
        # h_p1 = h_p
        # u_gt1 = u_gt
        # u_p1 = u_p
        # v_gt1 = v_gt
        # v_p1 = v_p
        # h_gt1[h_gt1 < 0.1] = 0
        # h_p1[h_p1 < 0.1] = 0
        # u_gt1[u_gt1 < 0.1] = 0
        # u_p1[u_p1 < 0.1] = 0
        # v_gt1[v_gt1 < 0.1] = 0
        # v_p1[v_p1 < 0.1] = 0
        # error_h_01 = np.mean(np.abs(h_gt1 - h_p1))
        # error_u_01 = np.mean(np.abs(u_gt1 - u_p1))
        # error_v_01 = np.mean(np.abs(v_gt1 - v_p1))
        #
        # ##MAE  bigger than 5 cm
        # h_gt5 = h_gt
        # h_p5 = h_p
        # u_gt5 = u_gt
        # u_p5 = u_p
        # v_gt5 = v_gt
        # v_p5 = v_p
        # h_gt5[h_gt5 < 0.5] = 0
        # h_p5[h_p5 < 0.5] = 0
        # u_gt5[u_gt5 < 0.5] = 0
        # u_p5[u_p5 < 0.5] = 0
        # v_gt5[v_gt5 < 0.5] = 0
        # v_p5[v_p5 < 0.5] = 0
        # error_h_05 = np.mean(np.abs(h_gt5 - h_p5))
        # error_u_05 = np.mean(np.abs(u_gt5 - u_p5))
        # error_v_05 = np.mean(np.abs(v_gt5 - v_p5))
        #
        # print('error_h_abs',error_h_abs)
        # print('error_u_abs', error_u_abs)
        # print('error_v_abs', error_v_abs)
        #
        #
        # print('error_hmax_abs', error_hmax_abs)
        # print('error_umax_abs', error_umax_abs)
        # print('error_vmax_abs', error_vmax_abs)
        #
        #
        # print('error_h_everytime', error_h_everytime)
        # print('error_u_everytime', error_u_everytime)
        # print('error_v_everytime', error_v_everytime)
        #
        # print('error_h_01', error_h_01)
        # print('error_u_01', error_u_01)
        # print('error_v_01', error_v_01)
        #
        # print('error_h_05', error_h_05)
        # print('error_u_05', error_u_05)
        # print('error_v_05', error_v_05)

        # val_loss_h = criterion(h, h_g)
        # val_err_h.append(val_loss_h.item())
        # rel_loss_h = criterion2(h, h_g)
        # rel_err_h.append(rel_loss_h.item())
        # val_loss_u = criterion(u, u_g)
        # val_err_u.append(val_loss_u.item())
        # rel_loss_u = criterion2(u, u_g)
        # rel_err_u.append(rel_loss_u.item())
        # val_loss_v = criterion(v, v_g)
        # val_err_v.append(val_loss_v.item())
        # rel_loss_v = criterion2(v, v_g)
        # rel_err_v.append(rel_loss_v.item())
        # N = len(val_err_h)
        # avg_err_h = np.mean(val_err_h)
        # std_err_h = np.std(val_err_h, ddof=1) / np.sqrt(N)
        # avg_err_u = np.mean(val_err_u)
        # std_err_u = np.std(val_err_u, ddof=1) / np.sqrt(N)
        # avg_err_v = np.mean(val_err_v)
        # std_err_v = np.std(val_err_v, ddof=1) / np.sqrt(N)
    # file0 = "/mnt/SSD1/qinqsong/Berlin_flood/GeoPINS_FD_supervised_2/results_rand/avg_err_hr.txt"
    # with open(file0, 'a', encoding='utf-8') as f0:
    #     f0.writelines(str(avg_err_hr) + '\n')
    # file1 = "/mnt/SSD1/qinqsong/Berlin_flood/GeoPINS_FD_supervised_2/results_rand/avg_err_ha.txt"
    # with open(file1, 'a', encoding='utf-8') as f1:
    #     f1.writelines(str(avg_err_ha) + '\n')
    #
    # file01 = "/mnt/SSD1/qinqsong/Berlin_flood/GeoPINS_FD_supervised_2/results_rand/avg_err_ur.txt"
    # with open(file01, 'a', encoding='utf-8') as f2:
    #     f2.writelines(str(avg_err_ur) + '\n')
    # file11 = "/mnt/SSD1/qinqsong/Berlin_flood/GeoPINS_FD_supervised_2/results_rand/avg_err_ua.txt"
    # with open(file11, 'a', encoding='utf-8') as f3:
    #     f3.writelines(str(avg_err_ua) + '\n')
    #
    # file02 = "/mnt/SSD1/qinqsong/Berlin_flood/GeoPINS_FD_supervised_2/results_rand/avg_err_vr.txt"
    # with open(file02, 'a', encoding='utf-8') as f4:
    #     f4.writelines(str(avg_err_vr) + '\n')
    # file12 = "/mnt/SSD1/qinqsong/Berlin_flood/GeoPINS_FD_supervised_2/results_rand/avg_err_va.txt"
    # with open(file12, 'a', encoding='utf-8') as f5:
    #     f5.writelines(str(avg_err_va) + '\n')
        


def eval_high(model, dem_map, log_dir, t0, i, nt):
    dem_map = dem_map[::12, ::12]
    print('dem_map_high', dem_map.shape)
    model.eval()
    days = 14
    t00, tfinal = 0, days * 24 * 60 * 60
    dt = 30
    lmbdleft, lmbdright = 0, dem_map.shape[0] * 30
    thtlower, thtupper = 0, dem_map.shape[1] * 30
    m = dem_map.shape[0]
    n = dem_map.shape[1]
    l = int((days * 24 * 60 * 60) / dt)

    ta = np.linspace(t00, tfinal, l)
    x = np.linspace(lmbdleft, lmbdright, m)
    y = np.linspace(thtlower, thtupper, n)
    data_star = np.hstack((x.flatten(), y.flatten(), ta.flatten()))
    lb = data_star.min(0)
    ub = data_star.max(0)
    #input_data
    gridx = torch.from_numpy(x)
    gridx = gridx.reshape(1, m, 1, 1, 1).repeat([1, 1, n, t0, 1])
    gridy = torch.from_numpy(y)
    gridy = gridy.reshape(1, 1, n, 1, 1).repeat([1, m, 1, t0, 1])
    t = ta[i * (t0 - 1):(i + 1) * (t0 - 1) + 1]
    gridt = torch.from_numpy(t)
    gridt = gridt.reshape(1, 1, 1, t0, 1).repeat([1, m, n, 1, 1])
    input_data = torch.cat((gridt, gridx, gridy), dim=-1)
    input_data = 2.0 * (input_data - lb) / (ub - lb) - 1.0
    input_data = input_data.float().to(device)
    with torch.no_grad():
        out = model(input_data)
    h, qx, qy = out[:,:,:,:,:1], out[:,:,:,:,1:2], out[:,:,:,:,2:3]
    h, h05, qx, qy = F.threshold(h, threshold=0, value=0), F.threshold(h, threshold=0.05, value=0), F.threshold(qx, threshold=0, value=0), F.threshold(qy, threshold=0, value=0)
    h, h05, qx, qy = torch.squeeze(h), torch.squeeze(h05), torch.squeeze(qx), torch.squeeze(qy)
    h, h05, qx, qy = h.permute(2,0,1), h05.permute(2,0,1), qx.permute(2,0,1), qy.permute(2,0,1)
    h, h05, qx, qy = h.cpu().detach().numpy(), h05.cpu().detach().numpy(), qx.cpu().detach().numpy(), qy.cpu().detach().numpy()
    plot_steps = [n for n in range(t0)]
    for m in plot_steps:
        time = i*(t0-1) + m
        # save as Image
        _EPSILON = 1e-6
        h_a = h[m, :, :]
        h_a_05 = h05[m, :, :]
        qx_a = qx[m, :, :]
        qy_a = qy[m, :, :]
        q_a = (qx_a ** 2 + qy_a ** 2 + _EPSILON) ** 0.5
        # save
        h_a_m = np.mean(h_a)
        q_a_m = np.mean(q_a)
        file11 = "/mnt/SSD2/qingsong/qinqsong/data_Berlin2/Moabit/results_rand/height_high.txt"
        with open(file11, 'a', encoding='utf-8') as f11:
            f11.writelines(str(h_a_m) + '\n')
        file12 = "/mnt/SSD2/qingsong/qinqsong/data_Berlin2/Moabit/results_rand/discharge_high.txt"
        with open(file12, 'a', encoding='utf-8') as f12:
            f12.writelines(str(q_a_m) + '\n')
        h_aa = Image.fromarray(h_a)
        h_aa_05 = Image.fromarray(h_a_05)
        qx_aa = Image.fromarray(qx_a)
        qy_aa = Image.fromarray(qy_a)
        q_aa = Image.fromarray(q_a)
        h_aa.save(os.path.join(log_dir, 'h_high/%s.tiff'%time))
        h_aa_05.save(os.path.join(log_dir, 'h05_high/%s.tiff' % time))
        qx_aa.save(os.path.join(log_dir, 'qx_high/%s.tiff' % time))
        qy_aa.save(os.path.join(log_dir, 'qy_high/%s.tiff' % time))
        q_aa.save(os.path.join(log_dir, 'q_high/%s.tiff' % time))


if __name__ == '__main__':
    train()