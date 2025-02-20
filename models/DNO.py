# Codes for section: Results on Navier Stocks Equation (3D)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .integral_operators import *
import matplotlib.pyplot as plt
from .utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from .Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


#########

class DNO(nn.Module):
    """
    The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .

    input: the solution of the first 10 timesteps (u(1), ..., u(10)).
    input shape: (batchsize, x=S, y=S, t=T, c=1)
    output: the solution of the next 20 timesteps
    output shape: (batchsize, x=S, y=S, t=2*T, c=1)

    S,S,T = grid size along x,y and t (input function)
    S,S,2*T = grid size along x,y and t (output function)

    in_width = 4; [a(x,y,x),x,y,z]
    with = projection dimesion
    pad = padding amount
    pad_both = boolean, if true pad both size of the domain
    factor = scaling factor of the co-domain dimesions
    """

    def __init__(self, num_channels, width, initial_step, pad=2, factor=1, pad_both=False):
        super(DNO, self).__init__()

        self.in_width = initial_step * num_channels + 3  # input channel
        self.width = width

        self.padding = pad
        self.pad_both = pad_both

        # self.fc = nn.Linear(self.in_width, self.in_width*2)
        #
        # self.fc0 = nn.Linear(self.in_width*2, self.width)
        #
        # self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,48, 48, 10, 22,22, 8, Normalize = True)
        #
        # self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 32, 32,10,  14,14,8)
        #
        # # self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 12,  10,10,6)
        # #
        # # self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 16, 16, 12,  10,10,6, Normalize = True)
        # #
        # # self.conv6 = OperatorBlock_3D(16*factor*self.width, 4*factor*self.width, 32, 32, 18,  10,10,6)
        #
        # self.conv7 = OperatorBlock_3D(4*factor*self.width, 2*factor*self.width, 48, 48, 20,  14,14,8, Normalize = True)
        #
        # self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 64, 64, 20,  22,22, 8) # will be reshaped
        #
        # self.fc1 = nn.Linear(3*self.width, 4*self.width)
        # self.fc2 = nn.Linear(4*self.width, 3)
        self.fc = nn.Linear(self.in_width, self.in_width * 2)

        self.fc0 = nn.Linear(self.in_width * 2, self.width)

        self.conv0 = OperatorBlock_3D(self.width, 1*factor*self.width, 48, 48, 10, 20, 20, 8, Normalize=True)

        # self.conv1 = OperatorBlock_3D(factor * self.width, 2 * factor * self.width, 32, 32, 10, 14, 14, 8)

        # self.conv2 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 16, 16, 12,  10,10,6)
        #
        # self.conv3 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 16, 16, 12,  10,10,6, Normalize = True)

        # self.conv6 = OperatorBlock_3D(2*factor*self.width, 2*factor*self.width, 32, 32, 18,  10,10,8)

        self.conv7 = OperatorBlock_3D(1 * factor * self.width, 1 * factor * self.width, 48, 48, 20, 14, 14, 8, Normalize=True)

        self.conv8 = OperatorBlock_3D(2 * factor * self.width, self.width, 64, 64, 20, 20, 20, 8)  # will be reshaped

        self.fc1 = nn.Linear(2 * self.width, 4 * self.width)
        self.fc2 = nn.Linear(4 * self.width, 3)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        # print('x_grid', x.shape)
        grid = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x.float())
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)

        x_fc0 = F.gelu(x_fc0)

        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        # print('x_fc0', x_fc0.shape)

        self.padding = int(self.padding * 0.1 * x_fc0.shape[-1])
        if self.pad_both:
            x_fc0 = F.pad(x_fc0, [self.padding, self.padding, 0, 0, 0, 0], mode='constant')
        else:
            x_fc0 = F.pad(x_fc0, [0, self.padding, 0, 0, 0, 0], mode='constant')

        D1, D2, D3 = x_fc0.shape[-3], x_fc0.shape[-2], x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0, int(3 * D1 / 4), int(3 * D2 / 4), D3)
        # print('x_c0', x_c0.shape)
        # x_c1 = self.conv1(x_c0, D1 // 2, D2 // 2, D3)
        # print('x_c1', x_c1.shape)
        # x_c2 = self.conv2(x_c1, D1//4, D2//4, int(D3*1.2))
        # # # print('x_c2', x_c2.shape)
        # #
        # x_c3 = self.conv3(x_c2, D1//4, D2//4, int(D3*1.2))
        # print('x_c3', x_c3.shape)

        # x_c6 = self.conv6(x_c1,D1//2, D2//2, int(D3*1.8))
        # # # print('x_c6', x_c6.shape)
        # x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c0, size = (x_c6.shape[2], x_c6.shape[3], x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
        # # # print('x_c6', x_c6.shape)
        #

        x_c7 = self.conv7(x_c0, int(3 * D1 / 4), int(3 * D2 / 4), int(2.0 * D3))
        # print('x_c7', x_c7.shape)
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size=(x_c7.shape[2], x_c7.shape[3], x_c7.shape[4]),
                                                          mode='trilinear', align_corners=True)], dim=1)
        # print('x_c7', x_c7.shape)
        #
        x_c8 = self.conv8(x_c7, D1, D2, D3)
        # print('x_c8', x_c8.shape)

        x_c8 = torch.cat([x_c8, torch.nn.functional.interpolate(x_fc0, size=(x_c8.shape[2], x_c8.shape[3], x_c8.shape[4]),
                                                          mode='trilinear', align_corners=True)], dim=1)
        # print('x_c8', x_c8.shape)

        if self.padding != 0:
            if self.pad_both:
                x_c8 = x_c8[..., 2 * self.padding:-2 * self.padding]
            else:
                x_c8 = x_c8[..., :-2 * self.padding]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)
        # print('x_c8', x_c8.shape)

        x_fc1 = self.fc1(x_c8)

        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        # print('x_out',x_out.shape)

        return x_out

    def get_grid(self, shape):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)