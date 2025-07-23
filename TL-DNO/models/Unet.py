from collections import OrderedDict
import torch.nn.functional as F


import torch
import torch.nn as nn
from utils import grid


class UNet1d(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet1d, self).__init__()

        features = init_features
        self.encoder1 = UNet1d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet1d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


class UNet2d(nn.Module):

    def __init__(self, in_channels, out_channels, init_features, grid_type):
        super(UNet2d, self).__init__()

        self.grid = grid(twoD=True, grid_type=grid_type)
        features = init_features
        self.encoder1 = UNet2d._block(in_channels + self.grid.grid_dim, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = self.grid(x)
        x = x.permute(0, 3, 1, 2)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        # if dec4.shape[-2] == 100 and dec4.shape[-2] == 54:
        dec4 = F.pad(dec4, [0, 1])
        dec4 = dec4.permute(0, 1, 3, 2)
        dec4 = F.pad(dec4, [0, 1])
        dec4 = dec4.permute(0, 1, 3, 2)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = dec2.permute(0, 1, 3, 2)
        dec2 = F.pad(dec2, [0, 1])
        dec2 = dec2.permute(0, 1, 3, 2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = F.pad(dec1, [0, 1])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)
        dec1 = dec1.permute(0, 2, 3, 1)
        return dec1.unsqueeze(-2)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


class UNet3d(nn.Module):

    def __init__(self, in_channels, out_channels, init_features, grid_type, time, time_pad=False):
        super(UNet3d, self).__init__()

        self.time = time
        self.time_pad = time_pad
        self.padding = 4  # pad the domain if input is non-periodic
        features = init_features
        self.grid = grid(twoD=False, grid_type=grid_type)
        self.encoder1 = UNet3d._block(in_channels + self.grid.grid_dim, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        # grid = self.get_grid(x.shape).to(x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.grid(x)
        # x = F.pad(x, [0, self.padding])
        x = x.permute(0, 4, 1, 2, 3)
        if self.time and self.time_pad:
            x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        if x.shape[-1] == 10:
            x = F.pad(x, [0, 6])

        enc1 = self.encoder1(x.float())
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        # print('dec4', dec4.shape)
        # print('enc4', enc4.shape)
        # dec4 = dec4.permute(0, 4, 1, 2, 3)
        dec4 = F.pad(dec4, [0, 1])
        # # dec4 = dec4.permute(0, 1, 2, 4, 3)
        # # dec4 = F.pad(dec4, [0, 1])
        # dec4 = dec4.permute(0, 2, 3, 4, 1)
        # print('dec4', dec4.shape)
        # print('enc4', enc4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        # print('dec3', dec3.shape)
        # print('enc3', enc3.shape)
        # if dec3.shape[-1] == 4 and enc3.shape[-1] == 5:
        #     dec3 = F.pad(dec3, [0, 1])
        dec3 = dec3.permute(0, 4, 1, 2, 3)
        dec3 = F.pad(dec3, [0, 1])
        dec3 = dec3.permute(0, 1, 2, 4, 3)
        dec3 = F.pad(dec3, [0, 1])
        dec3 = dec3.permute(0, 2, 4, 3, 1)
        # print('dec3', dec3.shape)
        # print('enc3', enc3.shape)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        # print('enc2', enc2.shape)
        # print('dec2', dec2.shape)
        # dec2 = F.pad(dec2, [0, 1])
        dec2 = dec2.permute(0, 4, 1, 2, 3)
        dec2 = F.pad(dec2, [0, 1])
        dec2 = dec2.permute(0, 1, 2, 4, 3)
        dec2 = F.pad(dec2, [0, 1])
        dec2 = dec2.permute(0, 2, 4, 3, 1)
        # dec2 = dec2.permute(0, 4, 1, 2, 3)
        # dec2 = F.pad(dec2, [0, 1])
        # dec2 = dec2.permute(0, 1, 2, 4, 3)
        # dec2 = F.pad(dec2, [0, 1])
        # dec2 = dec2.permute(0, 2, 4, 3, 1)
        # print('enc2', enc2.shape)
        # print('dec2', dec2.shape)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        # print('enc1', enc1.shape)
        # print('dec1', dec1.shape)
        # # dec1 = F.pad(dec1, [0, 1])
        # dec1 = dec1.permute(0, 4, 1, 3, 2)
        # dec1 = F.pad(dec1, [0, 1])
        dec1 = dec1.permute(0, 1, 2, 4, 3)
        dec1 = F.pad(dec1, [0, 1])
        dec1 = dec1.permute(0, 1, 2, 4, 3)
        # dec1 = dec1.permute(0, 2, 4, 3, 1)
        # dec1 = F.pad(dec1, [0, 1])
        # print('enc1', enc1.shape)
        # print('dec1', dec1.shape)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        if self.time and self.time_pad:
            dec1 = dec1[..., :-self.padding]
        if dec1.shape[-1] == 16:
            dec1 = dec1[..., :-6]
        dec1 = self.conv(dec1)
        dec1 = dec1.permute(0, 2, 3, 4, 1)
        if not self.time:  # add a time axis
            dec1 = dec1.unsqueeze(-2)
        return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )

    def get_grid(self, shape):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)