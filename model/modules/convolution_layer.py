"""
Attribution: Code is adapted from the following repositories...

    LingUNet: Blukis et al. 2018 (CoRL), Misra et al. 2018 (EMNLP)
    Code: Chen et al. 2019 (CVPR), https://github.com/clic-lab/street-view-navigation
          Blukis et al. 2018 (CoRL); and ongoing work by Valts Blukis.

    Official drone sim code:
        https://github.com/clic-lab/drone-sim/blob/release/learning/modules/unet/unet_5_contextual_bneck3.py

"""
from __future__ import annotations

import torch
from torch import nn as nn

from model.modules import hex_conv


class ConvolutionLayer(torch.nn.Module):
    def __init__(self,
                 depth: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 use_hex_conv: bool = False):
        super(ConvolutionLayer, self).__init__()
        self._conv_layers = nn.ModuleList([])
        for i in range(depth):
            s: int = stride if i == 0 else 1
            o: int = out_channels if i == depth - 1 else in_channels

            if use_hex_conv:
                layer: nn.Module = hex_conv.HexConv(in_channels,
                                                    o,
                                                    kernel_size,
                                                    stride=s,
                                                    padding=padding,
                                                    bias=True)
                nn.init.kaiming_uniform_(layer._weight)
            else:
                layer: nn.Module = nn.Conv2d(in_channels,
                                             o,
                                             kernel_size,
                                             stride=s,
                                             padding=padding)
                nn.init.kaiming_uniform_(layer.weight)

            self._conv_layers.append(layer)

            # Add a leaky relu between layers (but not on the output)
            if i < depth - 1:
                self._conv_layers.append(nn.LeakyReLU())

    def forward(self, img):
        x = img
        for l in self._conv_layers:
            x = l(x)
        return x


class ResBlockStrided(torch.nn.Module):
    def __init__(self,
                 c_in: int,
                 stride: int,
                 down_padding: int,
                 groups: int = 1):
        super(ResBlockStrided, self).__init__()
        self._c_in = c_in
        self._conv1 = nn.Conv2d(c_in,
                                c_in,
                                3,
                                padding=down_padding,
                                groups=groups)
        self._conv2 = nn.Conv2d(c_in,
                                c_in,
                                3,
                                stride=stride,
                                padding=1,
                                groups=groups)

        torch.nn.init.kaiming_uniform_(self._conv1.weight)
        torch.nn.init.kaiming_uniform_(self._conv2.weight)

        self._conv1.bias.data.fill_(0)
        self._conv2.bias.data.fill_(0)

        self._act1 = nn.LeakyReLU()
        self._act2 = nn.LeakyReLU()

        self._norm1 = nn.InstanceNorm2d(c_in)
        self._norm2 = nn.InstanceNorm2d(c_in)

        self._average_pool = nn.AvgPool2d(3,
                                          stride=stride,
                                          padding=down_padding)

    def forward(self, images):
        x = self._act1(self._conv1(self._norm1(images)))
        x_out = self._act2(self._conv2(self._norm2(x)))
        x_in = self._average_pool(images)
        return x_in + x_out


class ResNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth: int,
                 down_pad: bool = True):
        super(ResNetBlock, self).__init__()

        self._downscale_factor = out_channels

        down_padding = 0
        if down_pad:
            down_padding = 1

        # inchannels, outchannels, kernel size
        self._conv = nn.Conv2d(in_channels,
                               out_channels,
                               3,
                               stride=3,
                               padding=down_padding)
        torch.nn.init.kaiming_uniform_(self._conv.weight)
        self._conv.bias.data.fill_(0)

        self._blocks = nn.ModuleList([])
        for i in range(depth):
            block = ResBlockStrided(out_channels,
                                    stride=3,
                                    down_padding=down_padding)
            self._blocks.append(block)

    def get_downscale_factor(self):
        return self._downscale_factor

    def forward(self, input_tensor: torch.Tensor):
        x = self._conv(input_tensor)

        for block in self._blocks:
            x = block(x)

        return x
