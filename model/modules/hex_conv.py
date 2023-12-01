"""Utilities for hex convolutions.

Based on HexaConv (Hoogeboom et al. 2018, https://arxiv.org/pdf/1803.02108.pdf)
"""
import torch
import math

from torch import nn
from torch.nn import functional
from typing import List, Tuple

from environment.position import EDGE_WIDTH


def _get_hex_conv_mask(kernel_size: int) -> torch.Tensor:
    # This is a mask on the filter which zeros out the corners of the convolution.
    # See https://arxiv.org/pdf/1803.02108.pdf, Figure 4a.
    mask = torch.ones((kernel_size, kernel_size))
    cutoff_amount = (kernel_size - 1) // 2
    for i in range(cutoff_amount):
        for j in range(cutoff_amount - i):
            mask[i][j] = 0.
            mask[kernel_size - 1 - i][kernel_size - 1 - j] = 0.
    return mask


class HexConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True):
        super(HexConv, self).__init__()

        if kernel_size % 2 != 1:
            raise ValueError('Kernel size must be odd for Hex Conv: %s' %
                             kernel_size)

        self.register_parameter(name='_weight',
                                param=torch.nn.Parameter(torch.zeros(
                                    out_channels, in_channels, kernel_size,
                                    kernel_size),
                                                         requires_grad=True))
        nn.init.kaiming_uniform_(self._weight, a=math.sqrt(5))
        if bias:
            self.register_parameter(name='_bias',
                                    param=torch.nn.Parameter(
                                        torch.zeros(out_channels),
                                        requires_grad=True))

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self._bias, -bound, bound)
        else:
            self._bias = None

        self._stride = stride
        self._kernel_size = kernel_size
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._padding = padding
        self._mask = nn.Parameter(_get_hex_conv_mask(kernel_size),
                                  requires_grad=False)

    def extra_repr(self):
        return 'input_channels={}, output_channels={}, kernel_size={}, stride={}, padding={}, bias={}'.format(
            self._in_channels, self._out_channels, self._kernel_size,
            self._stride, self._padding, self._bias is not None)

    def forward(self, input_tensor: torch.Tensor):
        """Input must be in axial coordinates. """
        masked_filter = self._weight * self._mask.detach()
        return functional.conv2d(input_tensor,
                                 masked_filter,
                                 bias=self._bias,
                                 stride=self._stride,
                                 padding=self._padding)


class HexConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True):
        super(HexConvTranspose2d, self).__init__()
        if kernel_size % 2 != 1:
            raise ValueError('Kernel size must be odd for Hex Conv: %s' %
                             kernel_size)

        self.register_parameter(name='_weight',
                                param=torch.nn.Parameter(torch.zeros(
                                    in_channels, out_channels, kernel_size,
                                    kernel_size),
                                                         requires_grad=True))
        nn.init.kaiming_uniform_(self._weight, a=math.sqrt(5))
        if bias:
            self.register_parameter(name='_bias',
                                    param=torch.nn.Parameter(
                                        torch.zeros(out_channels),
                                        requires_grad=True))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self._bias, -bound, bound)
        else:
            self._bias = None
        self._stride = stride
        self._kernel_size = kernel_size
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._padding = padding
        self._mask = nn.Parameter(_get_hex_conv_mask(kernel_size),
                                  requires_grad=False)

    def extra_repr(self):
        return 'input_channels={}, output_channels={}, kernel_size={}, stride={}, padding={}, bias={}'.format(
            self._in_channels, self._out_channels, self._kernel_size,
            self._stride, self._padding, self._bias is not None)

    def _output_padding(self, input: torch.Tensor, output_size: List[int],
                        stride: List[int], padding: List[int],
                        kernel_size: List[int]) -> List[int]:
        """Taken from HexConvTranspose2d."""
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})".format(
                    k, k + 2, len(output_size)))

        min_sizes = torch.jit.annotate(List[int], [])
        max_sizes = torch.jit.annotate(List[int], [])
        for d in range(k):
            dim_size = ((input.size(d + 2) - 1) * stride[d] - 2 * padding[d] +
                        kernel_size[d])
            min_sizes.append(dim_size)
            max_sizes.append(min_sizes[d] + stride[d] - 1)

        for i in range(len(output_size)):
            size = output_size[i]
            min_size = min_sizes[i]
            max_size = max_sizes[i]
            if size < min_size or size > max_size:
                raise ValueError(
                    ("requested an output size of {}, but valid sizes range "
                     "from {} to {} (for an input of {})").format(
                         output_size, min_sizes, max_sizes,
                         input.size()[2:]))

        res = torch.jit.annotate(List[int], [])
        for d in range(k):
            res.append(output_size[d] - min_sizes[d])

        return res

    def forward(self, input_tensor: torch.Tensor, output_size: int):
        masked_filter = self._weight * self._mask.detach()

        output_padding = self._output_padding(
            input_tensor, output_size, [self._stride, self._stride],
            [self._padding, self._padding],
            [self._kernel_size, self._kernel_size])

        return functional.conv_transpose2d(input_tensor,
                                           masked_filter,
                                           bias=self._bias,
                                           stride=self._stride,
                                           padding=self._padding,
                                           output_padding=output_padding)


class HexCrop(nn.Module):
    def __init__(self, crop_size: int, env_size: int = EDGE_WIDTH):
        """Crops an N x N region around the center of a tensor, where N = crop size."""
        super(HexCrop, self).__init__()
        if crop_size % 2 != 1:
            raise ValueError('Crop size must be odd for Hex Crop: %s' %
                             crop_size)
        self._crop_mask = nn.Parameter(_get_hex_conv_mask(crop_size),
                                       requires_grad=False)
        self._crop_size = crop_size
        self._environment_size = env_size

    def forward(self, input_tensor: torch.Tensor,
                center_positions: torch.Tensor,
                mask: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crops a square portion around the center of the input tensor, masking out values not in the neighborhood
        of the crop value. Input must be in axial coordinates."""
        batch_size, num_channels, height, width = input_tensor.size()

        crop_center = (self._crop_size - 1) // 2  # Need to pad the input
        padded_state = torch.nn.functional.pad(
            input_tensor, [crop_center, crop_center, crop_center, crop_center
                           ])  # Convert the position to axial coordinates
        r_pos = center_positions[:, 0]
        q_pos = center_positions[:, 1]
        v_pos = q_pos

        add_u = (self._environment_size - 1) // 2
        u_pos = r_pos - v_pos // 2 + add_u
        us = [
            u_pos + (slack - crop_center) for slack in range(self._crop_size)
        ]
        us = torch.stack(us, 0).unsqueeze(1)
        us = us.repeat(1, self._crop_size, 1).long()
        us += crop_center  # Because of padding
        vs = [
            v_pos + (slack - crop_center) for slack in range(self._crop_size)
        ]
        vs = torch.stack(vs, 0).unsqueeze(0)
        vs = vs.repeat(self._crop_size, 1, 1).long()
        vs += crop_center  # Because of padding
        batch_indices = torch.tensor([i for i in range(batch_size)
                                      ]).long().unsqueeze(0).unsqueeze(0)
        batch_indices = batch_indices.repeat(self._crop_size, self._crop_size,
                                             1)
        cropped_square = padded_state[batch_indices, :, us, vs]
        cropped_square = cropped_square.permute(2, 3, 0, 1)

        # Mask
        if mask:
            cropped_square *= self._crop_mask.detach()
        return cropped_square, self._crop_mask.detach()
