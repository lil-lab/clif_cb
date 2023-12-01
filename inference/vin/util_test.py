# python -m inference.vin.util_test
import torch
import numpy as np
import torch.nn.functional as F
from util import torch_util
from inference.vin.vin_model import _get_cerealbar_axial_2d_kernels


def test_conv3d_dimensionality():
    """
    ref: https://pytorch.org/docs/master/generated/torch.nn.functional.conv3d.html
    """
    filters = torch.randn(4, 6, 1, 3, 3)
    inputs = torch.randn(1, 6, 1, 25, 25)
    outputs = F.conv3d(inputs, filters, padding=(0, 1, 1))
    print("input shape {}, kernel shape {} and output shape {}".format(
        inputs.shape, filters.shape, outputs.shape))


def test_conv2d_group():
    """
    ref: https://discuss.pytorch.org/t/convolution-operation-without-the-final-summation/56466/3
    ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    """
    conv = nn.Conv2d(24, 24, 3, groups=24)
    inputs = torch.randn(10, 24, 25, 25)
    outputs = conv(inputs)
    print("input shape {}, kernel shape {}, and output shape {}".format(inputs.shape, conv.weight.shape, outputs.shape))
    """
    filters = torch.randn(24, 1, 3, 3)
    inputs = torch.randn(16, 24, 25, 25)
    outputs = F.conv2d(inputs, filters, padding=(1, 1), groups=24)
    print("input shape {}, kernel shape {}, and output shape {}".format(
        inputs.shape, filters.shape, outputs.shape))


def test_conv2d_transpose():
    kernels = torch.zeros(2, 1, 3, 3).to(torch_util.DEVICE)
    kernels[..., 1, 2] = 1.
    inputs = torch.zeros((1, 2, 3, 3)).to(torch_util.DEVICE)
    inputs[..., 1, 2] = 1.
    print(inputs)
    outputs = F.conv2d(inputs, kernels, padding=(1, 1), groups=2)
    print(outputs)


def mul_braodacast():
    inputs = torch.randn(16, 24, 25, 25)
    mask = torch.randn(16, 1, 25, 25)
    outputs = inputs * mask
    print("input shape {}, mask shape {}, and output shape {}".format(
        inputs.shape, mask.shape, outputs.shape))


def get_cerealbar_axial_2d_kernels_test():
    transition_kernels = _get_cerealbar_axial_2d_kernels()
    inputs = torch.zeros((1, 24, 37, 25)).to(torch_util.DEVICE)
    inputs[:, 11, 12, 6] = 1
    outputs = F.conv2d(inputs, transition_kernels, padding=(1, 1), groups=24)
    print(np.argwhere(outputs == 1.))


if __name__ == "__main__":
    #test_conv2d_transpose()
    #dot_product_braodacast()
    get_cerealbar_axial_2d_kernels_test()
