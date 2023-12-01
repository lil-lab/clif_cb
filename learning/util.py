"""Various training utilities."""
from __future__ import annotations
import math

import torch
from torch import nn


def check_for_nan_params(named_parameters, print_stats: bool = False):
    has_nan_param = False
    for param_name, param in named_parameters:
        if param.grad is not None:
            if print_stats:
                print('%s\t%s' % (param_name, torch.mean(param).item()))
            if math.isnan(torch.mean(param)):
                has_nan_param = True
                print('NaN param %s with grad %s/%s/%s' %
                      (param_name, torch.min(param.grad), torch.mean(
                          param.grad), torch.max(param.grad)))
    if has_nan_param:
        raise ValueError('Has NaN param(s)!')

    if print_stats:
        exit()


def check_for_nan_grads(named_parameters, print_stats: bool = False):
    has_nan_grad = False
    for param_name, param in named_parameters:
        if param.grad is not None:
            if print_stats:
                print('%s\t%s\t%s' % (param_name, torch.mean(
                    param.grad).item(), torch.mean(param).item()))
            if math.isnan(torch.mean(param.grad)):
                has_nan_grad = True
                print('NaN grad on param %s' % param_name)
    if has_nan_grad:
        raise ValueError('Has NaN grad(s)!')

    if print_stats:
        exit()
