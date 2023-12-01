"""Various torch utilities"""

from __future__ import annotations

from enum import Enum
import torch

if torch.cuda.device_count() >= 1:
    DEVICE: torch.device = torch.device('cuda')
else:
    DEVICE: torch.device = torch.device('cpu')

NUM_GPUS = torch.cuda.device_count()


class TensorType(Enum):
    TORCH: str = 'TORCH'
    NUMPY: str = 'NUMPY'
