"""Optimizes q-values predicted by the network according to feedback."""
from __future__ import annotations

import numpy as np
import torch

from environment.action import Action
from learning import util
from learning.optimizers import ips
from learning.optimizers.supervised_position_prediction import voxel_prediction_entropy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.training_util_configs import OptimizerConfig
    from inference.predicted_voxel import VoxelPredictions
    from learning.batching.step_batch import StepBatch
    from torch import nn
    from typing import Dict, List, Tuple


def q_value_loss(batch: StepBatch,
                 predictions: VoxelPredictions,
                 clip_max_ips: bool,
                 voxel_coeff: float,
                 use_ips: bool,
                 always_use_ips: bool,
                 reduce: bool = True) -> Tuple[torch.Tensor, List[float]]:
    if use_ips:
        ips_coeffs: List[float] = ips.get_ips(batch, predictions, clip_max_ips,
                                              always_use_ips)
    else:
        ips_coeffs: List[float] = [1. for _ in range(batch.get_batch_size())]

    losses: List[torch.Tensor] = list()
    batch_size: int = batch.get_batch_size()

    for i in range(batch_size):
        if batch.feedbacks.sampled_configurations[i] == Action.STOP:
            item_to_optimize: torch.Tensor = torch.log(
                predictions.stop_probabilities[i])
        else:
            item_to_optimize: torch.Tensor = voxel_coeff * predictions.get_q_value(
                i)

        # Minimize the negative Q-value
        losses.append(-ips_coeffs[i] * item_to_optimize *
                      batch.feedbacks.feedbacks[i] *
                      batch.feedbacks.weights[i])

    if reduce:
        return torch.mean(torch.stack(losses)), ips_coeffs
    return torch.stack(losses), ips_coeffs


class QValueOptimizer:
    def __init__(self, module: nn.Module, config: OptimizerConfig,
                 use_ips: bool, clip_max_ips: bool, voxel_loss_coeff: float,
                 always_use_ips: bool, entropy_coefficient: float):
        self._optimizer: torch.optim.Adam = torch.optim.Adam(
            module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2_coefficient)

        self._named_parameters = module.named_parameters()

        self._use_ips: bool = use_ips
        self._clip_max_ips: bool = clip_max_ips
        self._voxel_loss_coeff: float = voxel_loss_coeff
        self._always_use_ips: bool = always_use_ips
        self._entropy_coefficient: float = entropy_coefficient

    def _apply_loss(self, loss: torch.Tensor):
        loss.backward()
        self._optimizer.step()
        util.check_for_nan_grads(self._named_parameters)
        self._optimizer.zero_grad()
        util.check_for_nan_params(self._named_parameters)

    def compute_and_apply_loss(
        self, batch: StepBatch, predictions: VoxelPredictions
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        loss, ips_coeff = q_value_loss(batch,
                                       predictions,
                                       self._clip_max_ips,
                                       self._voxel_loss_coeff,
                                       self._use_ips,
                                       self._always_use_ips,
                                       reduce=False)

        entropy = voxel_prediction_entropy(predictions, reduce=False)

        total_loss = loss
        if self._entropy_coefficient:
            total_loss -= entropy * self._entropy_coefficient * 3751

        mean_loss = torch.mean(total_loss)

        self._apply_loss(mean_loss)

        ips_np: np.ndarray = np.array(ips_coeff)
        loss_dict = {
            'loss/batch': torch.mean(loss).item(),
            'loss/total': mean_loss.item(),
            'loss/max_ips': np.max(ips_np),
            'loss/min_ips': np.min(ips_np),
            'loss/mean_ips': np.mean(ips_np),
            'loss/entropy': torch.mean(entropy).item()
        }

        return loss_dict, total_loss
