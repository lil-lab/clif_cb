"""Optimizes predictions of gold-standard actions."""
from __future__ import annotations

import torch

from torch import nn

from config.training_configs import OptimizerConfig
from inference.predicted_action_distribution import ActionPredictions, PREDICTABLE_ACTIONS
from learning import util
from learning.batching.step_batch import StepBatch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, List


def target_action_probabilities(
        batch: StepBatch, predictions: ActionPredictions) -> torch.Tensor:
    probs: List[torch.Tensor] = list()

    for b, i in zip(predictions.action_probabilities,
                    batch.target.action_type_labels):
        probs.append(b[i].unsqueeze(0))

    return torch.cat(probs)


def action_prediction_loss(batch: StepBatch,
                           predictions: ActionPredictions,
                           average: bool = True) -> torch.Tensor:
    target_probabilities: torch.Tensor = target_action_probabilities(
        batch, predictions)

    neg_log_loss: torch.Tensor = -torch.log(target_probabilities)

    if average:
        return torch.mean(neg_log_loss)
    return neg_log_loss


def action_prediction_entropy(predictions: ActionPredictions,
                              reduce: bool = True) -> torch.Tensor:
    entropy: torch.Tensor = torch.sum(
        -predictions.action_probabilities *
        torch.log(predictions.action_probabilities + 0.0000001),
        dim=1)

    if reduce:
        return torch.mean(entropy)
    return entropy


class SupervisedActionPredictionOptimizer:
    def __init__(self, module: nn.Module, config: OptimizerConfig):
        self._optimizer: torch.optim.Adam = torch.optim.Adam(
            module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2_coefficient)
        self._named_parameters = module.named_parameters()

    def _apply_loss(self, loss: torch.Tensor):
        loss.backward()
        self._optimizer.step()
        util.check_for_nan_grads(self._named_parameters)
        self._optimizer.zero_grad()
        util.check_for_nan_params(self._named_parameters)

    def compute_and_apply_loss(
            self, batch: StepBatch,
            predictions: ActionPredictions) -> Dict[str, float]:
        loss_dict: Dict[str, float] = dict()
        action_loss: torch.Tensor = action_prediction_loss(batch, predictions)
        loss_dict['loss/action'] = action_loss.item()

        self._apply_loss(action_loss)

        loss_dict['loss/batch'] = action_loss.item()
        return loss_dict, 0
