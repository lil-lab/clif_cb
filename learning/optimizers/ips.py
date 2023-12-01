"""Utilities for inverse propensity scoring (IPS)."""
from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Union

from environment.action import Action
from environment.player import Player
from environment.position import Position
from environment.rotation import Rotation, ROTATIONS
from inference.predicted_action_distribution import ActionPredictions, PREDICTABLE_ACTIONS

if TYPE_CHECKING:
    from inference.predicted_voxel import VoxelPredictions
    from learning.batching.step_batch import StepBatch
    from typing import List, Union


def get_ips(batch: StepBatch, predictions: Union[VoxelPredictions,
                                                 ActionPredictions],
            clip_max_ips: bool, always_use_ips: bool,
            divide_by_original_probability: bool) -> List[float]:
    ips_coeffs: List[float] = list()
    batch_size: int = batch.get_batch_size()

    for i in range(batch_size):
        action_prob: torch.Tensor = get_current_policy_probability(
            batch, predictions, i)

        if batch.feedbacks.use_ips[i] or always_use_ips:
            sample: Union[Action,
                          Player] = batch.feedbacks.sampled_configurations[i]

            if batch.feedbacks.use_ips[i] and divide_by_original_probability:
                if isinstance(predictions, ActionPredictions):
                    original_prob: torch.Tensor = batch.feedbacks.original_distributions.action_probabilities[
                        i][PREDICTABLE_ACTIONS.index(sample)]
                else:
                    if sample == Action.STOP:
                        original_prob: torch.Tensor = batch.feedbacks.original_distributions.stop_probabilities[
                            i]
                    else:
                        original_prob: torch.Tensor = batch.feedbacks.original_distributions.voxel_probabilities[
                            i][ROTATIONS.index(sample.rotation)][
                                sample.position.x][sample.position.y]
            else:
                original_prob = 1.

            ips_coeff: float = (action_prob / original_prob).item()
            if clip_max_ips:
                # Cuts off at max value of 1. IPS can be smaller, but not larger than 1.
                ips_coeff = min(ips_coeff, 1)
        else:
            # This happens when there are no original probabilities; i.e., when the examples are "original" examples
            # without having come from an actual game.
            ips_coeff: float = 1.0

        ips_coeffs.append(ips_coeff)
    return ips_coeffs


def get_current_policy_probability(batch: StepBatch,
                                   predictions: Union[VoxelPredictions,
                                                      ActionPredictions],
                                   batch_idx: int) -> torch.Tensor:
    sample: Union[Action,
                  Player] = batch.feedbacks.sampled_configurations[batch_idx]

    if isinstance(predictions, ActionPredictions):
        return predictions.action_probabilities[batch_idx][
            PREDICTABLE_ACTIONS.index(sample)]
    else:
        if sample == Action.STOP:
            return predictions.stop_probabilities[batch_idx]
        else:
            pos: Position = sample.position
            rot: Rotation = sample.rotation

            return predictions.voxel_probabilities[batch_idx][ROTATIONS.index(
                rot)][pos.x][pos.y]
