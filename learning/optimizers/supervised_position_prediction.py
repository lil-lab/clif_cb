"""
Optimizer for supervised position prediction, where there is a target set of plausible correct voxels (or stop 
action).
"""
from __future__ import annotations

import torch
from torch import nn

from environment.action import Action
from learning import util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.training_util_configs import OptimizerConfig
    from inference.predicted_voxel import VoxelPredictions
    from learning.batching.step_batch import StepBatch
    from typing import Dict, List, Tuple


def voxel_prediction_probabilities(
        batch: StepBatch, predictions: VoxelPredictions) -> torch.Tensor:
    batch_size: int = batch.get_batch_size()

    # This leaves only voxels which are plausible predictions.
    masked_voxel_predictions: torch.Tensor = predictions.voxel_probabilities * batch.target.plausible_target_voxels.float(
    )

    flattened_voxels: torch.Tensor = masked_voxel_predictions.view(
        (batch_size, -1))

    # EM-style loss.
    # This gets max_g p(g | x, e, \theta) where g \in G, and G is the set of plausible targets.
    # Get rid of the argmax (second element); grab only the maxes (first element)
    # For examples without any plausible next voxels (i.e., STOP examples), the 0th element will be returned (
    # with value 0).
    max_masked_prediction: torch.Tensor = torch.max(flattened_voxels, dim=1)[0]

    # B x 2, where [b,0] is the max voxel prediction, and [b, 1] is the stop probability.
    all_probs: torch.Tensor = torch.cat(
        (max_masked_prediction.unsqueeze(1),
         predictions.stop_probabilities.unsqueeze(1)),
        dim=1)

    if predictions.copy_probabilities is not None:
        # Also include copy probabilities.
        all_probs: torch.Tensor = torch.cat(
            (all_probs, predictions.copy_probabilities.unsqueeze(1)), dim=1)

    # B: grabs either the 0th or 1st index according to the action type tensor, which is 0 if STOP is not the correct
    # action and 1 if it is. The index may also be 2 if the correct action is to copy.
    probs: List[torch.Tensor] = list()

    for b, i in zip(all_probs, batch.target.action_type_labels):
        probs.append(torch.index_select(b, 0, i).unsqueeze(0))

    return torch.cat(probs)


def voxel_prediction_loss(batch: StepBatch,
                          predictions: VoxelPredictions,
                          use_ips: bool,
                          average: bool = True) -> torch.Tensor:
    relevant_probs: torch.Tensor = voxel_prediction_probabilities(
        batch, predictions)

    neg_log_loss: torch.Tensor = -torch.log(relevant_probs)

    if use_ips:
        # Multiply by the probability according to the current model.
        neg_log_loss *= relevant_probs.detach()

    if average:
        return torch.mean(neg_log_loss)
    return neg_log_loss


def target_position_probabilities(
        batch: StepBatch, predictions: VoxelPredictions) -> List[float]:
    probs: List[float] = list()

    for i in range(batch.get_batch_size()):
        if batch.original_examples[i].target_action == Action.STOP:
            probs.append(predictions.stop_probabilities[i].item())
        else:
            target_voxel = batch.original_examples[i].final_target

            # Sum over rotations
            probs.append(
                torch.sum(predictions.voxel_probabilities[
                    i, :, target_voxel.position.x,
                    target_voxel.position.y]).item())
            raise NotImplementedError('Check the implementation of this')

    return probs


def voxel_prediction_entropy(predictions: VoxelPredictions,
                             reduce: bool = True) -> torch.Tensor:
    batch_size: int = predictions.get_batch_size()
    flattened_voxels: torch.Tensor = predictions.voxel_probabilities.view(
        batch_size, -1)
    entire_dist: torch.Tensor = torch.cat(
        (flattened_voxels, predictions.stop_probabilities.unsqueeze(1)), dim=1)

    entropy: torch.Tensor = -entire_dist * torch.log(entire_dist + 0.0000001)

    if reduce:
        return torch.mean(entropy)
    return torch.mean(entropy, dim=1)


def same_target_consistency_loss(
        batch: StepBatch, predictions: VoxelPredictions) -> torch.Tensor:
    # Sort examples into targets
    grouped_target_indices: Dict[str, List[int]] = dict()
    for i, step in enumerate(batch.original_examples):
        target_id: str = step.unique_target_id
        if target_id not in grouped_target_indices:
            grouped_target_indices[target_id] = list()
        grouped_target_indices[target_id].append(i)

    # For target groups with more than one example, compute all pairwise KL-divergences
    kl_empirical: List[torch.Tensor] = list()
    kl_target: List[torch.Tensor] = list()

    num_comparisons: int = 0
    for target_id, steps in grouped_target_indices.items():
        if len(steps) > 1:
            # Can compute KL-differences.
            for step_1_idx in range(len(steps)):
                for step_2_idx in range(len(steps)):
                    if step_1_idx == step_2_idx:
                        continue

                    # These should both be 1 x (625 + 1)
                    step_1_probs: torch.Tensor = predictions.get_flat_distribution(
                        steps[step_1_idx])
                    step_2_probs: torch.Tensor = predictions.get_flat_distribution(
                        steps[step_2_idx])

                    # Compute KL of this pair, using step 1 as empirical and step 2 as target
                    kl_empirical.append(step_1_probs)
                    kl_target.append(step_2_probs)

                    num_comparisons += 1

    if num_comparisons:
        # Actually compute KL
        return nn.KLDivLoss(reduction='batchmean')(torch.log(
            torch.cat(kl_empirical, dim=0)), torch.cat(kl_target, dim=0))

    return torch.zeros(1)


class SupervisedPositionPredictionOptimizer:
    def __init__(self, module: nn.Module, config: OptimizerConfig,
                 same_target_consistency_coefficient: float,
                 entropy_coefficient: float, use_ips: bool):
        self._optimizer: torch.optim.Adam = torch.optim.Adam(
            module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2_coefficient)
        self._same_target_consistency_coefficient: float = same_target_consistency_coefficient
        self._entropy_coefficient: float = entropy_coefficient

        self._named_parameters = module.named_parameters()

        self._use_ips: bool = use_ips

    def _apply_loss(self, loss: torch.Tensor):
        loss.backward()
        self._optimizer.step()
        util.check_for_nan_grads(self._named_parameters)
        self._optimizer.zero_grad()
        util.check_for_nan_params(self._named_parameters)

    def compute_and_apply_loss(
            self, batch: StepBatch,
            predictions: VoxelPredictions) -> Dict[str, float]:
        loss_dict: Dict[str, float] = dict()
        voxel_loss: torch.Tensor = voxel_prediction_loss(batch,
                                                         predictions,
                                                         use_ips=self._use_ips)
        loss_dict['loss/voxel'] = voxel_loss.item()

        total_loss: torch.Tensor = voxel_loss
        if self._same_target_consistency_coefficient > 0:
            consistency_loss: torch.Tensor = same_target_consistency_loss(
                batch, predictions)
            total_loss += self._same_target_consistency_coefficient * consistency_loss

            loss_dict['loss/consistency'] = consistency_loss.item()

        entropy: torch.Tensor = voxel_prediction_entropy(predictions)
        loss_dict['loss/entropy'] = entropy.item()
        if self._entropy_coefficient:
            total_loss -= self._entropy_coefficient * entropy

        self._apply_loss(total_loss)

        relevant_probs: torch.Tensor = voxel_prediction_probabilities(
            batch, predictions)
        loss_dict['loss/training_probs'] = torch.mean(relevant_probs)

        loss_dict['loss/batch'] = total_loss.item()
        return loss_dict
