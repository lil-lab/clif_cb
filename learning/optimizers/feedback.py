"""Optimizing feedback (bandit learning)."""
from __future__ import annotations

import numpy as np
import torch

from torch import nn

from environment.action import Action
from evaluation.metric import Metric
from inference.predicted_action_distribution import ActionPredictions, PREDICTABLE_ACTIONS
from learning import util
from learning.optimizers import ips
from learning.optimizers.supervised_action_prediction import action_prediction_entropy
from learning.optimizers.supervised_position_prediction import voxel_prediction_entropy

from util.torch_util import DEVICE

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.training_util_configs import OptimizerConfig
    from inference.predicted_voxel import VoxelPredictions
    from learning.batching.step_batch import StepBatch
    from typing import Dict, List, Tuple, Union


def _get_counterfactual_em_probability(batch: StepBatch,
                                       predictions: Union[VoxelPredictions,
                                                          ActionPredictions],
                                       batch_idx: int) -> torch.Tensor:
    if isinstance(predictions, ActionPredictions):
        mask: torch.Tensor = torch.ones(len(PREDICTABLE_ACTIONS)).to(DEVICE)

        # Zero-out the negative-feedback action.
        sample: Action = batch.feedbacks.sampled_configurations[batch_idx]
        mask[PREDICTABLE_ACTIONS.index(sample)] = 0.

        masked_probs = predictions.action_probabilities[batch_idx] * mask
        return torch.max(masked_probs)
    else:
        raise NotImplementedError


def compute_expected_feedback(
        batch: StepBatch, predictions: VoxelPredictions) -> Dict[str, float]:
    batch_size: int = batch.get_batch_size()

    pos_expected_fbs: List[float] = list()
    neg_expected_fbs: List[float] = list()

    for i in range(batch_size):
        # Only considers examples that are from the online feedback set.
        if batch.original_examples[i].is_converted_feedback_example:
            continue

        feedback: float = batch.feedbacks.feedbacks[i].item()
        expected_fb: float = feedback * ips.get_current_policy_probability(
            batch, predictions, i).item()

        if feedback < 0:
            neg_expected_fbs.append(expected_fb)
        elif feedback > 0:
            pos_expected_fbs.append(expected_fb)

    results_dict: Dict[str, float] = dict()
    if pos_expected_fbs + neg_expected_fbs:
        results_dict[
            f'expected_feedback/{Metric.EXPECTED_FEEDBACK}'] = np.mean(
                np.array(pos_expected_fbs + neg_expected_fbs))
    if pos_expected_fbs:
        results_dict[
            f'expected_feedback/{Metric.EXPECTED_FEEDBACK_POS}'] = np.mean(
                np.array(pos_expected_fbs))
    if neg_expected_fbs:
        results_dict[
            f'expected_feedback/{Metric.EXPECTED_FEEDBACK_NEG}'] = np.mean(
                np.array(neg_expected_fbs))

    return results_dict


def feedback_loss(batch: StepBatch,
                  predictions: VoxelPredictions,
                  clip_max_ips: bool,
                  always_use_ips: bool,
                  use_ips: bool,
                  divide_by_original_probability: bool,
                  counterfactual_negative_examples: bool,
                  reduce=True) -> Tuple[torch.Tensor, List[float]]:
    # Very simple method:
    #  - ips coefficients
    #  - no traces
    #  - applied directly to distribution (no backprop through VIN)
    losses: List[torch.Tensor] = list()
    batch_size: int = batch.get_batch_size()

    if use_ips:
        ips_coeffs: List[float] = ips.get_ips(batch, predictions, clip_max_ips,
                                              always_use_ips,
                                              divide_by_original_probability)
    else:
        ips_coeffs: List[float] = [1. for _ in range(batch.get_batch_size())]

    for i in range(batch_size):
        action_prob: torch.Tensor = ips.get_current_policy_probability(
            batch, predictions, i)

        feedback: torch.Tensor = batch.feedbacks.feedbacks[i]
        if counterfactual_negative_examples and batch.feedbacks.feedbacks[
                i] < 0:
            # Get the probability of the most likely action that is *not* the negative-feedback action instead
            # TODO: IPS is still the same as the original negative example though
            feedback = 1.
            action_prob: torch.Tensor = _get_counterfactual_em_probability(
                batch, predictions, i)

        losses.append(-ips_coeffs[i] * torch.log(action_prob) * feedback *
                      batch.feedbacks.weights[i])

    if reduce:
        return torch.mean(torch.stack(losses)), ips_coeffs
    return torch.stack(losses), ips_coeffs


class FeedbackOptimizer:
    def __init__(self, module: nn.Module, config: OptimizerConfig,
                 clip_ips_max: bool, always_use_ips: bool, use_ips: bool,
                 divide_by_original_probability_ips: bool,
                 entropy_coefficient: float,
                 counterfactual_negative_examples: bool):
        self._optimizer: torch.optim.Adam = torch.optim.Adam(
            module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2_coefficient)

        self._named_parameters = module.named_parameters()

        self._clip_ips_max: bool = clip_ips_max
        self._always_use_ips: bool = always_use_ips
        self._use_ips: bool = use_ips
        self._entropy_coefficient: float = entropy_coefficient
        self._divide_by_original_probability: bool = divide_by_original_probability_ips

        self._counterfactual_negative_examples: bool = counterfactual_negative_examples

    def _apply_loss(self, loss: torch.Tensor):
        loss.backward()
        self._optimizer.step()
        util.check_for_nan_grads(self._named_parameters)
        self._optimizer.zero_grad()
        util.check_for_nan_params(self._named_parameters)

    def compute_and_apply_loss(
        self, batch: StepBatch, predictions: Union[VoxelPredictions,
                                                   ActionPredictions]
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        loss, ips_coeff = feedback_loss(batch,
                                        predictions,
                                        self._clip_ips_max,
                                        self._always_use_ips,
                                        self._use_ips,
                                        self._divide_by_original_probability,
                                        self._counterfactual_negative_examples,
                                        reduce=False)

        if isinstance(predictions, ActionPredictions):
            entropy = action_prediction_entropy(predictions, reduce=False)
        else:
            # Multiply by 3751 because the entropy function is broken and
            # doesn't actually sum over the distribution correctly...
            entropy = voxel_prediction_entropy(predictions,
                                               reduce=False) * 3751

        total_loss = loss
        if self._entropy_coefficient:
            total_loss -= entropy * self._entropy_coefficient

        mean_loss = torch.mean(total_loss)

        self._apply_loss(mean_loss)

        supervised_losses: List[float] = list()
        feedback_losses: List[float] = list()
        for i, item in enumerate(batch.original_examples):
            if item.is_converted_feedback_example:
                supervised_losses.append((loss[i] / ips_coeff[i]).item())
            else:
                feedback_losses.append((loss[i] / ips_coeff[i]).item())

        ips_np: np.ndarray = np.array(ips_coeff)
        loss_dict = {
            'loss/batch': torch.mean(loss).item(),
            'loss/total': mean_loss.item(),
            'loss/max_ips': np.max(ips_np),
            'loss/min_ips': np.min(ips_np),
            'loss/mean_ips': np.mean(ips_np),
            'loss/entropy': torch.mean(entropy).item()
        }

        if supervised_losses:
            loss_dict['loss/supervised_only'] = np.mean(
                np.array(supervised_losses))
        if feedback_losses:
            loss_dict['loss/feedback_only'] = np.mean(
                np.array(feedback_losses))

        return loss_dict, total_loss
