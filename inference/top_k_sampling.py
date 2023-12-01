"""Implements top-k sampling for voxels / STOP action."""
from __future__ import annotations

import numpy as np
import random
import torch

from dataclasses import dataclass

from environment.action import Action, MOVEMENT_ACTIONS
from environment.player import Player
from inference.vin.vin_model import get_vin_predictions

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from environment.position import Position
    from environment.rotation import Rotation
    from environment.state import State
    from inference.predicted_voxel import VoxelPredictions
    from inference.vin.vin_model import Cerealbar_VIN
    from learning.batching import step_batch
    from simulation.unity_game import UnityGame
    from typing import Dict, List, Optional, Set, Tuple, Union

NUM_TOP_K_SAMPLES: int = 5


@dataclass
class Sample:
    action: Action
    target_voxel: Optional[Player]
    probability: float


def _get_action_distributions_from_vin_predictions(
    top_k_voxel_probs: Dict[Tuple[Position, Rotation], float],
    vin_action_results: Dict[Tuple[Position, Rotation],
                             Set[Action]], stop_probability: float,
    randomly_choose_action: bool, max_probs_for_action_voxels: bool
) -> Dict[Action, Tuple[float, Optional[Tuple[Position, Rotation]]]]:
    # Sort voxels into buckets by these actions.
    action_buckets: Dict[Action, Dict[Tuple[Position, Rotation], float]] = {
        action: dict()
        for action in MOVEMENT_ACTIONS
    }

    # Handle the STOP action -- throw out the lowest-probability voxel if STOP is in the top-k.
    sorted_voxel_probs: List[Tuple[Tuple[Position, Rotation],
                                   float]] = sorted(top_k_voxel_probs.items(),
                                                    key=lambda x: x[1])
    min_voxel_probability: float = sorted_voxel_probs[0][1]

    voxel_to_ignore: Optional[Tuple[Position, Rotation]] = None
    include_stop_in_sampling: bool = False
    if stop_probability > min_voxel_probability:
        # Throw out the minimum-probability voxel
        voxel_to_ignore = sorted_voxel_probs[0][0]
        include_stop_in_sampling = True

    for voxel, actions in vin_action_results.items():
        if voxel == voxel_to_ignore:
            continue

        if randomly_choose_action:
            action = random.choice(list(actions))
            action_buckets[action][voxel] = top_k_voxel_probs[voxel]
        else:
            for action in actions:
                action_buckets[action][voxel] = top_k_voxel_probs[voxel]

    # Associate each action with a probability.
    action_probabilities: Dict[Action,
                               Tuple[float,
                                     Optional[Tuple[Position,
                                                    Rotation]]]] = dict()
    if include_stop_in_sampling:
        action_probabilities[Action.STOP] = (stop_probability, None)

    # Maps each action to the voxel with the highest probability associated with it, for reference later when an
    # action is sampled.
    action_max_voxel: Dict[Action, Tuple[Position, Rotation]] = dict()

    for action, bucketed_voxels in action_buckets.items():
        if bucketed_voxels:
            sorted_voxels: List[Tuple[Tuple[Position, Rotation]],
                                float] = sorted(bucketed_voxels.items(),
                                                key=lambda x: x[1])
            max_voxel: Tuple[Position, Rotation] = sorted_voxels[-1][0]

            if max_probs_for_action_voxels:
                action_probabilities[action] = (sorted_voxels[-1][1],
                                                max_voxel)
            else:
                action_probabilities[action] = (sum(bucketed_voxels.values()),
                                                max_voxel)
    return action_probabilities


def _sample_for_rollout(
    action_probabilities: Dict[Action, Tuple[float, Optional[Tuple[Position,
                                                                   Rotation]]]]
) -> Sample:
    # Sample from these highest-probability actions according to their probabilities.
    sorted_action_probs: List[Tuple[Action, float]] = sorted(
        action_probabilities.items(), key=lambda x: x[1])
    prob_mass: float = sum(action_probabilities.values())

    randfloat: float = random.random() * prob_mass
    baseprob: float = 0

    sampled_action: Optional[Action] = None
    for ac, prob in sorted_action_probs:
        baseprob += prob
        if randfloat < baseprob:
            sampled_action = ac
            break

    assert sampled_action is not None

    # Add a sample including the probability, the action, and target voxel (if applicable) of the sample.
    sampled_voxel: Optional[Player] = None
    if sampled_action != Action.STOP:
        pos, rot = action_probabilities[sampled_action][1]
        sampled_voxel = Player(True, pos, rot)

    return Sample(sampled_action, sampled_voxel,
                  action_probabilities[sampled_action][0])


def get_top_k_vin_samples(
    predictions: VoxelPredictions, k: int, obstacle_mask: torch.Tensor,
    vin: Cerealbar_VIN, current_states: List[State],
    batch: step_batch.StepBatch, allow_player_intersections: bool
) -> Tuple[List[Dict[Tuple[Position, Rotation], float]], List[Dict[Tuple[
        Position, Rotation], Set[Action]]]]:
    batch_size: int = predictions.get_batch_size()

    all_top_k_probabilities: List[Dict[Tuple[Position, Rotation],
                                       float]] = list()
    all_top_k_voxels: List[List[Tuple[Position, Rotation]]] = list()

    for j in range(batch_size):
        # Get the K most probably voxels (or stop).
        top_voxel_probabilities: Dict[Tuple[Position, Rotation],
                                      float] = predictions.get_top_k_voxels(
                                          j, k, obstacle_mask[j])
        assert len(top_voxel_probabilities) == k
        all_top_k_probabilities.append(top_voxel_probabilities)
        all_top_k_voxels.append(sorted(top_voxel_probabilities.keys()))

    # Now do K VIN forward passes
    vin_results: List[Dict[Tuple[Position, Rotation],
                           Set[Action]]] = [dict() for _ in range(batch_size)]

    for i in range(k):
        # For each of these top k, run the VIN on top of them to get an action (randomly sample an action if
        # there is more than one possible action).
        relevant_targets: List[Player] = list()
        for j in range(batch_size):
            target_pos, target_rot = all_top_k_voxels[j][i]
            relevant_targets.append(Player(True, target_pos, target_rot))

        possible_actions: List[Set[Action]] = get_vin_predictions(
            vin, current_states, batch, allow_player_intersections,
            relevant_targets)

        for j, ex_actions in enumerate(possible_actions):
            vin_results[j][(relevant_targets[j].position,
                            relevant_targets[j].rotation)] = ex_actions

    return all_top_k_probabilities, vin_results


def get_batched_action_distributions(
    all_top_k_probabilities: List[Dict[Tuple[Position, Rotation], float]],
    vin_results: List[Dict[Tuple[Position, Rotation], Set[Action]]],
    batch_size: int, stop_probabilities: torch.Tensor,
    randomly_choose_action: bool, max_probs_for_action_voxels: bool
) -> List[Dict[Action, Tuple[float, Optional[Tuple[Position, Rotation]]]]]:

    assert len(all_top_k_probabilities) == len(vin_results)

    distributions: List[Dict[Action,
                             Tuple[float,
                                   Optional[Tuple[Position,
                                                  Rotation]]]]] = list()
    for j in range(batch_size):
        top_k_voxel_probs: Dict[Tuple[Position, Rotation],
                                float] = all_top_k_probabilities[j]
        vin_action_results: Dict[Tuple[Position, Rotation],
                                 Set[Action]] = vin_results[j]

        assert len(vin_action_results) == len(top_k_voxel_probs)

        distributions.append(
            _get_action_distributions_from_vin_predictions(
                top_k_voxel_probs, vin_action_results,
                stop_probabilities[j].item(), randomly_choose_action,
                max_probs_for_action_voxels))
    return distributions


def sample_from_top_k(
        predictions: VoxelPredictions,
        k: int,
        obstacle_mask: torch.Tensor,
        vin: Cerealbar_VIN,
        current_states: List[State],
        batch: step_batch.StepBatch,
        allow_player_intersections: bool,
        randomly_choose_action: bool = True,
        max_probs_for_action_voxels: bool = True) -> List[Sample]:
    batch_size: int = predictions.get_batch_size()

    top_k_probabilities, vin_actions = get_top_k_vin_samples(
        predictions, k, obstacle_mask, vin, current_states, batch,
        allow_player_intersections)

    distributions: List[Dict[Action, Tuple[float, Optional[Tuple[
        Position, Rotation]]]]] = get_batched_action_distributions(
            top_k_probabilities, vin_actions, batch_size,
            predictions.stop_probabilities, randomly_choose_action,
            max_probs_for_action_voxels)

    samples: List[Sample] = [
        _sample_for_rollout(distributions[i]) for i in range(batch_size)
    ]

    return samples


def get_action_distributions_up_to_k(
    predictions: VoxelPredictions,
    k: int,
    obstacle_mask: torch.Tensor,
    vin: Cerealbar_VIN,
    current_states: List[State],
    batch: step_batch.StepBatch,
    allow_player_intersections: bool,
    randomly_choose_action: bool = True,
    max_probs_for_action_voxels: bool = True
) -> Dict[int, List[Dict[Action, Tuple[float, Optional[Tuple[Position,
                                                             Rotation]]]]]]:

    batch_size: int = predictions.get_batch_size()

    top_k_probabilities, vin_actions = get_top_k_vin_samples(
        predictions, k, obstacle_mask, vin, current_states, batch,
        allow_player_intersections)

    all_distributions: Dict[int, List[Dict[Action, Tuple[float, Optional[Tuple[
        Position, Rotation]]]]]] = dict()
    for i in range(k):
        # Get action distributions up to i+1 items.
        current_k_top_probabilities: List[Dict[Tuple[Position, Rotation],
                                               float]] = list()
        current_k_actions: List[Dict[Tuple[Position, Rotation],
                                     Set[Action]]] = list()

        for j in range(batch_size):
            all_probs: Dict[Tuple[Position, Rotation],
                            float] = top_k_probabilities[j]
            sorted_probs: List[Tuple[Tuple[Position, Rotation]],
                               float] = sorted(all_probs.items(),
                                               key=lambda x: x[1])[::-1]
            keys_to_keep: Set[Tuple[Position, Rotation]] = {
                voxel
                for voxel, probability in sorted_probs[:i + 1]
            }

            current_k_top_probabilities.append(
                {key: all_probs[key]
                 for key in keys_to_keep})
            current_k_actions.append(
                {key: vin_actions[j][key]
                 for key in keys_to_keep})

        all_distributions[i + 1] = get_batched_action_distributions(
            current_k_top_probabilities, current_k_actions, batch_size,
            predictions.stop_probabilities, randomly_choose_action,
            max_probs_for_action_voxels)

    return all_distributions


def get_top_k_vin_sample(predictions: VoxelPredictions, vin: Cerealbar_VIN,
                         current_state: State, game: Optional[UnityGame],
                         batch: step_batch.StepBatch,
                         allow_player_intersections: bool, log_fn,
                         k: int) -> Tuple[Action, Optional[Player]]:
    if game:
        dist_to_send: np.ndarray = torch.sum(
            predictions.voxel_probabilities[0], dim=0).numpy()
        game.send_map_probability_dist(dist_to_send)

    # Get the top-k voxels and STOP action.
    obstacle_mask: torch.Tensor = batch.environment_state.static_info.obstacle_mask
    if not allow_player_intersections:
        obstacle_mask = batch.environment_state.get_all_obstacles()

    sample: Sample = sample_from_top_k(predictions, k, obstacle_mask, vin,
                                       [current_state], batch,
                                       allow_player_intersections)[0]

    text: str = f'Executing action {sample.action} (probability: {(100. * sample.probability):.1f}%)'
    if log_fn == print:
        input(text)
    else:
        log_fn(text)

    return sample.action, sample.target_voxel
