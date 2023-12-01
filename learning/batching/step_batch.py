"""Batches StepExamples."""
from __future__ import annotations

from dataclasses import dataclass

import copy
import logging
import torch

from environment.action import Action
from environment.player import Player
from environment.position import EDGE_WIDTH, Position
from environment.rotation import ROTATIONS, Rotation
from inference.predicted_action_distribution import PREDICTABLE_ACTIONS
from inference.predicted_voxel import VoxelPredictions
from inference.predicted_action_distribution import ActionPredictions
from learning.batching import environment_batch, environment_batcher, instruction_batch
from simulation.planner import find_path_between_positions

from util import torch_util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.bpe_tokenizer import BPETokenizer
    from data.dataset import GamesCollection
    from data.step_example import StepExample
    from environment.state import State
    from typing import List, Optional, Tuple, Union

REBOOT_PENALTY: int = 5


@dataclass
class StepTarget:
    """
    A target for a single step, including whether the agent should stop, and if not, the plausible locations it 
    should stop in, represented as tensors.
    
    This is batched.
    
    Attributes:
        self.action_type_labels
            The stop labels for the examples: 0 if agent should not stop; 1 if it should. Size: B (long)
        self.plausible_target_voxels
            The labels for the plausible target goals: 0 if voxel is not a plausible goal; 1 if it is. 
            Size: B x 6 x 25 x 25 (long)
            
    For an example b in the batch, exactly one of action_type_labels[b] and plausible_target_voxels[b] should be completely 
    zero (the target should be EITHER stop OR a set of voxels.
    """
    action_type_labels: torch.Tensor
    plausible_target_voxels: Optional[torch.Tensor]

    def to_device(self):
        self.action_type_labels = self.action_type_labels.to(torch_util.DEVICE)

        if self.plausible_target_voxels is not None:
            self.plausible_target_voxels = self.plausible_target_voxels.to(
                torch_util.DEVICE)


@dataclass
class StepFeedbackBatch:
    """A batch of feedback annotations on an example."""
    original_distributions: Union[VoxelPredictions, ActionPredictions]
    feedbacks: torch.Tensor
    weights: torch.Tensor
    sampled_configurations: List[Union[Action, Player]]
    use_ips: List[bool]

    vin_obstacles: Optional[torch.Tensor]
    executed_actions: List[Action]
    shortest_path_lengths: List[int]

    def to_device(self):
        self.original_distributions.to_device()
        self.feedbacks = self.feedbacks.to(torch_util.DEVICE)
        self.weights = self.weights.to(torch_util.DEVICE)

        if self.vin_obstacles is not None:
            self.vin_obstacles = self.vin_obstacles.to(torch_util.DEVICE)


@dataclass
class StepBatch:
    """A Batch of StepExamples.
    
    Attributes:
        self.original_examples
            Reference to the original example, for ease of accessing things like the target, or original (
            non-batched) data.
        self.instructions
            The batched instructions.
        self.current_positions
            The agent's current positions (x, y)
        self.current_rotations
            The agent's current rotations (one of six possible, in radians)
    
    """
    original_examples: List[StepExample]

    instructions: Optional[instruction_batch.InstructionBatch]

    environment_state: environment_batch.EnvironmentBatch

    target: Optional[StepTarget]

    feedbacks: Optional[StepFeedbackBatch]

    def get_batch_size(self) -> int:
        return len(self.original_examples)

    def to_device(self):
        self.environment_state.to_device()

        if self.instructions is not None:
            self.instructions.to_device()
        if self.target is not None:
            self.target.to_device()
        if self.feedbacks is not None:
            self.feedbacks.to_device()


def _batch_targets(examples: List[StepExample],
                   directly_predict_actions: bool) -> StepTarget:
    action_type_labels: torch.Tensor = torch.zeros(len(examples)).long()

    for i, example in enumerate(examples):
        if directly_predict_actions:
            target = example.target_action
            if target is None:
                target = example.sampled_action

            action_type_labels[i] = PREDICTABLE_ACTIONS.index(target)
        else:
            if example.target_action == Action.STOP:
                action_type_labels[i] = 1
            elif example.should_copy:
                action_type_labels[i] = 2

    voxel_labels: Optional[torch.Tensor] = None
    if not directly_predict_actions:
        batch_size: int = len(examples)
        voxel_labels: torch.Tensor = torch.zeros(
            (batch_size, len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH)).long()

        for i, example in enumerate(examples):
            if not example.is_supervised_example:
                raise ValueError(
                    'Cannot batch targets for example without targets.')

            if example.target_action == Action.STOP:
                # No target voxel: predict STOP instead.
                continue

            for target_voxel in example.possible_target_configurations:
                position: Position = target_voxel.position
                rotation: Rotation = target_voxel.rotation

                voxel_labels[i][ROTATIONS.index(rotation)][position.x][
                    position.y] = 1

    return StepTarget(action_type_labels, voxel_labels)


def get_vin_settings_for_example(
        current_config: Player, goal_config: Player,
        obstacle_locations: List[Position],
        avoid_card_locations: List[Position]) -> Tuple[int, bool, bool]:
    mask_cards: bool = True
    mask_objects: bool = True
    shortest_path = find_path_between_positions(
        obstacle_locations + avoid_card_locations, current_config, goal_config)
    if shortest_path is None:
        # Don't mask out other cards.
        mask_cards = False

        # Get a new shortest path, without using cards as obstacles.
        shortest_path = find_path_between_positions(obstacle_locations,
                                                    current_config,
                                                    goal_config)

        if shortest_path is None:
            # This may still happen in very rare cases, e.g., if there is a position which is completely enclosed
            #  by obstacles or cut off by the leader. In that case, the VIN should be ran without any obstacles,
            # and the action chosen must be executable.
            shortest_path = find_path_between_positions(
                list(), current_config, goal_config)

            mask_objects = False
    return len(shortest_path[0]), mask_cards, mask_objects


def get_vin_obstacles(all_mask_cards: List[bool], all_mask_objects: List[bool],
                      default_obstacle_mask: torch.Tensor,
                      current_states: List[State],
                      goal_positions: List[Position]) -> torch.Tensor:
    assert len(all_mask_cards) == len(all_mask_objects)
    batch_size: int = len(all_mask_cards)
    reconstructed_obstacles: List[torch.Tensor] = list()

    cards_to_avoid: torch.Tensor = torch.zeros(
        (batch_size, EDGE_WIDTH, EDGE_WIDTH), device=torch_util.DEVICE)

    for i in range(batch_size):
        current_game_state: State = current_states[i]
        current_follower: Player = current_game_state.follower
        argmax_pos: Position = goal_positions[i]

        obstacle_locations: List[Position] = list()
        for x in range(EDGE_WIDTH):
            for y in range(EDGE_WIDTH):
                if default_obstacle_mask[i][x][y] > 0:
                    # This may also include the leader.
                    obstacle_locations.append(Position(x, y))

        avoid_card_locations: List[Position] = list()
        for card in current_game_state.cards:
            if card.position not in {current_follower.position, argmax_pos}:
                avoid_card_locations.append(card.position)

        if all_mask_objects[i]:
            reconstructed_obstacles.append(
                default_obstacle_mask[i].unsqueeze(0).to(torch_util.DEVICE))
        else:
            logging.info(
                f'Had to find shortest path between positions for index {i} in batch without any '
                f'obstacles.')
            reconstructed_obstacles.append(
                torch.zeros((1, EDGE_WIDTH, EDGE_WIDTH),
                            device=torch_util.DEVICE))

        if all_mask_cards[i]:
            for card in current_game_state.cards:
                if card.position not in {
                        current_follower.position, argmax_pos
                }:
                    cards_to_avoid[i][card.position.x][card.position.y] = 1

    return torch.cat(reconstructed_obstacles, dim=0) + cards_to_avoid


def _batch_feedbacks(data: List[StepExample], obstacle_masks: torch.Tensor,
                     direct_action_prediction: bool) -> StepFeedbackBatch:
    batch_size: int = len(data)

    feedbacks: torch.Tensor = torch.zeros(batch_size)
    weights: torch.Tensor = torch.zeros(batch_size)
    samples: List[Union[Action, Player]] = list()

    vin_obstacles: List[torch.Tensor] = list()
    executed_actions: List[Action] = list()
    shortest_path_lengths: List[int] = list()

    use_ips: List[bool] = list()

    # Voxel-prediction model only
    voxel_probabilities: List[torch.Tensor] = list()
    combined_stop_probabilities: torch.Tensor = torch.zeros(
        (len(data)), device=torch_util.DEVICE)
    combined_copy_probabilities: Optional[torch.Tensor] = None

    # Action-prediction model only
    action_probabilities: List[torch.Tensor] = list()

    for i, example in enumerate(data):
        ips: bool = True

        if not example.is_converted_feedback_example and example.use_ips:
            if direct_action_prediction:
                action_probabilities.append(
                    torch.tensor(
                        copy.deepcopy(
                            example.action_annotation.probability_dist.
                            action_probabilities)).unsqueeze(0))
            else:
                voxel_probabilities.append(
                    torch.tensor(
                        copy.deepcopy(
                            example.action_annotation.probability_dist.
                            voxel_probabilities)).unsqueeze(0))
                combined_stop_probabilities[i] = torch.tensor(
                    copy.deepcopy(example.action_annotation.probability_dist.
                                  stop_probabilities))
                if hasattr(example.action_annotation.probability_dist,
                           'copy_probabilities') and example.action_annotation.probability_dist.copy_probabilities is not\
                        None:
                    raise ValueError(
                        'Need to support copy probabilities in batching feedback!'
                    )
        else:
            voxel_probabilities.append(
                torch.zeros(1, len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH))
            action_probabilities.append(
                torch.zeros(1, len(PREDICTABLE_ACTIONS)))

            ips = False

        feedback: int = example.action_annotation.feedback.num_positive - \
                        example.action_annotation.feedback.num_negative
        if example.action_annotation.feedback.reboot:
            feedback -= REBOOT_PENALTY

        # TODO: don't just map to -1 or 1
        if feedback > 0:
            feedbacks[i] = 1.
        elif feedback < 0:
            feedbacks[i] = -1

        weights[i] = example.action_annotation.feedback.weight

        executed_actions.append(example.sampled_action)
        if example.sampled_action == Action.STOP:
            samples.append(example.sampled_action)
            vin_obstacles.append(torch.zeros((EDGE_WIDTH, EDGE_WIDTH)))
            shortest_path_lengths.append(-1)
        elif direct_action_prediction:
            samples.append(example.sampled_action)
        else:
            sampled_voxel: Player = example.action_annotation.sampled_goal_voxel
            sampled_voxel_position: Position = sampled_voxel.position

            samples.append(sampled_voxel)

            current_state: State = example.state
            current_follower: Player = current_state.follower

            obstacle_locations: List[Position] = list()
            for x in range(EDGE_WIDTH):
                for y in range(EDGE_WIDTH):
                    if obstacle_masks[i][x][y] > 0:
                        # This may also include the leader.
                        obstacle_locations.append(Position(x, y))
            avoid_card_locations: List[Position] = list()
            for card in current_state.cards:
                if card.position not in {
                        current_follower.position, sampled_voxel_position
                }:
                    avoid_card_locations.append(card.position)

            # TODO: Don't do this every batch; instead pre-compute for every StepExample
            shortest_path_length, mask_cards, mask_objects = get_vin_settings_for_example(
                current_follower, sampled_voxel, obstacle_locations,
                avoid_card_locations)
            vin_obstacles.append(
                get_vin_obstacles([mask_cards], [mask_objects],
                                  obstacle_masks[i].unsqueeze(0),
                                  [current_state], [sampled_voxel_position]))
            shortest_path_lengths.append(shortest_path_length)

        use_ips.append(ips)

    if direct_action_prediction:
        original_predictions = ActionPredictions(
            torch.cat(action_probabilities, dim=0),
            format=torch_util.TensorType.TORCH)
        all_vin_obstacles = None
    else:
        original_predictions = VoxelPredictions(
            torch.cat(voxel_probabilities, dim=0),
            combined_stop_probabilities,
            combined_copy_probabilities,
            global_interpretation=True,
            format=torch_util.TensorType.TORCH)
        all_vin_obstacles: torch.Tensor = torch.cat([
            obstacles.view(1, EDGE_WIDTH, EDGE_WIDTH).to(torch_util.DEVICE)
            for obstacles in vin_obstacles
        ],
                                                    dim=0)

    return StepFeedbackBatch(original_predictions, feedbacks, weights, samples,
                             use_ips, all_vin_obstacles, executed_actions,
                             shortest_path_lengths)


def batch_steps(data: List[StepExample],
                games: GamesCollection,
                instruction_tokenizer: Optional[BPETokenizer],
                batcher: environment_batcher.EnvironmentBatcher,
                batch_targets: bool = True,
                batch_feedback: bool = False,
                directly_predict_actions: bool = False,
                allow_player_intersections: bool = False) -> StepBatch:
    if batch_targets and batch_feedback:
        raise ValueError(
            'Only one of batching targets or feedback can be set to True.')

    environment: environment_batch.EnvironmentBatch = batcher.batch_environments(
        data, games, batch_previous_targets=not directly_predict_actions)

    obstacle_mask: torch.Tensor = environment.static_info.obstacle_mask
    if not allow_player_intersections:
        # TODO: During feedback fine-tuning, this will condition will be triggered because the feedback examples
        # don't allow intersection. But a small number of supervised examples require intersection.
        obstacle_mask = environment.get_all_obstacles()

    targets: Optional[StepTarget] = None
    feedbacks: Optional[StepFeedbackBatch] = None
    if batch_targets:
        targets = _batch_targets(data, directly_predict_actions)
    elif batch_feedback:
        feedbacks = _batch_feedbacks(data, obstacle_mask,
                                     directly_predict_actions)

    return StepBatch(
        data,
        instruction_batch.batch_instructions(
            [example.instruction for example in data], instruction_tokenizer)
        if instruction_tokenizer is not None else None, environment, targets,
        feedbacks)
