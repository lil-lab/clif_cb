# Originally adopted from https://github.com/kentsommer/pytorch-value-iteration-networks/blob/master/model.py
from __future__ import annotations

import logging

from data.step_example import StepExample
from environment.player import Player
from environment.state import State
from learning.batching.step_batch import StepBatch, get_vin_settings_for_example, get_vin_obstacles

from model.hex_space import hex_util
from environment.position import EDGE_WIDTH, Position, out_of_bounds
from simulation.planner import get_new_player_orientation
from util import torch_util
from environment.action import MOVEMENT_ACTIONS, Action
from environment.rotation import ROTATIONS, Rotation

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import TYPE_CHECKING, List, Set, Dict

if TYPE_CHECKING:
    from typing import Dict, Optional, Tuple

MAX_NUM_ITER: int = 100


class VIN(nn.Module):
    def __init__(self, config):
        super(VIN, self).__init__()

    def forward(self, input_view, state_x, state_y, k):
        pass


def _get_q_for_actions(q_map: torch.Tensor,
                       axial_current_state: torch.Tensor) -> torch.Tensor:
    batch_size: int = q_map.size(0)
    return q_map[[i for i in range(batch_size)], :,
                 (axial_current_state[:, 0].long() + 1) % len(ROTATIONS),
                 axial_current_state[:, 1],
                 axial_current_state[:, 2]]  # q: (batch_sz, num_actions)


class Cerealbar_VIN(nn.Module):
    def __init__(self, configs={"gamma": 0.9}):
        super(Cerealbar_VIN, self).__init__()
        self._initial_q_map = torch.zeros(
            (24, 1, 3, 3), requires_grad=False).to(torch_util.DEVICE)
        self._value_transitions = torch.zeros(
            (24, 1, 3, 3), requires_grad=False).to(torch_util.DEVICE)
        self._gamma = configs["gamma"]

        self._set_kernels(_get_cerealbar_axial_2d_kernels(), is_axial=True)

        # hexaconv related
        self._offset_to_axial_converter: hex_util.OffsetToAxialConverter = hex_util.OffsetToAxialConverter(
            EDGE_WIDTH)
        offset_mask = torch.ones((EDGE_WIDTH, EDGE_WIDTH),
                                 requires_grad=False).unsqueeze(0).unsqueeze(0)
        self._axial_mask = self._offset_to_axial_converter(
            offset_mask.to(torch_util.DEVICE))

    def _apply_cerealbar_transition(self, inputs, kernels, obstacles):
        batch_sz, num_orientation, env_height, env_width = inputs.shape
        num_action = 4

        # Transform inputs to make the transition problem 2D conv ops
        # (batch_sz, num_orientation, env_height, env_width)  ==> (batch_sz, num_actions * num_orientation, env_height, env_width)
        transformed_inputs = self._transform_value_map(inputs)

        # Mask obstacle locations and out of map locations in axial maps
        masked_inputs = transformed_inputs * obstacles
        masked_inputs = masked_inputs * self._axial_mask

        # Apply hex_conv op with grouping
        # (batch_sz, num_actions * num_orientation, env_height, env_width)  ==> (batch_sz, num_actions * num_orientation, num_orientation, env_height, env_width)
        conv_output = F.conv2d(masked_inputs,
                               kernels,
                               padding=(1, 1),
                               groups=masked_inputs.shape[1])

        # Reshape outputs
        # (batch_sz, num_actions * num_orientation, num_orientation, env_height, env_width) ==> (batch_sz, num_actions, num_orientation, env_height, env_width)
        output = conv_output.view(batch_sz, num_action, num_orientation,
                                  env_height, env_width)

        # Mask output
        output = output * obstacles.unsqueeze(1)
        output = output * self._axial_mask.unsqueeze(1)

        return output

    def _transform_value_map(self, v):
        # actions are sorted as following: [MF, MB, RR, RL]
        # orientations (alpha) are sorted as following: [NE, E, SE, SW, W, NW]
        batch_sz, num_orientation, env_height, env_width = v.shape
        new_value_map = torch.zeros(
            (batch_sz, 24, env_height, env_width)).to(torch_util.DEVICE)

        # 0: output: (action, alpha) = (MF, NE) <= input: (alpha) = (NE)
        new_value_map[:, 0, :, :] = v[:, 0, :, :]
        # 1: output: (action, alpha) = (MF, E) <= input: (alpha) = (E)
        new_value_map[:, 1, :, :] = v[:, 1, :, :]
        # 2: output: (action, alpha) = (MF, SE) <= input: (alpha) = (SE)
        new_value_map[:, 2, :, :] = v[:, 2, :, :]
        # 3: output: (action, alpha) = (MF, SW) <= input: (alpha) = (SW)
        new_value_map[:, 3, :, :] = v[:, 3, :, :]
        # 4: output: (action, alpha) = (MF, W) <= input: (alpha) = (W)
        new_value_map[:, 4, :, :] = v[:, 4, :, :]
        # 5: output: (action, alpha) = (MF, NW) <= input: (alpha) = (NW)
        new_value_map[:, 5, :, :] = v[:, 5, :, :]

        # 6: output: (action, alpha) = (MB, NE) <= input: (alpha) = (NE)
        new_value_map[:, 6, :, :] = v[:, 0, :, :]
        # 7: output: (action, alpha) = (MB, E) <= input: (alpha) = (E)
        new_value_map[:, 7, :, :] = v[:, 1, :, :]
        # 8: output: (action, alpha) = (MB, SE) <= input: (alpha) = (SE)
        new_value_map[:, 8, :, :] = v[:, 2, :, :]
        # 9: output: (action, alpha) = (MB, SW) <= input: (alpha) = (SW)
        new_value_map[:, 9, :, :] = v[:, 3, :, :]
        # 10: output: (action, alpha) = (MB, W) <= input: (alpha) = (W)
        new_value_map[:, 10, :, :] = v[:, 4, :, :]
        # 11: output: (action, alpha) = (MB, NW) <= input: (alpha) = (NW)
        new_value_map[:, 11, :, :] = v[:, 5, :, :]

        # 12: output: (action, alpha) = (RR, NE) <= input: (alpha) = (E)
        new_value_map[:, 12, :, :] = v[:, 1, :, :]
        # 13: output: (action, alpha) = (RR, E) <= input: (alpha) = (SE)
        new_value_map[:, 13, :, :] = v[:, 2, :, :]
        # 14: output: (action, alpha) = (RR, SE) <= input: (alpha) = (SW)
        new_value_map[:, 14, :, :] = v[:, 3, :, :]
        # 15: output: (action, alpha) = (RR, SW) <= input: (alpha) = (W)
        new_value_map[:, 15, :, :] = v[:, 4, :, :]
        # 16: output: (action, alpha) = (RR, W) <= input: (alpha) = (NW)
        new_value_map[:, 16, :, :] = v[:, 5, :, :]
        # 17: output: (action, alpha) = (RR, NW) <= input: (alpha) = (NE)
        new_value_map[:, 17, :, :] = v[:, 0, :, :]

        # 18: output: (action, alpha) = (RL, NE) <= input: (alpha) = (NW)
        new_value_map[:, 18, :, :] = v[:, 5, :, :]
        # 19: output: (action, alpha) = (RL, E) <= input: (alpha) = (NE)
        new_value_map[:, 19, :, :] = v[:, 0, :, :]
        # 20: output: (action, alpha) = (RL, SE) <= input: (alpha) = (E)
        new_value_map[:, 20, :, :] = v[:, 1, :, :]
        # 21: output: (action, alpha) = (RL, SW) <= input: (alpha) = (SE)
        new_value_map[:, 21, :, :] = v[:, 2, :, :]
        # 22: output: (action, alpha) = (RL, W) <= input: (alpha) = (SW)
        new_value_map[:, 22, :, :] = v[:, 3, :, :]
        # 23: output: (action, alpha) = (RL, NW) <= input: (alpha) = (W)
        new_value_map[:, 23, :, :] = v[:, 4, :, :]

        return new_value_map

    def _get_axial_representations(self, offset_input_goals: torch.Tensor, offset_obstacles: torch.Tensor,
                                   offset_current_state: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Converting tensors from offset to axial
        axial_input_goals: torch.Tensor = self._offset_to_axial_converter(
            offset_input_goals)
        axial_obstacles: torch.Tensor = self._offset_to_axial_converter(
            offset_obstacles.unsqueeze(1))
        axial_obstacles = -(axial_obstacles - 1.)
        additional_size = (EDGE_WIDTH - 1) // 2
        u = offset_current_state[:, 1].unsqueeze(1)
        v = offset_current_state[:, 2].unsqueeze(1)
        u = u - v // 2
        u += additional_size
        alpha = offset_current_state[:, 0].unsqueeze(1)
        axial_current_state: torch.Tensor = torch.cat([alpha, u, v], 1).long()

        return axial_input_goals, axial_obstacles, axial_current_state

    def _update_q_map(
            self, reward_map: torch.Tensor, value_map: torch.Tensor,
            axial_obstacles: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reward_transitioned: torch.Tensor = self._apply_cerealbar_transition(
            reward_map, self._initial_q_map, axial_obstacles)
        value_transitioned: torch.Tensor = self._apply_cerealbar_transition(
            value_map, self._value_transitions, axial_obstacles)

        q_map = reward_transitioned + value_transitioned
        value_map, _ = torch.max(q_map, dim=1, keepdim=False)

        return q_map, value_map

    def forward(self,
                offset_input_goals: torch.tensor,
                offset_current_state: torch.tensor,
                offset_obstacles: torch.tensor,
                num_iterations=-1):
        axial_input_goals, axial_obstacles, axial_current_state = self._get_axial_representations(
            offset_input_goals, offset_obstacles, offset_current_state)

        # hyperparams
        batch_sz, num_orientation, env_height, env_width = axial_input_goals.shape

        num_iterations = num_iterations if num_iterations is not None else max(
            env_width, num_orientation)

        # t = 0 step
        reward_map = axial_input_goals  # Reward
        q_map = self._apply_cerealbar_transition(reward_map,
                                                 self._initial_q_map,
                                                 axial_obstacles)

        action_qs = _get_q_for_actions(q_map, axial_current_state)

        value_map, _ = torch.max(q_map, dim=1, keepdim=False)

        # Update q and v values
        if num_iterations >= 0:
            for i in range(num_iterations):
                q_map, value_map = self._update_q_map(reward_map, value_map,
                                                      axial_obstacles)

                action_qs = _get_q_for_actions(q_map, axial_current_state)
        elif num_iterations < 0:
            # Run until the q values are nonzero.
            nonzero_qs = action_qs
            found_nonzero: Dict[int, bool] = {
                i: torch.sum(nonzero_qs[i]).item() > 0
                for i in range(batch_sz)
            }

            num_iter = 0
            while False in set(
                    found_nonzero.values()) and num_iter < MAX_NUM_ITER:
                # Continue doing iterations.
                q_map, value_map = self._update_q_map(reward_map, value_map,
                                                      axial_obstacles)
                action_qs = _get_q_for_actions(q_map, axial_current_state)
                num_iter += 1

                # Check if any Q values are now nonzero, and if so, record them.
                for i in range(batch_sz):
                    if not found_nonzero[i] and torch.sum(
                            action_qs[i]).item() > 0:
                        nonzero_qs[i] = action_qs[i]
                        found_nonzero[i] = True

            action_qs = nonzero_qs

        return action_qs

    def _set_kernels(self, q_weights: torch.tensor, is_axial: bool = False):
        """
        set transition kernels
        """
        if is_axial:
            self._initial_q_map = q_weights
            self._initial_q_map.requires_grad = False

            # setting a discounting factor gamma for for valu iteration: gamma * sum_{s'} P(s'|s,a) V_n(S')
            self._value_transitions[self._initial_q_map != 0.] = self._gamma
            self._value_transitions.requires_grad = False

        else:
            raise NotImplementedError(
                "Conv kernels have to be in axial coordinates.")


def _get_cerealbar_axial_2d_kernels() -> torch.tensor:
    """
    ref (the discussion of group ops in conv2d): https://discuss.pytorch.org/t/convolution-operation-without-the-final-summation/56466/3
    """
    kernels = torch.zeros((24, 1, 3, 3),
                          requires_grad=False).to(torch_util.DEVICE)
    # output: (action, alpha) = (MF, NE) <= input: (alpha) = (NE)
    kernels[0, 0, 1, 2] = 1
    # output: (action, alpha) = (MF, E) <= input: (alpha) = (E)
    kernels[1, 0, 2, 1] = 1
    # output: (action, alpha) = (MF, SE) <= input: (alpha) = (SE)
    kernels[2, 0, 2, 0] = 1
    # output: (action, alpha) = (MF, SW) <= input: (alpha) = (SW)
    kernels[3, 0, 1, 0] = 1
    # output: (action, alpha) = (MF, W) <= input: (alpha) = (W)
    kernels[4, 0, 0, 1] = 1
    # output: (action, alpha) = (MF, NW) <= input: (alpha) = (NW)
    kernels[5, 0, 0, 2] = 1

    # output: (action, alpha) = (MB, NE) <= input: (alpha) = (NE)
    kernels[6, 0, 1, 0] = 1
    # output: (action, alpha) = (MB, E) <= input: (alpha) = (E)
    kernels[7, 0, 0, 1] = 1
    # output: (action, alpha) = (MB, SE) <= input: (alpha) = (SE)
    kernels[8, 0, 0, 2] = 1
    # output: (action, alpha) = (MB, SW) <= input: (alpha) = (SW)
    kernels[9, 0, 1, 2] = 1
    # output: (action, alpha) = (MB, W) <= input: (alpha) = (W)
    kernels[10, 0, 2, 1] = 1
    # output: (action, alpha) = (MB, NW) <= input: (alpha) = (NW)
    kernels[11, 0, 2, 0] = 1

    # output: (action, alpha) = (RR, NE) <= input: (alpha) = (E)
    kernels[12, 0, 1, 1] = 1
    # output: (action, alpha) = (RR, E) <= input: (alpha) = (SE)
    kernels[13, 0, 1, 1] = 1
    # output: (action, alpha) = (RR, SE) <= input: (alpha) = (SW)
    kernels[14, 0, 1, 1] = 1
    # output: (action, alpha) = (RR, SW) <= input: (alpha) = (W)
    kernels[15, 0, 1, 1] = 1
    # output: (action, alpha) = (RR, W) <= input: (alpha) = (NW)
    kernels[16, 0, 1, 1] = 1
    # output: (action, alpha) = (RR, NW) <= input: (alpha) = (NE)
    kernels[17, 0, 1, 1] = 1

    # output: (action, alpha) = (RL, NE) <= input: (alpha) = (NW)
    kernels[18, 0, 1, 1] = 1
    # output: (action, alpha) = (RL, E) <= input: (alpha) = (NE)
    kernels[19, 0, 1, 1] = 1
    # output: (action, alpha) = (RL, SE) <= input: (alpha) = (E)
    kernels[20, 0, 1, 1] = 1
    # output: (action, alpha) = (RL, SW) <= input: (alpha) = (SE)
    kernels[21, 0, 1, 1] = 1
    # output: (action, alpha) = (RL, W) <= input: (alpha) = (SW)
    kernels[22, 0, 1, 1] = 1
    # output: (action, alpha) = (RL, NW) <= input: (alpha) = (W)
    kernels[23, 0, 1, 1] = 1

    return kernels


def get_vin_predictions(
    vin: Cerealbar_VIN, current_states: List[State], batch: StepBatch,
    allow_player_intersections: bool,
    goal_configurations: List[Optional[Player]]
) -> List[Optional[Set[Action]]]:
    batch_size: int = len(current_states)

    goals: torch.Tensor = torch.zeros(
        (batch_size, len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH),
        device=torch_util.DEVICE)

    default_obstacle_mask: torch.Tensor = batch.environment_state.static_info.obstacle_mask
    if not allow_player_intersections:
        # Also include the leader position
        default_obstacle_mask = batch.environment_state.get_all_obstacles()

    all_static_obstacle_locations: List[List[Position]] = list()

    all_mask_cards: List[bool] = list()
    all_mask_objects: List[bool] = list()

    goal_positions: List[Position] = list()
    shortest_path_lengths: List[int] = list()

    for i in range(batch_size):
        current_game_state: State = current_states[i]
        current_follower: Player = current_game_state.follower

        # Mask the obstacles.
        if goal_configurations[i]:
            argmax_pos: Position = goal_configurations[i].position
            argmax_rot: Rotation = goal_configurations[i].rotation
        else:
            # This is the default state, when the target action is actually STOP.
            argmax_pos: Position = current_follower.position

            # Clockwise one turn
            argmax_rot: Rotation = ROTATIONS[
                (ROTATIONS.index(current_follower.rotation) + 1) %
                len(ROTATIONS)]

        goals[i][ROTATIONS.index(argmax_rot)][argmax_pos.x][argmax_pos.y] = 1

        goal_positions.append(argmax_pos)

        # Add cards to avoid as obstacles. These are all cards except the ones that (1) the agent is currently
        # standing on (if any) and (2) the predicted target card (if any).

        # Get all of the obstacle locations to find the shortest path.
        obstacle_locations: List[Position] = list()
        for x in range(EDGE_WIDTH):
            for y in range(EDGE_WIDTH):
                if default_obstacle_mask[i][x][y] > 0:
                    # This may also include the leader.
                    obstacle_locations.append(Position(x, y))
        all_static_obstacle_locations.append(obstacle_locations)

        avoid_card_locations: List[Position] = list()
        for card in current_game_state.cards:
            if card.position not in {current_follower.position, argmax_pos}:
                avoid_card_locations.append(card.position)

        shortest_path_length, mask_cards, mask_objects = get_vin_settings_for_example(
            current_follower, Player(True, argmax_pos, argmax_rot),
            obstacle_locations, avoid_card_locations)

        shortest_path_lengths.append(shortest_path_length)
        all_mask_cards.append(mask_cards)
        all_mask_objects.append(mask_objects)

    # VIN returns Q-values and argmax actions; just grab the Q-values so we can compute the set of argmaxes ourselves.
    q_values = vin(
        goals,
        torch.cat(
            (batch.environment_state.dynamic_info.current_rotations.long(),
             batch.environment_state.dynamic_info.current_positions),
            dim=1),
        get_vin_obstacles(all_mask_cards, all_mask_objects,
                          default_obstacle_mask, current_states,
                          goal_positions),
        num_iterations=-1)

    argmax_actions: List[Set[Action]] = _get_argmax_actions_for_q(
        batch.original_examples, q_values, goal_configurations,
        all_static_obstacle_locations)

    final_actions: List[Optional[Set[Action]]] = list()
    for i in range(batch_size):
        if goal_configurations[i] is not None:
            final_actions.append(argmax_actions[i])
        else:
            final_actions.append(None)
    return final_actions


def _get_argmax_actions_for_q(
        original_examples: List[StepExample], q_values: torch.Tensor,
        goal_configurations: List[Player],
        obstacle_locations: List[List[Position]]) -> List[Set[Action]]:
    batch_size: int = len(original_examples)
    argmaxes: List[Set[Action]] = list()

    for i in range(batch_size):
        item_q_values: torch.Tensor = q_values[i]
        original_example: StepExample = original_examples[i]
        if torch.sum(item_q_values) == 0:
            # The safest action here is just to rotate.
            argmaxes.append({Action.RR, Action.RL})

            logging.info(
                f'Got completely zero q-values for example ID {original_example.example_id}, step number '
                f'{original_example.step_idx}. Start configuration was {original_example.state.follower}; target '
                f'configuration was {goal_configurations[i]}.')
        else:
            possible_action_q_values: Dict[Action, torch.Tensor] = dict()
            current_agent: Player = original_examples[i].state.follower
            max_q_value: torch.Tensor = torch.max(item_q_values)

            for action, q_value in zip(MOVEMENT_ACTIONS, q_values[i]):
                new_pos, new_rot = get_new_player_orientation(
                    current_agent, action, set())
                if out_of_bounds(new_pos) or new_pos in obstacle_locations[i]:
                    if torch.allclose(q_value, max_q_value):
                        logging.info(
                            f'One of the max q-value actions was inexecutable! Example ID {original_example.example_id}'
                            f'; step ID {original_example.step_idx}; inexecutable action was {action}; q-values were '
                            f'{item_q_values} (max of {max_q_value}; this action had {q_value}).'
                        )
                    continue
                possible_action_q_values[action] = q_value

            # There may be a new max if all if the only max value action was inexecutable (in a very rare case).
            new_max: torch.Tensor = torch.max(
                torch.stack(list(possible_action_q_values.values())))

            max_actions: Set[Action] = set()
            for action, q_value in possible_action_q_values.items():
                if torch.allclose(q_value, new_max):
                    max_actions.add(action)

            if not max_actions:
                logging.info(
                    f'Got no actions with maximum probability; there is probably a bug. Q values: '
                    f'{item_q_values}; possible action q values {possible_action_q_values}; new max: {new_max}. '
                    f'Example ID {original_example.example_id}; step {original_example.step_idx}.'
                )
                max_actions = {Action.RR, Action.RL}
            argmaxes.append(max_actions)
    return argmaxes
