"""
Predicted distribution over actions; analogoust to PredictedVoxel but without any probabilities on the voxel 
space.
"""
from __future__ import annotations

import logging
import torch

from dataclasses import dataclass

from environment.action import Action, MOVEMENT_ACTIONS
from environment.position import out_of_bounds
from simulation import planner
from util import torch_util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Optional, Set, Tuple
    from environment.player import Player
    from environment.position import Position

PREDICTABLE_ACTIONS: List[Action] = MOVEMENT_ACTIONS + [Action.STOP]


@dataclass
class ActionPredictions:
    """Distributions over actions.
    
    Attributes:
        self.action_probabilities
            probabilities over all 5 predictable actions. Size: B x N, where N is the number of actions.
    """
    action_probabilities: torch.Tensor

    format: torch_util.TensorType = torch_util.TensorType.TORCH

    def to_device(self):
        if self.format != torch_util.TensorType.TORCH:
            raise ValueError(
                'Tensor type must be Torch to move data to device.')

        self.action_probabilities = self.action_probabilities.to(
            torch_util.DEVICE)

    def get_batch_size(self) -> int:
        return self.action_probabilities.size(0)

    def argmax(self,
               batch_index: int,
               current_agent: Optional[Player] = None,
               obstacle_positions: Optional[Set[Position]] = None,
               sample: bool = False) -> Tuple[Action, float]:
        assert self.format == torch_util.TensorType.TORCH
        assert 0 <= batch_index < self.get_batch_size()

        assert (current_agent is None) == (obstacle_positions is None)

        action_mask: torch.Tensor = torch.ones(len(PREDICTABLE_ACTIONS),
                                               device=torch_util.DEVICE)

        possible_actions: Set[Action] = {Action.STOP}
        if obstacle_positions:
            # Create a mask for actions.
            assert current_agent is not None

            for action in MOVEMENT_ACTIONS:
                next_pos, next_rot = planner.get_new_player_orientation(
                    current_agent, action, set())

                if next_pos in obstacle_positions or out_of_bounds(next_pos):
                    # Mask out this action.
                    action_mask[PREDICTABLE_ACTIONS.index(action)] = 0.
                else:
                    possible_actions.add(action)

        this_example_probabilities: torch.Tensor = self.action_probabilities[
            batch_index] * action_mask

        if sample:
            max_idx = torch.multinomial(this_example_probabilities, 1).item()
        else:
            max_prob, max_idx = torch.max(this_example_probabilities, dim=0)

        max_action: Action = PREDICTABLE_ACTIONS[int(max_idx)]
        if max_action not in possible_actions and obstacle_positions:
            # Will only happen if 100% of the probability mass is on an inexecutable action.
            logging.info(
                f'{max_action} was not possible (distribution is {self.action_probabilities[batch_index]}); replacing with RR.'
            )
            max_action = Action.RR

        return max_action, this_example_probabilities[
            PREDICTABLE_ACTIONS.index(max_action)].item()

    def off_graph(self, batch_index: int):
        assert self.format == torch_util.TensorType.TORCH
        assert 0 <= batch_index < self.get_batch_size()

        return ActionPredictions(
            self.action_probabilities[batch_index].numpy(),
            format=torch_util.TensorType.NUMPY)
