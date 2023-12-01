"""A voxel prediction: a distribution over voxels and the STOP action."""
from __future__ import annotations

import numpy as np
import random
import torch

from dataclasses import dataclass
from environment.action import Action, MOVEMENT_ACTIONS
from environment.player import Player
from environment.position import EDGE_WIDTH, Position, out_of_bounds
from environment.rotation import ROTATIONS, Rotation
from simulation import planner
from util import torch_util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, List, Optional, Set, Tuple, Union


@dataclass
class VoxelPredictions:
    """A prediction over voxels and stop actions.
    
    Attributes:
        self.voxel_probabilities
            A tensor in size B x 6 x 25 x 25, where each voxel is the probability of that voxel being a goal location.
        self.stop_probabilities
            A tensor in size B, indicating the probability that the agent should stop.
            
    For each b, voxel_probabilities[b] + stop_probabilities[b] should equal 1.
    """
    voxel_probabilities: Union[torch.Tensor, np.ndarray]
    stop_probabilities: Union[torch.Tensor, np.ndarray]
    copy_probabilities: Optional[Union[torch.Tensor, np.ndarray]]

    global_interpretation: bool

    format: torch_util.TensorType = torch_util.TensorType.TORCH

    q_values: Optional[torch.Tensor] = None

    def to_device(self):
        if self.format != torch_util.TensorType.TORCH:
            raise ValueError(
                'Tensor type must be Torch to move data to device.')

        self.voxel_probabilities = self.voxel_probabilities.to(
            torch_util.DEVICE)
        self.stop_probabilities = self.stop_probabilities.to(torch_util.DEVICE)

    def get_batch_size(self) -> int:
        if self.format == torch_util.TensorType.TORCH:
            assert isinstance(self.voxel_probabilities, torch.Tensor)
            return self.voxel_probabilities.size(0)
        else:
            raise ValueError(
                'Numpy format of voxel predictions should not have a batch size (batch size should be 1 '
                'only).')

    def convert_channels_egocentric_to_global(
            self, rotations: torch.Tensor) -> VoxelPredictions:
        """
        Shuffles channels from an egocentric to global view. 
        """
        assert self.format == torch_util.TensorType.TORCH

        if self.global_interpretation:
            raise ValueError('Voxel prediction already in global view.')

        shifted_tensors: List[torch.Tensor] = list()

        for i, rotation in enumerate(rotations.long().tolist()):
            item_tensor: torch.Tensor = self.voxel_probabilities[i]

            shifted_tensor: torch.Tensor = torch.zeros(
                (len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH),
                device=torch_util.DEVICE)

            # Shifts by subtracting agent's current rotation.
            for j in range(len(ROTATIONS)):
                shifted_tensor[j, :, :] = item_tensor[(j - rotation[0]) %
                                                      len(ROTATIONS), :, :]
            shifted_tensors.append(shifted_tensor.unsqueeze(0))

        return VoxelPredictions(torch.cat(shifted_tensors,
                                          dim=0), self.stop_probabilities,
                                self.copy_probabilities, True, self.format)

    def get_flat_distribution(self, batch_index: int) -> torch.Tensor:
        flattened_voxel_probs: torch.Tensor = self.voxel_probabilities[
            batch_index].view((1, EDGE_WIDTH * EDGE_WIDTH * len(ROTATIONS)))
        return torch.cat(
            (flattened_voxel_probs, self.stop_probabilities[batch_index].view(
                1, 1)),
            dim=1)

    def sample(
        self,
        batch_index: int,
        allow_stop: bool = True,
        obstacle_mask: torch.Tensor = None
    ) -> Tuple[Union[Action, Tuple[Position, Rotation]], float]:
        assert self.format == torch_util.TensorType.TORCH
        assert 0 <= batch_index < self.get_batch_size()

        voxel_probabilities = self.voxel_probabilities
        if obstacle_mask is not None:
            voxel_probabilities = voxel_probabilities * (
                1 - obstacle_mask.unsqueeze(1).repeat(
                    (1, len(ROTATIONS), 1, 1)))

        sample_prob: float = random.random()

        flattened_voxel_probs: torch.Tensor = voxel_probabilities[
            batch_index].view((-1, ))
        stop_prob: torch.Tensor = self.stop_probabilities[batch_index]

        if sample_prob <= stop_prob and allow_stop:
            return Action.STOP, stop_prob.item()

        base_prob = stop_prob.item()
        max_voxel_prob = -1
        max_voxel_pos = -1

        for idx, prob in enumerate(flattened_voxel_probs.tolist()):
            base_prob += prob
            if sample_prob <= base_prob:
                max_voxel_prob = prob
                max_voxel_pos = idx
                break

        # Find the maximum voxel orientation from the flattened index
        rot_idx, x, y = np.unravel_index(
            max_voxel_pos, (len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH))

        return (Position(x, y), ROTATIONS[rot_idx]), max_voxel_prob

    def argmax(
        self,
        batch_index: int,
        allow_stop: bool = True,
        obstacle_mask: torch.Tensor = None,
        allow_copy: bool = False,
    ) -> Tuple[Union[Action, Tuple[Position, Rotation]], float]:
        assert self.format == torch_util.TensorType.TORCH
        assert 0 <= batch_index < self.get_batch_size()

        voxel_probabilities = self.voxel_probabilities[batch_index]
        if obstacle_mask is not None:
            h, w = obstacle_mask.size()
            if h != EDGE_WIDTH or w != EDGE_WIDTH:
                raise ValueError(
                    f'Obstacle mask must be in size (25, 25) but was size {obstacle_mask.size()}'
                )

            voxel_probabilities = voxel_probabilities * (
                1 - obstacle_mask.unsqueeze(0).repeat((len(ROTATIONS), 1, 1)))

        flattened_voxel_probs: torch.Tensor = voxel_probabilities.view((-1, ))
        stop_prob: torch.Tensor = self.stop_probabilities[batch_index]

        max_voxel_prob, max_voxel_pos = torch.max(flattened_voxel_probs, dim=0)

        copy_prob: float = 0.
        if self.copy_probabilities is not None:
            copy_prob: torch.Tensor = self.copy_probabilities[batch_index]

        if allow_stop and stop_prob > max_voxel_prob and (stop_prob > copy_prob
                                                          or not allow_copy):
            # If allowing stop, and stop is better than the max voxel prob (and, unless copy is disallowed,
            # higher than the max copy prob), just stop.
            return Action.STOP, stop_prob.item()
        elif allow_copy and copy_prob > max_voxel_prob:
            # If allowed to copy, and copy prob is higher than the max voxel prob, return it.
            return Action.COPY, copy_prob.item()

        # Find the maximum voxel orientation from the flattened index
        rot_idx, x, y = np.unravel_index(
            max_voxel_pos.item(), (len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH))

        return (Position(x, y), ROTATIONS[rot_idx]), max_voxel_prob.item()

    def get_top_k_voxels(
            self, batch_index: int, k: int, obstacle_mask: torch.Tensor
    ) -> Dict[Tuple[Position, Rotation], float]:
        # ONLY returns voxels, not STOP.
        h, w = obstacle_mask.size()
        if h != EDGE_WIDTH or w != EDGE_WIDTH:
            raise ValueError(
                f'Size of obstacle mask not expected: {obstacle_mask.size()}')

        voxel_probs: torch.Tensor = self.voxel_probabilities[batch_index] * (
            1 - obstacle_mask.unsqueeze(0).repeat(len(ROTATIONS), 1, 1))

        flattened_voxel_probs: torch.Tensor = voxel_probs.view((-1, 1))

        # Get the top-k voxels
        top_k_voxels: Dict[Union[Action, Tuple[Position, Rotation]],
                           float] = dict()

        for i in range(k):
            max_voxel_prob, max_voxel_idx = torch.max(flattened_voxel_probs,
                                                      dim=0)
            rot_idx, x, y = np.unravel_index(
                max_voxel_idx.item(), (len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH))

            # Set to zero here so the next argmax is different
            flattened_voxel_probs[max_voxel_idx] = 0.

            voxel: Tuple[Position,
                         Rotation] = (Position(x, y), ROTATIONS[rot_idx])
            top_k_voxels[voxel] = max_voxel_prob.item()

        return top_k_voxels

    def argmax_position(self, batch_index: int,
                        can_copy: bool) -> Union[Position, Action]:
        """
        Returns the argmax position in the map by summing over rotations. Returns a position *even if* the argmax 
        is STOP.
        """
        assert self.format == torch_util.TensorType.TORCH
        assert 0 <= batch_index < self.get_batch_size()

        argmax: Union[Action, Player] = self.argmax(batch_index,
                                                    allow_copy=can_copy)
        if argmax == Action.STOP:
            return Action.STOP
        elif argmax == Action.COPY:
            return Action.COPY

        voxel_probs: torch.Tensor = self.voxel_probabilities[batch_index]

        # Should be size 25 x 25
        position_probs: torch.Tensor = torch.sum(voxel_probs, dim=0)

        _, pos = torch.max(position_probs.view((-1, )), dim=0)

        x, y = np.unravel_index(pos.item(), (EDGE_WIDTH, EDGE_WIDTH))

        return Position(x, y)

    def argmax_neighbor_action(
            self, batch_index: int, current_agent: Player,
            obstacle_positions: Set[Position],
            can_copy: bool) -> Tuple[Action, float, Optional[Player]]:
        """
        Considers only the voxels that are reachable from the current state. Returns the one with the highest 
        probability. (If STOP has the highest probability, return that.)
        
        TODO: write a different function that gives batched results for probabilities of all actions, and use that 
        function in this one
        """
        assert self.format == torch_util.TensorType.TORCH
        assert 0 <= batch_index < self.get_batch_size()

        action_probabilities: Dict[Action, float] = dict()
        action_resulting_configurations: Dict[Action, Player] = dict()
        for action in MOVEMENT_ACTIONS:
            next_pos, next_rot = planner.get_new_player_orientation(
                current_agent, action, set())

            if next_pos in obstacle_positions or out_of_bounds(next_pos):
                continue

            action_probabilities[action] = self.voxel_probabilities[
                batch_index][ROTATIONS.index(next_rot)][next_pos.x][
                    next_pos.y].item()

            action_resulting_configurations[action] = Player(
                True, next_pos, next_rot)

        obstacle_mask: torch.Tensor = torch.zeros((EDGE_WIDTH, EDGE_WIDTH),
                                                  device=torch_util.DEVICE)
        for pos in obstacle_positions:
            obstacle_mask[pos.x][pos.y] = 1

        argmax: Union[Action,
                      Player] = self.argmax(batch_index,
                                            obstacle_mask=obstacle_mask,
                                            allow_copy=can_copy)[0]
        if argmax == Action.STOP:
            return Action.STOP, self.stop_probabilities[batch_index].item(
            ), None
        elif argmax == Action.COPY:
            return Action.COPY, self.copy_probabilities[batch_index].item(
            ), None

        sorted_actions_probs: List[Tuple[Action, float]] = sorted(
            action_probabilities.items(), key=lambda x: x[1])

        best_action: Action = sorted_actions_probs[-1][0]
        best_action_prob: float = sorted_actions_probs[-1][1]

        return best_action, best_action_prob, action_resulting_configurations[
            best_action]

    def off_graph(self, batch_index: int) -> VoxelPredictions:
        assert self.format == torch_util.TensorType.TORCH
        assert 0 <= batch_index < self.get_batch_size()

        return VoxelPredictions(
            self.voxel_probabilities[batch_index].numpy(),
            self.stop_probabilities[batch_index].numpy(),
            self.copy_probabilities[batch_index].numpy()
            if self.copy_probabilities else None, self.global_interpretation,
            torch_util.TensorType.NUMPY)

    def get_q_value(self, batch_index) -> torch.Tensor:
        # Returns the q-value for this item in the batch.
        assert 0 <= batch_index < self.get_batch_size()
        assert self.q_values is not None

        return self.q_values[batch_index]
