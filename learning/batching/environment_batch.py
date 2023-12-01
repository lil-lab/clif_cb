"""Batches static and dynamic information about the agent's current observation."""
from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass

from util import torch_util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Optional, Union


@dataclass
class StaticEnvironmentBatch:
    """
    Static information about the environment. Each member is size B x 25 x 25 (type long, for embedding lookup, 
    except for obstacles).
    
    Attributes:
        self.prop_types
            The identities of the props (e.g., tree, plant, etc.); also indicates if there is no prop.
        self.hut_colors
            The colors of the huts, if there is a hut in a location. 
        self.hut_rotations
            The rotations of the huts, if there is a hut in a location.
        self.tree_types
            The type of the tree, if there is a tree in that location.
        self.plant_types
            The type of the plant, if there is a plant in that location.
        self.windmill_rotations
            The rotation of the windmill, if there is a windmill in that location.
        self.tent_rotations
            The rotation of the tent, if there is a tent in that location.
        self.terrain    
            The types of terrain on the board.
        self.obstacle_mask
            Indicates where obstacles are on the board. 0 = no obstacle; 1 = obstacle.
        self.format
            The format of the batch (Numpy or Torch tensor).
        self.batched
            Whether this batch actually includes a batch dimension, or if it just represents a single (
            non-batched) example.
    """
    prop_types: Union[torch.Tensor, np.ndarray]
    hut_colors: Union[torch.Tensor, np.ndarray]
    hut_rotations: Union[torch.Tensor, np.ndarray]
    tree_types: Union[torch.Tensor, np.ndarray]
    plant_types: Union[torch.Tensor, np.ndarray]
    windmill_rotations: Union[torch.Tensor, np.ndarray]
    tower_rotations: Union[torch.Tensor, np.ndarray]
    tent_rotations: Union[torch.Tensor, np.ndarray]
    terrain: Union[torch.Tensor, np.ndarray]

    # B x 25 x 25 (float)
    obstacle_mask: Union[torch.Tensor, np.ndarray]

    nonempty_property_mask: Union[torch.Tensor, np.ndarray]

    format: torch_util.TensorType
    batched: bool

    def __str__(self):
        def size(item: Union[torch.Tensor, np.ndarray]):
            if self.format == torch_util.TensorType.TORCH:
                if not isinstance(item, torch.Tensor):
                    raise ValueError(
                        'Batch is tensor format, but item is not.')
                return item.size()
            else:
                if not isinstance(item, np.ndarray):
                    raise ValueError('Batch is numpy format, but item is not.')
                return item.shape

        data: str = f'\n\tprop types ({size(self.prop_types)}) =\n{self.prop_types}' \
                    f'\n\thut colors ({size(self.hut_colors)}) =\n{self.hut_colors}' \
                    f'\n\thut rotations ({size(self.hut_rotations)}) =\n{self.hut_rotations}' \
                    f'\n\ttree types ({size(self.tree_types)}) =\n{self.tree_types}' \
                    f'\n\tplant types ({size(self.plant_types)}) =\n{self.plant_types}' \
                    f'\n\twindmill rotations ({size(self.windmill_rotations)}) =\n{self.windmill_rotations}' \
                    f'\n\ttower rotations ({size(self.tower_rotations)}) =\n{self.tower_rotations}' \
                    f'\n\ttent rotations ({size(self.tent_rotations)}) =\n{self.tent_rotations}' \
                    f'\n\tterrain ({size(self.terrain)}) =\n{self.terrain}' \
                    f'\n\tobstacles ({size(self.obstacle_mask)}) =\n{self.obstacle_mask}'
        return f'Static environment batch (batched={self.batched}) with format {self.format} and data:{data}'

    def to_device(self):
        if self.format == torch_util.TensorType.NUMPY:
            raise ValueError('Cannot move numpy-formatted batch to device.')

        self.prop_types = self.prop_types.to(torch_util.DEVICE)
        self.hut_colors = self.hut_colors.to(torch_util.DEVICE)
        self.hut_rotations = self.hut_rotations.to(torch_util.DEVICE)
        self.tree_types = self.tree_types.to(torch_util.DEVICE)
        self.plant_types = self.plant_types.to(torch_util.DEVICE)
        self.windmill_rotations = self.windmill_rotations.to(torch_util.DEVICE)
        self.tower_rotations = self.tower_rotations.to(torch_util.DEVICE)
        self.tent_rotations = self.tent_rotations.to(torch_util.DEVICE)
        self.terrain = self.terrain.to(torch_util.DEVICE)

        self.nonempty_property_mask = self.nonempty_property_mask.to(
            torch_util.DEVICE)

        self.obstacle_mask = self.obstacle_mask.to(torch_util.DEVICE)

    def to_tensor(self):
        if self.format == torch_util.TensorType.TORCH:
            raise ValueError('Format is already a Torch tensor.')

        return StaticEnvironmentBatch(
            torch.tensor(self.prop_types).long(),
            torch.tensor(self.hut_colors).long(),
            torch.tensor(self.hut_rotations).long(),
            torch.tensor(self.tree_types).long(),
            torch.tensor(self.plant_types).long(),
            torch.tensor(self.windmill_rotations).long(),
            torch.tensor(self.tower_rotations).long(),
            torch.tensor(self.tent_rotations).long(),
            torch.tensor(self.terrain).long(),
            torch.tensor(self.obstacle_mask).float(),
            torch.tensor(self.nonempty_property_mask).float(),
            format=torch_util.TensorType.TORCH,
            batched=self.batched)


@dataclass
class DynamicEnvironmentBatch:
    """Dynamic information about the environment. Each member is size B x 25 x 25 (type long, for embedding lookup, 
    except for observation masks and player locations), except for the current position/rotation.
    
    Attributes:
        self.card_colors
            TThe colors of cards in the environment (plus a null value)
        self.card_counts
            The counts of cards in the environment (plus a null value)
        self.card_shapes
            The shapes of cards in the environment (plus a null value)
        self.card_selections
            The selections of cards in the environment (plus a null value)
        self.leader_location
            Indicates the leader's current location.
        self.leader_rotation
            Indicates the leader's rotation in its current position.
        self.follower_location
            Indicates the follower's current location.
        self.follower_rotation
            Indicates the follower's rotation in its current position.
        self.current_positions
            The current positions of the follower.
        self.current_rotations
            The current rotations of the follower.
        self.format
            The format of the batch (Numpy or Torch tensor).
        self.batched
            Whether this batch actually includes a batch dimension, or if it just represents a single (
            non-batched) example.
    """
    card_counts: Union[torch.Tensor, np.ndarray]
    card_colors: Union[torch.Tensor, np.ndarray]
    card_shapes: Union[torch.Tensor, np.ndarray]
    card_selections: Union[torch.Tensor, np.ndarray]

    prev_visited_card_counts: Union[torch.Tensor, np.ndarray]
    prev_visited_card_colors: Union[torch.Tensor, np.ndarray]
    prev_visited_card_shapes: Union[torch.Tensor, np.ndarray]
    prev_visited_card_selections: Union[torch.Tensor, np.ndarray]

    leader_location: Union[torch.Tensor, np.ndarray]
    leader_rotation: Union[torch.Tensor, np.ndarray]

    follower_location: Union[torch.Tensor, np.ndarray]
    follower_rotation: Union[torch.Tensor, np.ndarray]

    observability_in_memory: Union[torch.Tensor, np.ndarray]
    observability_current: Union[torch.Tensor, np.ndarray]

    # B x 2 (long)
    current_positions: Union[torch.Tensor, np.ndarray]

    # B (float)
    current_rotations: Union[torch.Tensor, np.ndarray]

    nonempty_property_mask: Union[torch.Tensor, np.ndarray]

    format: torch_util.TensorType
    batched: bool

    def __str__(self):
        def size(item: Union[torch.Tensor, np.ndarray]):
            if self.format == torch_util.TensorType.TORCH:
                if not isinstance(item, torch.Tensor):
                    raise ValueError(
                        'Batch is tensor format, but item is not.')
                return item.size()
            else:
                if not isinstance(item, np.ndarray):
                    raise ValueError('Batch is numpy format, but item is not.')
                return item.shape

        data: str = f'\n\tcard counts ({size(self.card_counts)}) =\n{self.card_counts}' \
                    f'\n\tcard counts ({size(self.card_counts)}) =\n{self.card_counts}' \
                    f'\n\tcard colors ({size(self.card_colors)}) =\n{self.card_colors}' \
                    f'\n\tcard selections ({size(self.card_selections)}) =\n{self.card_selections}' \
                    f'\n\tprev visited card counts ({size(self.prev_visited_card_counts)}) =\n{self.prev_visited_card_counts}' \
                    f'\n\tprev visited card counts ({size(self.prev_visited_card_counts)}) =\n{self.prev_visited_card_counts}' \
                    f'\n\tprev visited card colors ({size(self.prev_visited_card_colors)}) =\n{self.prev_visited_card_colors}' \
                    f'\n\tprev visited card selections ({size(self.prev_visited_card_selections)}) =\n{self.prev_visited_card_selections}' \
                    f'\n\tleader locations ({size(self.leader_location)}) =\n{self.leader_location}' \
                    f'\n\tleader rotations ({size(self.leader_rotation)}) =\n{self.leader_rotation}' \
                    f'\n\tfollower locations ({size(self.follower_location)}) =\n{self.follower_location}' \
                    f'\n\tfollower rotations ({size(self.follower_rotation)}) =\n{self.follower_rotation}' \
                    f'\n\tobservability (in memory) ({size(self.observability_in_memory)}) ' \
                    f'=\n{self.observability_in_memory}' \
                    f'\n\tobservability (current) ({size(self.observability_current)}) =\n{self.observability_current}' \
                    f'\n\tcurrent positions ({size(self.current_positions)}) =\n{self.current_positions}' \
                    f'\n\tcurrent rotations ({size(self.current_rotations)}) =\n{self.current_rotations}' \

        return f'Static environment batch (batched={self.batched}) with format {self.format} and data:{data}'

    def to_device(self):
        if self.format == torch_util.TensorType.NUMPY:
            raise ValueError('Cannot move numpy-formatted batch to device.')
        self.card_counts = self.card_counts.to(torch_util.DEVICE)
        self.card_colors = self.card_colors.to(torch_util.DEVICE)
        self.card_shapes = self.card_shapes.to(torch_util.DEVICE)
        self.card_selections = self.card_selections.to(torch_util.DEVICE)

        self.prev_visited_card_counts = self.prev_visited_card_counts.to(
            torch_util.DEVICE)
        self.prev_visited_card_colors = self.prev_visited_card_colors.to(
            torch_util.DEVICE)
        self.prev_visited_card_shapes = self.prev_visited_card_shapes.to(
            torch_util.DEVICE)
        self.prev_visited_card_selections = self.prev_visited_card_selections.to(
            torch_util.DEVICE)

        self.leader_location = self.leader_location.to(torch_util.DEVICE)
        self.leader_rotation = self.leader_rotation.to(torch_util.DEVICE)

        self.follower_location = self.follower_location.to(torch_util.DEVICE)
        self.follower_rotation = self.follower_rotation.to(torch_util.DEVICE)

        self.observability_in_memory = self.observability_in_memory.to(
            torch_util.DEVICE)
        self.observability_current = self.observability_current.to(
            torch_util.DEVICE)

        self.current_positions = self.current_positions.to(torch_util.DEVICE)
        self.current_rotations = self.current_rotations.to(torch_util.DEVICE)

        self.nonempty_property_mask = self.nonempty_property_mask.to(
            torch_util.DEVICE)

    def to_tensor(self):
        if self.format == torch_util.TensorType.TORCH:
            raise ValueError('Format is already a Torch tensor.')
        return DynamicEnvironmentBatch(
            torch.tensor(self.card_counts).long(),
            torch.tensor(self.card_colors).long(),
            torch.tensor(self.card_shapes).long(),
            torch.tensor(self.card_selections).long(),
            torch.tensor(self.prev_visited_card_counts).long(),
            torch.tensor(self.prev_visited_card_colors).long(),
            torch.tensor(self.prev_visited_card_shapes).long(),
            torch.tensor(self.prev_visited_card_selections).long(),
            torch.tensor(self.leader_location).float(),
            torch.tensor(self.leader_rotation).long(),
            torch.tensor(self.follower_location).float(),
            torch.tensor(self.follower_rotation).long(),
            torch.tensor(self.observability_in_memory).float(),
            torch.tensor(self.observability_current).float(),
            torch.tensor(self.current_positions).long(),
            torch.tensor(self.current_rotations).float(),
            torch.tensor(self.nonempty_property_mask).float(),
            format=torch_util.TensorType.TORCH,
            batched=self.batched)


@dataclass
class EnvironmentBatch:
    """
    Contains static and dynamic information about an environment.
    
    Attributes:
        self.static_info: Static environment info, including the prop types and attributes, and terrain.
        self.dynamic_info: Dynamic environment info, including observation masks, player locations, and cards.
        self.previous_visitations
            Previous locations and rotations visited by the agent; each voxel contains the 
            number of times the agent was in that configuration.
    """
    static_info: StaticEnvironmentBatch
    dynamic_info: DynamicEnvironmentBatch

    # B x 6 x 25 x 25 (float; contains integers)
    previous_visitations: torch.Tensor

    # B x 6 x 25 x 25 (float; contains integers)
    all_previous_targets: Optional[torch.Tensor]
    previous_target: Optional[torch.Tensor]

    def to_device(self):
        self.static_info.to_device()
        self.dynamic_info.to_device()

        self.previous_visitations = self.previous_visitations.to(
            torch_util.DEVICE)

        if self.all_previous_targets is not None:
            self.all_previous_targets = self.all_previous_targets.to(
                torch_util.DEVICE)
        if self.previous_target is not None:
            self.previous_target = self.previous_target.to(torch_util.DEVICE)

    def get_all_obstacles(self) -> torch.Tensor:
        return self.static_info.obstacle_mask + self.dynamic_info.leader_location
