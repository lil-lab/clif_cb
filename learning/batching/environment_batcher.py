"""Utility class for batching environment information."""
from __future__ import annotations

import numpy as np
import torch

from environment import observation
from environment.card import Card, CardColor, CardShape, CardCount, CardSelection
from environment.position import EDGE_WIDTH, Position
from environment.rotation import ROTATIONS, Rotation
from environment.static_props.hut import HutColor
from environment.static_props.plant import PlantType
from environment.static_props.prop_types import PropType
from environment.static_props.tree import TreeType
from environment.terrain import Terrain, OBSTACLE_TERRAINS
from learning.batching.environment_batch import StaticEnvironmentBatch, DynamicEnvironmentBatch, \
    EnvironmentBatch
from util import torch_util

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.dataset import GamesCollection
    from data.step_example import StepExample
    from environment.player import Player
    from environment.static_environment import StaticEnvironment
    from typing import List, Optional, Tuple

EMPTY_STR: str = '_NONE'


def _new_environment_ndarray(value: int) -> np.ndarray:
    return np.full((EDGE_WIDTH, EDGE_WIDTH), value, dtype=int)


def _combine_prop(elems, fn) -> torch.Tensor:
    return torch.cat([fn(elem).unsqueeze(0) for elem in elems], dim=0)


def _batch_static_individual_batches(
        elements: List[StaticEnvironmentBatch]) -> StaticEnvironmentBatch:
    """ Inputs should be torch tensors already. """
    for elem in elements:
        if elem.batched:
            raise ValueError(
                'Inputs to batching should not be batched already.')
        if elem.format != torch_util.TensorType.TORCH:
            raise ValueError(
                'Inputs to batching should already be in tensor format.')

    return StaticEnvironmentBatch(
        prop_types=_combine_prop(elements, lambda x: x.prop_types),
        hut_colors=_combine_prop(elements, lambda x: x.hut_colors),
        hut_rotations=_combine_prop(elements, lambda x: x.hut_rotations),
        tree_types=_combine_prop(elements, lambda x: x.tree_types),
        plant_types=_combine_prop(elements, lambda x: x.plant_types),
        windmill_rotations=_combine_prop(elements,
                                         lambda x: x.windmill_rotations),
        tent_rotations=_combine_prop(elements, lambda x: x.tent_rotations),
        tower_rotations=_combine_prop(elements, lambda x: x.tower_rotations),
        terrain=_combine_prop(elements, lambda x: x.terrain),
        obstacle_mask=_combine_prop(elements, lambda x: x.obstacle_mask),
        nonempty_property_mask=_combine_prop(
            elements, lambda x: x.nonempty_property_mask),
        format=torch_util.TensorType.TORCH,
        batched=True)


def _batch_dynamic_individual_batches(
        elements: List[DynamicEnvironmentBatch]) -> DynamicEnvironmentBatch:
    """Inputs should be torch tensors already."""
    for elem in elements:
        if elem.batched:
            raise ValueError(
                'Inputs to batching should not be batched already.')
        if elem.format != torch_util.TensorType.TORCH:
            raise ValueError(
                'Inputs to batching should already be in tensor format.')

    return DynamicEnvironmentBatch(
        card_counts=_combine_prop(elements, lambda x: x.card_counts),
        card_colors=_combine_prop(elements, lambda x: x.card_colors),
        card_shapes=_combine_prop(elements, lambda x: x.card_shapes),
        card_selections=_combine_prop(elements, lambda x: x.card_selections),
        prev_visited_card_counts=_combine_prop(
            elements, lambda x: x.prev_visited_card_counts),
        prev_visited_card_colors=_combine_prop(
            elements, lambda x: x.prev_visited_card_colors),
        prev_visited_card_shapes=_combine_prop(
            elements, lambda x: x.prev_visited_card_shapes),
        prev_visited_card_selections=_combine_prop(
            elements, lambda x: x.prev_visited_card_selections),
        leader_location=_combine_prop(elements, lambda x: x.leader_location),
        leader_rotation=_combine_prop(elements, lambda x: x.leader_rotation),
        follower_location=_combine_prop(elements,
                                        lambda x: x.follower_location),
        follower_rotation=_combine_prop(elements,
                                        lambda x: x.follower_rotation),
        observability_in_memory=_combine_prop(
            elements, lambda x: x.observability_in_memory),
        observability_current=_combine_prop(elements,
                                            lambda x: x.observability_current),
        current_positions=_combine_prop(elements,
                                        lambda x: x.current_positions),
        current_rotations=_combine_prop(elements,
                                        lambda x: x.current_rotations),
        nonempty_property_mask=_combine_prop(
            elements, lambda x: x.nonempty_property_mask),
        format=torch_util.TensorType.TORCH,
        batched=True)


def _batch_previous_visitations(steps: List[StepExample]) -> torch.Tensor:
    visitations: torch.Tensor = torch.zeros(
        (len(steps), len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH),
        device=torch_util.DEVICE)

    # Simply takes the sum of the number of the times the agent has been in each position
    for i, step in enumerate(steps):
        for configuration in step.previous_configurations:
            position: Position = configuration.position
            visitations[i][ROTATIONS.index(
                configuration.rotation)][position.x][position.y] += 1

    return visitations


def _batch_previous_targets(
        steps: List[StepExample]) -> Tuple[torch.Tensor, torch.Tensor]:
    all_prev: torch.Tensor = torch.zeros(
        (len(steps), len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH),
        device=torch_util.DEVICE)
    prev: torch.Tensor = torch.zeros(
        (len(steps), len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH),
        device=torch_util.DEVICE)

    # Simply takes the sum of the number of the times the agent has made each position a target
    for i, step in enumerate(steps):
        if len(step.previous_targets) != step.step_idx:
            raise ValueError(
                f'Step at index {step.step_idx} should have only {step.step_idx} previous targets; '
                f'got {len(step.previous_targets)} instead.')

        for configuration in step.previous_targets:
            position: Position = configuration.position
            all_prev[i][ROTATIONS.index(
                configuration.rotation)][position.x][position.y] += 1

        if step.previous_targets:
            # Set the actual previous target here.
            previous_target: Player = step.previous_targets[-1]
            prev[i][ROTATIONS.index(previous_target.rotation)][
                previous_target.position.x][previous_target.position.y] = 1

    return all_prev, prev


class EnvironmentBatcher:
    """Used for batching environments."""
    def __init__(self, use_previous_cards_in_input: bool):
        super(EnvironmentBatcher, self).__init__()

        # DYNAMIC INFO: embeddings for cards, leader/follower rotations
        # Cards: colors, shapes, counts, and selections.
        self._card_color_indices: List[str] = [EMPTY_STR] + sorted(
            [str(color) for color in CardColor])
        self._card_shape_indices: List[str] = [EMPTY_STR] + sorted(
            [str(shape) for shape in CardShape])
        self._card_count_indices: List[str] = [EMPTY_STR] + sorted(
            [str(count) for count in CardCount])
        self._card_selection_indices: List[str] = [EMPTY_STR] + sorted(
            [str(selection) for selection in CardSelection])

        # Previously-visited card
        self._prev_visited_card_color_indices: List[str] = [
            EMPTY_STR
        ] + sorted([str(color) for color in CardColor])
        self._prev_visited_card_shape_indices: List[str] = [
            EMPTY_STR
        ] + sorted([str(shape) for shape in CardShape])
        self._prev_visited_card_count_indices: List[str] = [
            EMPTY_STR
        ] + sorted([str(count) for count in CardCount])
        self._prev_visited_card_selection_indices: List[str] = [
            EMPTY_STR
        ] + sorted([str(selection) for selection in CardSelection])
        self._use_previous_cards_in_input: bool = use_previous_cards_in_input

        # Leader and follower rotations.
        self._leader_rotation_indices: List[str] = \
            [EMPTY_STR] + sorted([str(rot) for rot in Rotation])

        self._follower_rotation_indices: List[str] = \
            [EMPTY_STR] + sorted([str(rot) for rot in Rotation])

        # STATIC INFO: embeddings for terrain and static props
        self._terrain_indices: List[str] = sorted(
            [str(ter) for ter in Terrain])
        self._hut_color_indices: List[str] = [EMPTY_STR] + sorted(
            [str(color) for color in HutColor])
        self._hut_rotation_indices: List[str] = [EMPTY_STR] + sorted(
            [str(rot) for rot in Rotation])
        self._tree_type_indices: List[str] = [EMPTY_STR] + sorted(
            [str(obj) for obj in TreeType])
        self._plant_type_indices: List[str] = [EMPTY_STR] + sorted(
            [str(obj) for obj in PlantType])
        self._windmill_rotation_indices: List[str] = [EMPTY_STR] + sorted(
            [str(rot) for rot in Rotation])
        self._tower_rotation_indices: List[str] = [EMPTY_STR] + sorted(
            [str(rot) for rot in Rotation])
        self._tent_rotation_indices: List[str] = [EMPTY_STR] + sorted(
            [str(rot) for rot in Rotation])
        self._prop_type_indices: List[str] = \
            ([EMPTY_STR] + [str(prop_type) for prop_type in PropType])

        self._verify_indices()

    def _verify_indices(self):
        assert len(self._card_color_indices) == len(
            set(self._card_color_indices))
        assert len(self._card_shape_indices) == len(
            set(self._card_shape_indices))
        assert len(self._card_count_indices) == len(
            set(self._card_count_indices))
        assert len(self._card_selection_indices) == len(
            set(self._card_selection_indices))
        assert len(self._prev_visited_card_color_indices) == len(
            set(self._prev_visited_card_color_indices))
        assert len(self._prev_visited_card_shape_indices) == len(
            set(self._prev_visited_card_shape_indices))
        assert len(self._prev_visited_card_count_indices) == len(
            set(self._prev_visited_card_count_indices))
        assert len(self._prev_visited_card_selection_indices) == len(
            set(self._prev_visited_card_selection_indices))
        assert len(self._leader_rotation_indices) == len(
            set(self._leader_rotation_indices))
        assert len(self._follower_rotation_indices) == len(
            set(self._follower_rotation_indices))
        assert len(self._hut_rotation_indices) == len(
            set(self._hut_rotation_indices))
        assert len(self._windmill_rotation_indices) == len(
            set(self._windmill_rotation_indices))
        assert len(self._tent_rotation_indices) == len(
            set(self._tent_rotation_indices))
        assert len(self._tower_rotation_indices) == len(
            set(self._tower_rotation_indices))
        assert len(self._hut_color_indices) == len(set(
            self._hut_color_indices))
        assert len(self._terrain_indices) == len(set(self._terrain_indices))
        assert len(self._prop_type_indices) == len(set(
            self._prop_type_indices))
        assert len(self._plant_type_indices) == len(
            set(self._plant_type_indices))
        assert len(self._tree_type_indices) == len(set(
            self._tree_type_indices))

    def _batch_game_info(
            self, static_info: StaticEnvironment) -> StaticEnvironmentBatch:

        # Empty np arrays
        prop_types: np.ndarray = _new_environment_ndarray(
            self._prop_type_indices.index(EMPTY_STR))
        hut_colors: np.ndarray = _new_environment_ndarray(
            self._hut_color_indices.index(EMPTY_STR))
        hut_rotations: np.ndarray = _new_environment_ndarray(
            self._hut_rotation_indices.index(EMPTY_STR))
        tree_types: np.ndarray = _new_environment_ndarray(
            self._tree_type_indices.index(EMPTY_STR))
        plant_types: np.ndarray = _new_environment_ndarray(
            self._plant_type_indices.index(EMPTY_STR))
        windmill_rotations: np.ndarray = _new_environment_ndarray(
            self._windmill_rotation_indices.index(EMPTY_STR))
        tower_rotations: np.ndarray = _new_environment_ndarray(
            self._tower_rotation_indices.index(EMPTY_STR))
        tent_rotations: np.ndarray = _new_environment_ndarray(
            self._tent_rotation_indices.index(EMPTY_STR))
        terrains: np.ndarray = _new_environment_ndarray(
            self._terrain_indices.index(str(Terrain.GRASS)))

        obstacles: np.ndarray = _new_environment_ndarray(0)

        # Everything except terrain and obstacles.
        nonempty_masks: List[np.ndarray] = [
            _new_environment_ndarray(0) for _ in range(8)
        ]

        # Terrain
        for position, terrain in static_info.terrain.items():
            terrains[position.x][position.y] = self._terrain_indices.index(
                str(terrain))

            if terrain in OBSTACLE_TERRAINS:
                obstacles[position.x][position.y] = 1

        # Props: huts, plants, trees, windmills, tents, towers, lampposts
        for hut in static_info.huts:
            position: Position = hut.position
            hut_colors[position.x][position.y] = self._hut_color_indices.index(
                str(hut.color))
            hut_rotations[position.x][
                position.y] = self._hut_rotation_indices.index(
                    str(hut.rotation))
            prop_types[position.x][position.y] = self._prop_type_indices.index(
                str(PropType.HUT))
            obstacles[position.x][position.y] = 1

            nonempty_masks[0][position.x][position.y] = 1
            nonempty_masks[1][position.x][position.y] = 1
            nonempty_masks[2][position.x][position.y] = 1

        for tree in static_info.trees:
            position: Position = tree.position
            tree_types[position.x][position.y] = self._tree_type_indices.index(
                str(tree.tree_type))
            prop_types[position.x][position.y] = self._prop_type_indices.index(
                str(PropType.TREE))
            obstacles[position.x][position.y] = 1
            nonempty_masks[0][position.x][position.y] = 1
            nonempty_masks[3][position.x][position.y] = 1

        for plant in static_info.plants:
            position: Position = plant.position
            plant_types[position.x][
                position.y] = self._plant_type_indices.index(
                    str(plant.plant_type))
            prop_types[position.x][position.y] = self._prop_type_indices.index(
                str(PropType.PLANT))
            obstacles[position.x][position.y] = 1

            nonempty_masks[0][position.x][position.y] = 1
            nonempty_masks[4][position.x][position.y] = 1

        for windmill in static_info.windmills:
            position: Position = windmill.position
            windmill_rotations[position.x][
                position.y] = self._windmill_rotation_indices.index(
                    str(windmill.rotation))
            prop_types[position.x][position.y] = self._prop_type_indices.index(
                str(PropType.WINDMILL))
            obstacles[position.x][position.y] = 1

            nonempty_masks[0][position.x][position.y] = 1
            nonempty_masks[5][position.x][position.y] = 1

        for tower in static_info.towers:
            position: Position = tower.position
            tower_rotations[position.x][
                position.y] = self._tower_rotation_indices.index(
                    str(tower.rotation))
            prop_types[position.x][position.y] = self._prop_type_indices.index(
                str(PropType.TOWER))
            obstacles[position.x][position.y] = 1

            nonempty_masks[0][position.x][position.y] = 1
            nonempty_masks[6][position.x][position.y] = 1

        for tent in static_info.tents:
            position: Position = tent.position
            tent_rotations[position.x][
                position.y] = self._tent_rotation_indices.index(
                    str(tent.rotation))
            prop_types[position.x][position.y] = self._prop_type_indices.index(
                str(PropType.TENT))
            obstacles[position.x][position.y] = 1

            nonempty_masks[0][position.x][position.y] = 1
            nonempty_masks[7][position.x][position.y] = 1

        for lamppost in static_info.lampposts:
            position: Position = lamppost.position
            prop_types[position.x][position.y] = self._prop_type_indices.index(
                str(PropType.LAMPPOST))
            obstacles[position.x][position.y] = 1

        combined_mask = np.stack(nonempty_masks)

        return StaticEnvironmentBatch(prop_types,
                                      hut_colors,
                                      hut_rotations,
                                      tree_types,
                                      plant_types,
                                      windmill_rotations,
                                      tower_rotations,
                                      tent_rotations,
                                      terrains,
                                      obstacles,
                                      combined_mask,
                                      format=torch_util.TensorType.NUMPY,
                                      batched=False)

    def _batch_games(self, data: List[StepExample],
                     games: GamesCollection) -> StaticEnvironmentBatch:

        # These are *numpy* batches.
        individual_batches: List[StaticEnvironmentBatch] = list()

        for example in data:
            game_id: str = example.game_id

            if game_id in games.cached_indices:
                # Just grab the one from the cache.
                individual_batches.append(games.cached_indices[game_id])
            else:
                # Need to create a batch from this example.
                new_batch: StaticEnvironmentBatch = self._batch_game_info(
                    games.games[game_id].environment)

                # Put it in the cache.
                games.cached_indices[game_id] = new_batch

                individual_batches.append(new_batch)

        tensor_batches: List[StaticEnvironmentBatch] = [
            batch.to_tensor() for batch in individual_batches
        ]

        return _batch_static_individual_batches(tensor_batches)

    def _batch_dynamic_step(self,
                            step: StepExample) -> DynamicEnvironmentBatch:
        # Empty np arrays
        card_counts: np.ndarray = _new_environment_ndarray(
            self._card_count_indices.index(EMPTY_STR))
        card_colors: np.ndarray = _new_environment_ndarray(
            self._card_color_indices.index(EMPTY_STR))
        card_shapes: np.ndarray = _new_environment_ndarray(
            self._card_shape_indices.index(EMPTY_STR))
        card_selections: np.ndarray = _new_environment_ndarray(
            self._card_selection_indices.index(EMPTY_STR))

        prev_visited_card_counts: np.ndarray = _new_environment_ndarray(
            self._prev_visited_card_count_indices.index(EMPTY_STR))
        prev_visited_card_colors: np.ndarray = _new_environment_ndarray(
            self._prev_visited_card_color_indices.index(EMPTY_STR))
        prev_visited_card_shapes: np.ndarray = _new_environment_ndarray(
            self._prev_visited_card_shape_indices.index(EMPTY_STR))
        prev_visited_card_selections: np.ndarray = _new_environment_ndarray(
            self._prev_visited_card_selection_indices.index(EMPTY_STR))

        leader_location: np.ndarray = _new_environment_ndarray(0)
        leader_rotation: np.ndarray = _new_environment_ndarray(
            self._leader_rotation_indices.index(EMPTY_STR))

        follower_location: np.ndarray = _new_environment_ndarray(0)
        follower_rotation: np.ndarray = _new_environment_ndarray(
            self._follower_rotation_indices.index(EMPTY_STR))

        if step.observation.fully_observable:
            # All 1s: not masking anything out.
            observability_in_memory: torch.Tensor = _new_environment_ndarray(1)
        else:
            observability_in_memory: torch.Tensor = _new_environment_ndarray(0)
        observability_current: torch.Tensor = _new_environment_ndarray(0)

        # Cards: put all *believed* cards here.
        if step.observation.fully_observable:
            # Get the true cards on the board.
            believed_cards: List[Card] = step.state.cards
        else:
            believed_cards: List[Card] = step.observation.believed_cards

        nonempty_masks: List[np.ndarray] = [
            # Default is that there is nothing in the location.
            # A total of 10 properties: 4 card properties (previous and current) and 2 rotation properties.
            _new_environment_ndarray(0) for _ in range(4 + 4 + 2)
        ]

        for card in believed_cards:
            position: Position = card.position
            card_counts[position.x][
                position.y] = self._card_count_indices.index(str(card.count))
            card_colors[position.x][
                position.y] = self._card_color_indices.index(str(card.color))
            card_shapes[position.x][
                position.y] = self._card_shape_indices.index(str(card.shape))
            card_selections[position.x][
                position.y] = self._card_selection_indices.index(
                    str(card.selection))

            # First 4 properties are for cards: put a value of 1 here.
            nonempty_masks[0][position.x][position.y] = 1
            nonempty_masks[1][position.x][position.y] = 1
            nonempty_masks[2][position.x][position.y] = 1
            nonempty_masks[3][position.x][position.y] = 1

        for prev_visited_card in step.previously_visited_cards:
            position: Position = prev_visited_card.position
            prev_visited_card_counts[position.x][
                position.y] = self._prev_visited_card_count_indices.index(
                    str(prev_visited_card.count))
            prev_visited_card_colors[position.x][
                position.y] = self._prev_visited_card_color_indices.index(
                    str(prev_visited_card.color))
            prev_visited_card_shapes[position.x][
                position.y] = self._prev_visited_card_shape_indices.index(
                    str(prev_visited_card.shape))
            prev_visited_card_selections[position.x][
                position.y] = self._prev_visited_card_selection_indices.index(
                    str(prev_visited_card.selection))

            # Last 4 properties are for previous cards: put a value of 1 here.
            nonempty_masks[6][position.x][position.y] = 1
            nonempty_masks[7][position.x][position.y] = 1
            nonempty_masks[8][position.x][position.y] = 1
            nonempty_masks[9][position.x][position.y] = 1

        # Believed leader position
        if step.observation.fully_observable:
            # Get the true leader here.
            believed_leader: Player = step.state.leader
        else:
            believed_leader: Optional[
                Player] = step.observation.believed_leader

        if believed_leader:
            leader_position: Position = believed_leader.position
            leader_location[leader_position.x][leader_position.y] = 1
            leader_rotation[leader_position.x][
                leader_position.y] = self._leader_rotation_indices.index(
                    str(believed_leader.rotation))

            nonempty_masks[4] = leader_location

        # Actual follower (we know where it is)
        follower: Player = step.state.follower
        follower_position: Position = follower.position
        follower_location[follower_position.x][follower_position.y] = 1
        nonempty_masks[5] = follower_location

        follower_rotation[follower_position.x][
            follower_position.y] = self._follower_rotation_indices.index(
                str(follower.rotation))
        current_positions = np.array(
            [follower_position.x, follower_position.y])
        current_rotations = np.array([follower.rotation.to_radians()])

        # Observations
        if not step.observation.fully_observable:
            for position in step.observation.get_positions_in_memory():
                observability_in_memory[position.x][position.y] = 1

        for position in observation.get_player_observed_positions(follower):
            # Even if fully observable, give the agent a mask of what's in it's current field of view.
            observability_current[position.x][position.y] = 1

        combined_mask = np.stack(nonempty_masks)

        return DynamicEnvironmentBatch(card_counts,
                                       card_colors,
                                       card_shapes,
                                       card_selections,
                                       prev_visited_card_counts,
                                       prev_visited_card_colors,
                                       prev_visited_card_shapes,
                                       prev_visited_card_selections,
                                       leader_location,
                                       leader_rotation,
                                       follower_location,
                                       follower_rotation,
                                       observability_in_memory,
                                       observability_current,
                                       current_positions,
                                       current_rotations,
                                       combined_mask,
                                       format=torch_util.TensorType.NUMPY,
                                       batched=False)

    def _batch_dynamic_info(
            self, data: List[StepExample]) -> DynamicEnvironmentBatch:
        # Get individual *numpy* batches.
        individual_batches: List[DynamicEnvironmentBatch] = list()
        for example in data:
            new_batch: DynamicEnvironmentBatch = self._batch_dynamic_step(
                example)

            individual_batches.append(new_batch)

        tensor_batches: List[DynamicEnvironmentBatch] = [
            batch.to_tensor() for batch in individual_batches
        ]

        return _batch_dynamic_individual_batches(tensor_batches)

    def get_prop_types(self) -> List[str]:
        return self._prop_type_indices

    def get_tree_types(self) -> List[str]:
        return self._tree_type_indices

    def get_plant_types(self) -> List[str]:
        return self._plant_type_indices

    def get_hut_rotations(self) -> List[str]:
        return self._hut_rotation_indices

    def get_hut_colors(self) -> List[str]:
        return self._hut_color_indices

    def get_windmill_rotations(self) -> List[str]:
        return self._windmill_rotation_indices

    def get_tower_rotations(self) -> List[str]:
        return self._tower_rotation_indices

    def get_tent_rotations(self) -> List[str]:
        return self._tent_rotation_indices

    def get_terrains(self) -> List[str]:
        return self._terrain_indices

    def get_card_counts(self) -> List[str]:
        return self._card_count_indices

    def get_card_colors(self) -> List[str]:
        return self._card_color_indices

    def get_card_shapes(self) -> List[str]:
        return self._card_shape_indices

    def get_card_selections(self) -> List[str]:
        return self._card_selection_indices

    def get_leader_rotations(self) -> List[str]:
        return self._leader_rotation_indices

    def get_follower_rotations(self) -> List[str]:
        return self._follower_rotation_indices

    def use_previous_cards_in_input(self) -> bool:
        return self._use_previous_cards_in_input

    def get_num_static_embeddings(self) -> int:
        return len(self._prop_type_indices) + len(
            self._hut_rotation_indices) + len(self._hut_color_indices) + len(
                self._terrain_indices) + len(
                    self._tent_rotation_indices) + len(
                        self._windmill_rotation_indices) + len(
                            self._tower_rotation_indices) + len(
                                self._plant_type_indices) + len(
                                    self._tree_type_indices)

    def get_num_dynamic_embeddings(self) -> int:
        base_num: int = len(self._card_count_indices) + len(self._card_color_indices) + len(
            self._card_selection_indices) + \
               len(self._card_shape_indices) + len(self._leader_rotation_indices) + len(
            self._follower_rotation_indices)
        if self._use_previous_cards_in_input:
            base_num += len(self._prev_visited_card_count_indices) + len(
                self._prev_visited_card_color_indices) + len(
                    self._prev_visited_card_selection_indices) + len(
                        self._prev_visited_card_shape_indices)
        return base_num

    def batch_environments(self, data: List[StepExample],
                           games: GamesCollection,
                           batch_previous_targets: bool) -> EnvironmentBatch:
        """Batches a list of step examples into environment batches
        (batches the static and dynamic information about the environment)."""
        all_previous_targets: Optional[torch.Tensor] = None
        previous_target: Optional[torch.Tensor] = None
        if batch_previous_targets:
            all_previous_targets, previous_target = _batch_previous_targets(
                data)

        return EnvironmentBatch(self._batch_games(data, games),
                                self._batch_dynamic_info(data),
                                _batch_previous_visitations(data),
                                all_previous_targets, previous_target)
