"""
Utilities for planning agent paths and moves.
"""
from __future__ import annotations

import heapq
import sys
import copy

from environment.action import Action
from environment.position import Position
from environment.rotation import Rotation
from environment.position import EDGE_WIDTH
from environment.player import Player

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, List, Optional, Set, Tuple

FOLLOWER_BACKWARDS_COST: int = 3.5
ODD_ROTATIONS: Dict[Tuple[int, int], Rotation] = {
    (1, 1): Rotation.NORTHEAST,
    (1, 0): Rotation.EAST,
    (1, -1): Rotation.SOUTHEAST,
    (0, -1): Rotation.SOUTHWEST,
    (-1, 0): Rotation.WEST,
    (0, 1): Rotation.NORTHWEST
}

EVEN_ROTATIONS: Dict[Tuple[int, int], Rotation] = {
    (0, 1): Rotation.NORTHEAST,
    (1, 0): Rotation.EAST,
    (0, -1): Rotation.SOUTHEAST,
    (-1, -1): Rotation.SOUTHWEST,
    (-1, 0): Rotation.WEST,
    (-1, 1): Rotation.NORTHWEST
}


def rotate_clockwise(rotation: Rotation) -> Rotation:
    """
    Maps from a rotation to the one clockwise from it.

    Input:
        rotation (Rotation): The rotation to rotate.

    Returns:
        The resulting Rotation.

    Raises:
        ValueError, if the input rotation is not recognized.
    """
    if rotation == Rotation.NORTHEAST:
        return Rotation.EAST
    elif rotation == Rotation.EAST:
        return Rotation.SOUTHEAST
    elif rotation == Rotation.SOUTHEAST:
        return Rotation.SOUTHWEST
    elif rotation == Rotation.SOUTHWEST:
        return Rotation.WEST
    elif rotation == Rotation.WEST:
        return Rotation.NORTHWEST
    elif rotation == Rotation.NORTHWEST:
        return Rotation.NORTHEAST
    else:
        raise ValueError('Unrecognized rotation: ' + str(rotation))


def rotate_counterclockwise(rotation: Rotation) -> Rotation:
    """
    Maps from a rotation to the one counterclockwise from it.

    Input:
        rotation (Rotation): The rotation to rotate.

    Returns:
        The resulting Rotation.

    Raises:
        ValueError, if the input rotation is not recognized.
    """
    if rotation == Rotation.NORTHEAST:
        return Rotation.NORTHWEST
    elif rotation == Rotation.EAST:
        return Rotation.NORTHEAST
    elif rotation == Rotation.SOUTHEAST:
        return Rotation.EAST
    elif rotation == Rotation.SOUTHWEST:
        return Rotation.SOUTHEAST
    elif rotation == Rotation.WEST:
        return Rotation.SOUTHWEST
    elif rotation == Rotation.NORTHWEST:
        return Rotation.WEST
    else:
        raise ValueError('Unrecognized rotation: ' + str(rotation))


def get_new_player_orientation(
        player: Player, action: Action,
        obstacle_positions: Set[Position]) -> Tuple[Position, Rotation]:
    new_position: Position = player.position
    new_rotation: Rotation = player.rotation
    if action == Action.MF:
        facing_position: Position = get_neighbor_move_position(
            player.position, player.rotation)[0]
        if facing_position in obstacle_positions:
            raise ValueError(
                f'Cannot MF player {player} to position {facing_position} because there is an obstacle in the way.'
            )
        new_position = facing_position
    elif action == Action.MB:
        behind_position: Position = get_neighbor_move_position(
            player.position, player.rotation)[1]
        if behind_position in obstacle_positions:
            raise ValueError(
                f'Cannot MB player {player} to position {behind_position} because there is an obstacle in the way.'
            )
        new_position = behind_position
    elif action == Action.RR:
        new_rotation = rotate_clockwise(player.rotation)
    elif action == Action.RL:
        new_rotation = rotate_counterclockwise(player.rotation)

    return new_position, new_rotation


# todo: merge to get_new_player_orientation
def get_player_new_state(player: Player, action: Action,
                         obstacle_positions: Set[Position]) -> Player:
    new_position: Position = player.position
    new_rotation: Rotation = player.rotation
    if action == Action.MF:
        facing_position: Position = get_neighbor_move_position(
            player.position, player.rotation)[0]
        if facing_position in obstacle_positions:
            #new_position = player.position
            raise ValueError(
                f'Cannot MF player {player} to position {facing_position} because there is an obstacle in the way.'
            )
        new_position = facing_position
    elif action == Action.MB:
        behind_position: Position = get_neighbor_move_position(
            player.position, player.rotation)[1]
        if behind_position in obstacle_positions:
            # new_position = player.position
            raise ValueError(
                f'Cannot MB player {player} to position {behind_position} because there is an obstacle in the way.'
            )
        new_position = behind_position
    elif action == Action.RR:
        new_rotation = rotate_clockwise(player.rotation)
    elif action == Action.RL:
        new_rotation = rotate_counterclockwise(player.rotation)

    return Player(rotation=new_rotation,
                  position=new_position,
                  is_follower=True)


def get_neighbor_move_position(
        current_position: Position,
        current_rotation: Rotation) -> Tuple[Position, Position]:
    back_rotation: Rotation = rotate_clockwise(
        rotate_clockwise(rotate_clockwise(current_rotation)))

    facing_position: Position = None
    behind_position: Position = None

    if current_position.y % 2 == 0:
        for offset, rot in EVEN_ROTATIONS.items():
            if rot == current_rotation:
                facing_position = Position(current_position.x + offset[0],
                                           current_position.y + offset[1])
            elif rot == back_rotation:
                behind_position = Position(current_position.x + offset[0],
                                           current_position.y + offset[1])
    else:
        for offset, rot in ODD_ROTATIONS.items():
            if rot == current_rotation:
                facing_position = Position(current_position.x + offset[0],
                                           current_position.y + offset[1])
            elif rot == back_rotation:
                behind_position = Position(current_position.x + offset[0],
                                           current_position.y + offset[1])
    return facing_position, behind_position


def get_neighbors(position: Position, env_width: int,
                  env_depth: int) -> List[Position]:
    """Gets the neighbors for a 
    Inputs:
        position (Position): The position to get a neighbor for.
        env_width (int): The width of the environment in hexes.
        env_depth (int): The depth of the environment in hexes.
    Returns:
        A list of neighboring positions.
    """
    x_pos: int = position.x
    y_pos: int = position.y

    if y_pos % 2 == 0:
        neighbors: List[Position] = [
            Position(x_pos - 1, y_pos + 1),
            Position(x_pos, y_pos + 1),
            Position(x_pos + 1, y_pos),
            Position(x_pos, y_pos - 1),
            Position(x_pos - 1, y_pos - 1),
            Position(x_pos - 1, y_pos)
        ]
    else:
        neighbors: List[Position] = [
            Position(x_pos, y_pos + 1),
            Position(x_pos + 1, y_pos + 1),
            Position(x_pos + 1, y_pos),
            Position(x_pos + 1, y_pos - 1),
            Position(x_pos, y_pos - 1),
            Position(x_pos - 1, y_pos)
        ]

    neighbors = [
        neighbor for neighbor in neighbors
        if 0 <= neighbor.x < env_width and 0 <= neighbor.y < env_depth
    ]

    return neighbors


def rotation_possibilities(
        start_position: Position, end_position: Position,
        agent_rotation: Rotation) -> List[Tuple[List[Action], Rotation]]:
    current_rotation: Rotation = copy.deepcopy(agent_rotation)

    num_rotations: int = get_num_rotations(start_position, end_position,
                                           current_rotation)

    if num_rotations == 0:
        return [([Action.MF], current_rotation)]
    elif num_rotations == 1:
        return [([Action.RR, Action.MF], rotate_clockwise(current_rotation)),
                ([Action.RL, Action.RL, Action.MB],
                 rotate_counterclockwise(
                     rotate_counterclockwise(current_rotation)))]
    elif num_rotations == 2:
        return [([Action.RR, Action.RR, Action.MF],
                 rotate_clockwise(rotate_clockwise(current_rotation))),
                ([Action.RL,
                  Action.MB], rotate_counterclockwise(current_rotation))]
    elif num_rotations == 3:
        return [([Action.MB], current_rotation),
                ([Action.RR, Action.RR, Action.RR, Action.MF],
                 rotate_clockwise(
                     rotate_clockwise(rotate_clockwise(current_rotation))))]
    elif num_rotations == 4:
        return [([Action.RR, Action.MB], rotate_clockwise(current_rotation)),
                ([Action.RL, Action.RL, Action.MF],
                 rotate_counterclockwise(
                     rotate_counterclockwise(current_rotation)))]
    elif num_rotations == 5:
        return [([Action.RL,
                  Action.MF], rotate_counterclockwise(current_rotation)),
                ([Action.RR, Action.RR, Action.MB],
                 rotate_clockwise(rotate_clockwise(current_rotation)))]
    else:
        raise ValueError('Can\'t rotate ' + str(num_rotations) + ' times.')


def get_num_rotations(start_position: Position, end_position: Position,
                      start_rotation: Rotation) -> int:
    num_rotations: int = 0

    start_depth = start_position.y

    move_vector: Tuple[int, int] = (end_position.x - start_position.x,
                                    end_position.y - start_position.y)

    temp_rot: Rotation = copy.deepcopy(start_rotation)

    if start_depth % 2 == 0:
        while not temp_rot == EVEN_ROTATIONS[move_vector]:
            num_rotations += 1
            temp_rot = rotate_clockwise(temp_rot)
    else:
        while not temp_rot == ODD_ROTATIONS[move_vector]:
            num_rotations += 1
            temp_rot = rotate_clockwise(temp_rot)

    return num_rotations


def follower_action_cost(actions: List[Action]) -> int:
    cost: int = 0
    for action in actions:
        if action == Action.MB:
            cost += FOLLOWER_BACKWARDS_COST
        else:
            cost += 1

    return cost


def manhattan_distance(a: Position, b: Position) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


class PositionRotationQueue:
    def __init__(self):
        self.elem: List[Tuple[int, Tuple[Position, Rotation]]] = []

    def is_empty(self) -> bool:
        return len(self.elem) == 0

    def put(self, pos: Position, rot: Rotation, pri: int) -> None:
        heapq.heappush(self.elem, (pri, (pos, rot)))

    def get(self) -> Tuple[Position, Rotation]:
        return heapq.heappop(self.elem)[1]


def find_path_between_positions(
    avoid_locations: List[Position],
    start_state: Player,
    target_state: Player,
    ignore_target_rotation: bool = False
) -> Optional[List[Tuple[Player, Action]]]:
    """
    Finds a path between two positions in the map.
    Args:
        avoid_locations
            A list of locations to avoid, including obstacles and cards that should not be touched.
        start_state
            The starting configuration (position/rotation) of the agent.
        target_state
            The target configuration (position/rotation) of the agent.
        ignore_target_rotation
            Whether to terminate the shortest path once the target position has been reached, 
            ignoring the location.
    """
    queue: PositionRotationQueue = PositionRotationQueue()
    queue.put(start_state.position, start_state.rotation, 0)
    paths: Dict[Tuple[Position, Rotation],
                Optional[Tuple[Tuple[Position, Rotation],
                               List[Action]]]] = dict()
    paths[(start_state.position, start_state.rotation)] = None

    total_dist: Dict[Tuple[Position, Rotation], int] = dict()
    total_dist[(start_state.position, start_state.rotation)] = 0

    has_ended: bool = False
    current_pos_and_rot: Tuple[Optional[Position],
                               Optional[Rotation]] = (None, None)

    while not queue.is_empty():
        current_pos_and_rot: Tuple[Position, Rotation] = queue.get()
        current_position: Position = current_pos_and_rot[0]
        current_rotation: Rotation = current_pos_and_rot[1]
        if current_position == target_state.position:
            has_ended = True
            break

        for next_position in get_neighbors(current_position, EDGE_WIDTH,
                                           EDGE_WIDTH):
            if next_position not in avoid_locations:
                action_rot_pairs: List[Tuple[
                    List[Action], Rotation]] = rotation_possibilities(
                        current_position, next_position, current_rotation)

                # There could be several ways to get from one hex to another by  Iterate through all of them,
                # considering the cost of backwards moves.
                for move_possibility in action_rot_pairs:
                    actions: List[Action] = move_possibility[0]
                    resulting_rotation: Rotation = move_possibility[1]

                    new_dist: int = total_dist[
                        current_pos_and_rot] + follower_action_cost(actions)

                    move_pair: Tuple[Position, Rotation] = (next_position,
                                                            resulting_rotation)
                    if new_dist < total_dist.get(move_pair, sys.maxsize):
                        total_dist[move_pair] = new_dist
                        paths[move_pair] = (current_pos_and_rot, actions)
                        priority: int = new_dist + manhattan_distance(
                            target_state.position, next_position)
                        queue.put(next_position, resulting_rotation, priority)

    if not has_ended:
        return None

    final_rotation: Rotation = current_pos_and_rot[1]
    path_positions: List[Position] = []
    actions: List[Action] = []
    while current_pos_and_rot != (start_state.position, start_state.rotation):
        segment_actions = paths[current_pos_and_rot][1]
        segment_actions.reverse()
        actions += segment_actions
        path_positions.append(current_pos_and_rot[0])
        current_pos_and_rot = paths[current_pos_and_rot][0]

    actions.reverse()
    path_positions.reverse()

    if not ignore_target_rotation and target_state.rotation != final_rotation:
        num_right: int = 0
        temp_right_rotation: Rotation = final_rotation
        while temp_right_rotation != target_state.rotation:
            num_right += 1
            temp_right_rotation = rotate_clockwise(temp_right_rotation)

        if num_right <= 3:
            actions.extend([Action.RR for _ in range(num_right)])
        else:
            actions.extend([Action.RL for _ in range(6 - num_right)])

    path_positions = [start_state.position] + path_positions

    # Now create an actual list of states and actions
    path_states: List[Player] = list()
    path_actions: List[Action] = list()
    pos_idx: int = 0
    current_rotation: Rotation = start_state.rotation

    for action in actions:
        state = Player(
            rotation=current_rotation,
            position=path_positions[pos_idx],
            is_follower=True,
        )
        path_states.append(state)
        path_actions.append(action)

        if action in {Action.MF, Action.MB}:
            pos_idx += 1
        elif action in {Action.RR, Action.RL}:
            if action == Action.RR:
                current_rotation = rotate_clockwise(current_rotation)
            else:
                current_rotation = rotate_counterclockwise(current_rotation)
        else:
            raise ValueError('Action should not be generated: ' + str(action))

    # Should end up in the expected rotation
    if not ignore_target_rotation:
        assert current_rotation == target_state.rotation

    return path_states, path_actions
