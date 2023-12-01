"""Creates an empty environment in the standalone for visualizing test VIN outputs."""
from __future__ import annotations

from config.rollout import GameConfig
from environment.position import EDGE_WIDTH, Position
from environment.terrain import Terrain
from environment.state import State
from environment.static_environment import StaticEnvironment
from simulation.unity_game import UnityGame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from environment.player import Player
    from simulation.server_socket import ServerSocket
    from typing import Dict


def create_game(server_socket: ServerSocket, leader: Player,
                follower: Player) -> UnityGame:
    all_grass: Dict[Position, Terrain] = dict()
    for x in range(EDGE_WIDTH):
        for y in range(EDGE_WIDTH):
            all_grass[Position(x, y)] = Terrain.GRASS
    return UnityGame(StaticEnvironment(terrain=all_grass,
                                       huts=list(),
                                       plants=list(),
                                       trees=list(),
                                       windmills=list(),
                                       tents=list(),
                                       towers=list(),
                                       lampposts=list()),
                     initial_state=State(leader, follower, list()),
                     initial_instruction='placeholder',
                     initial_num_moves=1000,
                     game_config=GameConfig(allow_player_intersections=True),
                     leader_actions=list(),
                     expected_sets=list(),
                     connection=server_socket)
