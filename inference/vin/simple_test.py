# python -m inference.vin.simple_test
import random
import torch

from environment.player import Player
from environment.position import EDGE_WIDTH, Position
from environment.rotation import ROTATIONS, Rotation

from util import torch_util
from inference.vin import standalone_visualizer
from inference.vin.vin_model import Cerealbar_VIN, _get_cerealbar_axial_2d_kernels
from simulation.server_socket import ServerSocket


def _generate_random_player(is_follower: bool) -> Player:
    x: int = random.randint(0, EDGE_WIDTH)
    y: int = random.randint(0, EDGE_WIDTH)
    rot: Rotation = random.sample(ROTATIONS, 1)[0]

    return Player(is_follower, Position(x, y), rot)


def example_of_standalone(server: ServerSocket):
    follower: Player = _generate_random_player(True)

    # Place the leader in the opposite corner of the follower
    west_side: bool = follower.position.x < 13
    south_side: bool = follower.position.y < 13

    leader_x: int = 24 if west_side else 0
    leader_y: int = 24 if south_side else 0

    leader: Player = Player(False, Position(leader_x, leader_y),
                            Rotation.NORTHEAST)

    standalone_visualizer.create_game(server, leader, follower)


def simple_test_1(model):
    """
    """

    goals = torch.ones((2, 6, 25, 25)).to(torch_util.DEVICE)
    #goals[0,0,3,3] = 1.
    start_state = torch.tensor([[2, 0, 0]]).to(torch_util.DEVICE)
    obstacles = torch.ones((2, 5, 5)).to(torch_util.DEVICE)

    for _ in range(10):
        action = model(goals, start_state, obstacles, False)[0][0]
        print(action)


if __name__ == "__main__":
    model = Cerealbar_VIN()
    transition_kernels = _get_cerealbar_axial_2d_kernels()
    model._set_kernels(transition_kernels, is_axial=True)

    server_socket: ServerSocket = ServerSocket('localhost', 3706)
    server_socket.start_unity()

    example_of_standalone(server_socket)
    input('press enter to close unity')

    server_socket.close()

    simple_test_1(model)
