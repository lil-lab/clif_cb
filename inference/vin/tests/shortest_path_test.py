"""
Tests VIN by placing follower in a random location/rotation with random obstacles, and tries to get it to go to a 
random goal. Predicted action is validated via a shortest path.
"""
from __future__ import annotations

import copy
import numpy as np
import random
import torch

from config.rollout import GameConfig
from config.data_config import TokenizerConfig, DataConfig
from config.training_configs import SupervisedTargetConfig
from data.dataset_split import DatasetSplit
from environment.action import Action, MOVEMENT_ACTIONS
from environment.position import EDGE_WIDTH, Position
from environment.rotation import ROTATIONS
from environment.player import Player
from inference.vin.vin_model import Cerealbar_VIN, _get_cerealbar_axial_2d_kernels
from learning.batching.environment_batch import EnvironmentBatch
from learning.batching.environment_batcher import EnvironmentBatcher
from learning.training import load_training_data
from simulation.server_socket import ServerSocket
from simulation.unity_game import UnityGame
from simulation.planner import find_path_between_positions, get_player_new_state

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.dataset import DatasetCollection
    from data.step_example import StepExample
    from typing import List, Set, Tuple

NUM_TESTS: int = 50
RANDOM_GOALS: bool = False  # If true, a goal location is randomly sampled on the board. If not, a goal is sampled
# from the target goals of the example.
ROLL_OUT: bool = True
NEXT_TARGET: bool = False
BATCH_SIZE: int = 1
NEXT_TARGET = False if ROLL_OUT else ROLL_OUT
NEXT_TARGET = 1 if ROLL_OUT else BATCH_SIZE
MAX_STEPS: int = 50


def _load_data() -> DatasetCollection:
    data: DatasetCollection = load_training_data(DataConfig(TokenizerConfig()),
                                                 debug=True,
                                                 val_only=True)
    data.construct_supervised_step_examples(
        SupervisedTargetConfig(True, True, False))

    return data


def random_test(vin_model: Cerealbar_VIN, examples: List[StepExample],
                env_batch: EnvironmentBatch, all_data: DatasetCollection,
                server: ServerSocket):
    goals: torch.Tensor = torch.zeros(
        (len(examples), len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH))

    sampled_goals: List[Player] = list()

    for i, example in enumerate(examples):
        # Sample a goal
        if RANDOM_GOALS:
            raise NotImplementedError
        else:
            if NEXT_TARGET:
                goal = example.next_target_configuration
            else:
                possible_targets: Set[
                    Player] = example.possible_target_configurations
                goal: Player = random.sample(possible_targets, 1)[0]

        sampled_goals.append(goal)
        goals[i,
              ROTATIONS.index(goal.rotation), goal.position.x,
              goal.position.y] = 1.

    tuple_states: torch.Tensor = torch.cat([
        env_batch.dynamic_info.current_rotations.long(),
        env_batch.dynamic_info.current_positions
    ],
                                           dim=1)

    # Obstacles here is the static obstacle mask, not the one with the leader, because in rare cases in the original
    # CB data the players intersect with each other, so we don't want to treat it as an obstacle in these tests.
    q_values: torch.Tensor = vin_model(goals,
                                       tuple_states,
                                       env_batch.static_info.obstacle_mask,
                                       k=25)[0]

    for i, (example, qs) in enumerate(zip(examples, q_values)):
        argmaxes: List[Action] = list()
        max_value: torch.Tensor = torch.max(qs)
        for action, value in zip(MOVEMENT_ACTIONS, qs):
            if torch.allclose(max_value, value):
                argmaxes.append(action)

        predicted = ', '.join([str(action) for action in argmaxes])

        curent_state = (
            env_batch.dynamic_info.current_rotations[i].long().item(),
            env_batch.dynamic_info.current_positions[i][0].long().item(),
            env_batch.dynamic_info.current_positions[i][1].long().item(),
        )
        goal_state = (
            ROTATIONS.index(sampled_goals[i].rotation),
            sampled_goals[i].position.x,
            sampled_goals[i].position.y,
        )

        if NEXT_TARGET:
            print(
                f'true target: {example.target_action}; q predicted: {predicted}; current state {curent_state}; goal state {goal_state}'
            )
            target_action = example.target_action
        else:
            avoid_indexes: np.array = np.argwhere(
                env_batch.static_info.obstacle_mask[i, :, :] == 1).numpy()
            avoid_locations: List[Position] = [
                Position(x=avoid_indexes[0, i], y=avoid_indexes[1, i])
                for i in range(avoid_indexes.shape[1])
            ]
            target_states, target_actions = find_path_between_positions(
                avoid_locations=avoid_locations,
                start_state=Player(
                    rotation=ROTATIONS[curent_state[0]],
                    position=Position(curent_state[1], curent_state[2]),
                    is_follower=True,
                ),
                target_state=Player(
                    rotation=ROTATIONS[goal_state[0]],
                    position=Position(goal_state[1], goal_state[2]),
                    is_follower=True,
                ),
            )
            #print(target_states, target_actions)
            target_action = target_actions[0]

        print(
            f'true target: {target_action}; q predicted: {predicted}; current state {curent_state}; goal state {goal_state}'
        )

        if len(argmaxes) > 1 or argmaxes[0] != target_action:

            # Display on the board.
            game: UnityGame = UnityGame(
                all_data.games.games[example.game_id].environment,
                example.state, example.instruction, 100,
                GameConfig(allow_player_intersections=True), list(), list(),
                server)

            dist_to_send: np.ndarray = np.zeros((EDGE_WIDTH, EDGE_WIDTH))
            goal: Player = sampled_goals[i]
            dist_to_send[goal.position.x][goal.position.y] = 1

            print(f'target goal: {goal}')

            game.send_map_probability_dist(dist_to_send)

            input('Check out this mistake...')


def random_test_rollout(vin_model: Cerealbar_VIN,
                        example: StepExample,
                        env_batch: EnvironmentBatch,
                        all_data: DatasetCollection,
                        server: ServerSocket,
                        visualize_rollout: bool = True):
    goals: torch.Tensor = torch.zeros(
        (1, len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH))

    sampled_goals: List[Player] = list()

    # Sample a goal
    if RANDOM_GOALS:
        raise NotImplementedError
    else:
        possible_targets: Set[Player] = example.possible_target_configurations
        goal: Player = random.sample(possible_targets, 1)[0]  # todo: remove
        # goal: Player = list(possible_targets)[0] # todo: remove
        goal = example.final_target

    sampled_goals.append(goal)
    goals[0,
          ROTATIONS.index(goal.rotation), goal.position.x,
          goal.position.y] = 1.

    tuple_states: torch.Tensor = torch.cat([
        env_batch.dynamic_info.current_rotations.long(),
        env_batch.dynamic_info.current_positions
    ],
                                           dim=1)
    state = copy.deepcopy(example.state.follower)

    # For interactive debugging
    game: UnityGame = UnityGame(
        all_data.games.games[example.game_id].environment, example.state,
        example.instruction, 100, GameConfig(allow_player_intersections=True),
        list(), list(), server)

    dist_to_send: np.ndarray = np.zeros((EDGE_WIDTH, EDGE_WIDTH))
    goal: Player = sampled_goals[0]
    dist_to_send[goal.position.x][goal.position.y] = 1

    print(f'target goal: {goal}')

    game.send_map_probability_dist(dist_to_send)

    # Obstacles here is the static obstacle mask, not the one with the leader, because in rare cases in the original
    # CB data the players intersect with each other, so we don't want to treat it as an obstacle in these tests.
    action_history = []
    input("")
    for st in range(MAX_STEPS):
        # Policy rollout
        print(torch.sum(env_batch.static_info.obstacle_mask))
        vin_outputs: Tuple[torch.Tensor, List[Action]] = vin_model(
            goals,
            tuple_states,
            env_batch.static_info.obstacle_mask,
            k=25,
            debug=False)
        q_values, best_actions = vin_outputs

        print(q_values)

        avoid_indexes: np.array = np.argwhere(
            env_batch.static_info.obstacle_mask[0, :, :] == 1).numpy()
        avoid_locations: List[Position] = [
            Position(x=avoid_indexes[0, i], y=avoid_indexes[1, i])
            for i in range(avoid_indexes.shape[1])
        ]
        new_state = get_player_new_state(state, best_actions[0],
                                         avoid_locations)
        action_history.append(best_actions[0])
        game.execute_follower_action(best_actions[0])
        if new_state == sampled_goals[0]:
            break

        state = new_state
        tuple_states: torch.Tensor = torch.tensor(
            np.array([[
                int(state.rotation.to_radians()), state.position.x,
                state.position.y
            ]]))
        input("")

    # Visualize rollout
    if visualize_rollout or False:
        pass


def run_test(vin_model: Cerealbar_VIN, all_data: DatasetCollection,
             server: ServerSocket):
    batcher: EnvironmentBatcher = EnvironmentBatcher()
    for _ in range(NUM_TESTS):
        step_examples: List[StepExample] = all_data.static_datasets[
            DatasetSplit.TRAIN].step_examples

        if ROLL_OUT:
            for example in all_data.static_datasets[
                    DatasetSplit.TRAIN].instruction_examples:
                for step_example in example.step_examples[2:]:
                    batched_environments: EnvironmentBatch = batcher.batch_environments(
                        [step_example], all_data.games)
                    random_test_rollout(vin_model, step_example,
                                        batched_environments, all_data, server)
        else:
            random_test(vin_model, step_examples, batched_environments,
                        all_data, server)


if __name__ == '__main__':
    vin: Cerealbar_VIN = Cerealbar_VIN()
    vin._set_kernels(_get_cerealbar_axial_2d_kernels(), is_axial=True)

    dataset: DatasetCollection = _load_data()

    server_socket: ServerSocket = ServerSocket('localhost', 3706)
    server_socket.start_unity()
    run_test(vin, dataset, server_socket)
    server_socket.close()
