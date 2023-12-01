"""Records what happened in a game to disk."""
from __future__ import annotations

import logging
import os
import pickle

from dataclasses import dataclass

from environment.action import Action
from environment.observation import Observation
from environment.player import Player
from environment.state import State
from environment.static_environment import StaticEnvironment
from inference.predicted_voxel import VoxelPredictions
from inference.predicted_action_distribution import ActionPredictions

from typing import List, Optional, Union


@dataclass
class SampledAction:
    global_game_action_id: str

    # Inputs (including cards, players, observation. Empirical visitation distribution can be computed by looking at
    # other actions in the sequence.)
    preceding_state: State
    preceding_observation: Observation

    # Outputs
    voxel_predictions: Union[VoxelPredictions,
                             ActionPredictions]  # Numpy representation here

    # TODO: If using VIN, add Q-values here
    # TODO: If sampling, store the sampled gumbel noise here

    argmax_voxel: Optional[Player]

    executed_action: Action

    resulting_configuration: Player

    ensemble_model_predictions: Optional[List[ActionPredictions]] = None

    def __str__(self):
        return f'{self.global_game_action_id}: agent sampled {self.executed_action}'


class RecordedRollout:
    """A recording of an agent rollout."""
    def __init__(self, game_id: str, instruction_index: int, instruction: str):
        self._game_id: str = game_id
        self._instruction_index: int = instruction_index
        self._instruction: str = instruction

        self._sampled_actions: List[SampledAction] = list()

        self._finished_instruction = True

    def get_actions(self) -> List[SampledAction]:
        return self._sampled_actions

    def get_instruction(self) -> str:
        return self._instruction

    def set_unfinished(self):
        self._finished_instruction = False

    def add_sample(self, action: SampledAction):
        self._sampled_actions.append(action)

    def get_instruction_index(self) -> int:
        return self._instruction_index

    def __str__(self):
        return f'Instruction #{self._instruction_index} from game {self._game_id} ' \
               f' with sampled length of {len(self._sampled_actions)}; finished={self._finished_instruction}: ' \
               f'"{self._instruction}"'

    def save(self, game_dir: str):
        if not os.path.exists(game_dir):
            raise FileNotFoundError('Game directory does not exist: %s' %
                                    game_dir)

        rollout_filepath = os.path.join(
            game_dir, f'instruction_{self.get_instruction_index()}.pkl')
        if os.path.exists(rollout_filepath):
            logging.info('Rollout already exists at %s' % rollout_filepath)
            return

        with open(rollout_filepath, 'wb') as ofile:
            pickle.dump(self, ofile)


class RecordedGame:
    def __init__(self, game_id: str, seed: int, environment: StaticEnvironment,
                 model_path: str):
        self._game_id: str = game_id
        self._seed: int = seed
        self._static_environment: StaticEnvironment = environment
        self._model_path: str = model_path

    def get_game_id(self) -> str:
        return self._game_id

    def get_environment(self) -> StaticEnvironment:
        return self._static_environment

    def __str__(self):
        return f'Game ID {self._game_id} with seed {self._seed} and model path {self._model_path}'

    def save(self, rollout_dir: str) -> str:
        game_dir = os.path.join(rollout_dir, self._game_id)
        if os.path.exists(game_dir):
            raise FileExistsError('Game directory already exists: %s' %
                                  game_dir)

        os.mkdir(game_dir)
        with open(os.path.join(game_dir, f'game_{self._game_id}.pkl'),
                  'wb') as ofile:
            pickle.dump(self, ofile)
        return game_dir
