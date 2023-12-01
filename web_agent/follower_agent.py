"""The actual agent that does inference using the model and the current game state."""
from __future__ import annotations
import copy
import logging
import os
import platform
import time
import torch
import yaml

from dataclasses import dataclass
from datetime import datetime

from config import experiment, rollout
from data import bpe_tokenizer
from data.dataset import GamesCollection
from data.game import Game
from environment import observation
from environment.action import Action
from environment.player import Player
from environment.position import Position
from environment.rotation import Rotation
from environment.state import State, state_from_proto
from environment.static_environment import StaticEnvironment, static_from_proto
from evaluation.position_prediction_evaluation import get_argmax_vin_prediction
from inference.top_k_sampling import get_top_k_vin_sample, NUM_TOP_K_SAMPLES
from inference.rollout import get_ensemble_model_predictions, get_single_model_predictions
from inference.rollout_tracker import RolloutTracker
from inference.predicted_action_distribution import ActionPredictions
from learning.batching import step_batch
from model.position_prediction import PositionPredictionModel
from protobuf import CerealBarProto_pb2
from simulation.python_game import PythonGame
from simulation.game import LEADER_MOVES_PER_TURN
from util import torch_util
from web_agent.recording import RecordedGame, RecordedRollout, SampledAction

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from environment.card import Card
    from environment.observation import Observation
    from inference.predicted_voxel import VoxelPredictions
    from typing import List, Optional, Set, Tuple, Union

LINUX_OS: str = 'Linux'
MAC_OS: str = 'Darwin'
system_os: str = platform.system()

if system_os == MAC_OS:
    # Macbook
    AGENT_PREFIX = '../cb_vin_feedback/experiments/'
    LOG_PATH_PREFIX = '../cerealbar-game/'
elif system_os == LINUX_OS:
    # Bigbox
    AGENT_PREFIX = '/home/ubuntu/cb_vin_feedback/experiments/'
    LOG_PATH_PREFIX = '/home/ubuntu/cerealbar-game/'
else:
    raise ValueError('Unrecognized system type: %s' % system_os)

# Need to set this
UNIQUE_ID = "test"
assert UNIQUE_ID is not None

ROLLOUT_LOG_DIR: str = LOG_PATH_PREFIX + f'/agent_rollouts_{UNIQUE_ID}/'


def get_current_time():
    # Stores the current time and date (including microseconds)
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')


@dataclass
class FactoryArgs:
    experiment_dirs: List[str]
    model_saves: List[str]
    sampling: str


class NeuralNetworkFollower:
    def __init__(self,
                 socket,
                 port_id,
                 id,
                 experiment_name: str = '',
                 model_save_name: str = '',
                 do_sampling: str = '',
                 spec_file: str = ''):
        self._id = id
        self._port_id = port_id

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=LOG_PATH_PREFIX + "agent_logs/agent_" +
            str(get_current_time()).replace(' ', '_').replace('-', '_') + '_' +
            str(self._id) + "_console.log",
            level=logging.DEBUG,
            format="%(asctime)s:%(levelname)s:%(filename)s:%(message)s")

        self._experiment_path: str = ''
        self._model_name: str = ''
        self._pretrained_path: str = ''

        self._experiment_paths: List[Tuple[str, str]] = None
        self._experiment_configs: List[experiment.ExperimentConfig] = list()

        if spec_file:
            with open(spec_file) as infile:
                args = FactoryArgs(**yaml.load(infile))
            self._do_sampling = args.sampling == 'True'

            self._experiment_paths = list()
            for directory, save in zip(args.experiment_dirs, args.model_saves):
                self._experiment_paths.append((AGENT_PREFIX + directory, save))

            logging.info(
                'Created ensembling agents with port ID {port_id}, agent ID {id}, and agents:'
            )
            for directory, save in self._experiment_paths:
                logging.info(f'\t{directory} / {save}')

        else:
            self._do_sampling: bool = do_sampling == 'True'

            self._experiment_path = AGENT_PREFIX + experiment_name
            self._model_name = model_save_name
            self._pretrained_path: str = os.path.join(self._experiment_path,
                                                      self._model_name)

            logging.info(
                'Created follower agent with port ID %s, agent ID %s, experiment file %s/%s'
                % (port_id, id, experiment_name, model_save_name))

        logging.info(f'sampling? {self._do_sampling}')

        self._game_id: str = None
        self._game_seed: int = None
        self._num_cards: int = None
        self._game: PythonGame = None

        self._socket = socket

        self._models: Optional[List[PositionPredictionModel]] = None
        self._current_tracker: Optional[RolloutTracker] = None

        self._action_sequence = list()
        self._static_game_collection: Optional[GamesCollection] = None
        self._current_observation: Optional[observation.Observation] = None

        # Recording rollouts
        self._current_instruction_idx = 0
        self._game_dir: str = None
        self._current_rollout: Optional[RecordedRollout] = None

        # Consistency with the server
        self._global_action_idx = 0
        self._received_instruction_idx = 0

        # Only for printing
        self._num_actions_in_turn = 0

    def get_id(self) -> str:
        return self._id

    def get_game_id(self) -> str:
        return self._game_id

    def get_game(self):
        return self._game

    def set_game_id(self, game_id):
        self._game_id = game_id
        logging.info('Playing game ' + str(self._game_id))

    def set_game_seed(self, game_seed):
        self._game_seed = game_seed
        logging.info('Game seed: ' + str(self._game_seed))

    def set_num_cards(self, num_cards):
        self._num_cards = num_cards
        logging.info('Number of cards: ' + str(self._num_cards))

    def connect(self):
        logging.info('Connecting to port %s' % self._port_id)
        self._socket.connect('http://localhost:%s' % self._port_id)
        self._socket.emit('sendCredentials', {
            'worker_id': self._id,
            'assignment_id': self._id
        })
        logging.info('Started agent ' + str(self._id))

    def has_set_removed(self):
        """Checks whether there is a set missing from the board."""
        return len(self._game.get_current_state().cards) == self._num_cards - 3

    def add_instruction(self, instruction: str, instruction_idx: int):
        """Adds an instruction from the leader player to the game."""
        """
        if instruction_idx == 0:
            instruction = "turn around and get the red triangles next to the mountain"
        elif instruction_idx == 1:
            instruction = "then get the orange card"
        elif instruction_idx == 2:
            instruction = "deselect the card you are standing on"
        elif instruction_idx == 3:
            instruction = "turn left and get the orange square past the windmill"
            """

        if instruction_idx != self._received_instruction_idx:
            logging.warning(
                'Instruction #%s was received out of order (instruction: %s)' %
                (instruction_idx, instruction))
            self.process_end_of_game(exception=True)

        self._received_instruction_idx += 1

        logging.info('Added instruction #%s: %s' %
                     (instruction_idx, instruction))
        self._game.add_instruction(instruction)

    def _load_model(self):
        self._models = list()
        try:
            if self._experiment_paths:
                for exp_name, model_name in self._experiment_paths:
                    logging.info(f'Loading model {exp_name} / {model_name}')

                    tokenizer: bpe_tokenizer.BPETokenizer = bpe_tokenizer.load_bpe_tokenizer(
                        exp_name)

                    self._experiment_configs.append(
                        experiment.load_experiment_config_from_json(
                            os.path.join(exp_name,
                                         experiment.CONFIG_FILE_NAME)))

                    model = PositionPredictionModel(
                        self._experiment_configs[-1].get_model_config(),
                        tokenizer,
                        vin_backprop_config=None)
                    model.to(torch_util.DEVICE)

                    model.load_state_dict(
                        torch.load(os.path.join(exp_name, model_name),
                                   map_location=torch_util.DEVICE))
                    model.eval()

                    self._models.append(model)
            else:
                logging.info('Loading tokenizer')
                tokenizer: bpe_tokenizer.BPETokenizer = bpe_tokenizer.load_bpe_tokenizer(
                    self._experiment_path)

                self._experiment_configs.append(
                    experiment.load_experiment_config_from_json(
                        os.path.join(self._experiment_path,
                                     experiment.CONFIG_FILE_NAME)))

                logging.info('Loading model')

                model: PositionPredictionModel = PositionPredictionModel(
                    self._experiment_configs[-1].get_model_config(),
                    tokenizer,
                    vin_backprop_config=None)
                model.to(torch_util.DEVICE)

                model.load_state_dict(
                    torch.load(self._pretrained_path,
                               map_location=torch_util.DEVICE))

                logging.info(
                    f'Loading model from path {self._pretrained_path}')
                model.eval()

                self._models.append(model)

        except Exception as e:
            logging.info(e)
            self.process_end_of_game(exception=True)

    def start_game(self, static_environment: StaticEnvironment,
                   initial_state: State):
        if not self._game_id:
            logging.info(
                'Did not yet set the game ID; cannot process initial environment info.'
            )
            raise ValueError('Did not yet set game ID or seed!')

        self._load_model()

        self._static_game_collection = GamesCollection(
            {self._game_id: Game(self._game_id, static_environment)}, dict())

        logging.info('Started game!')
        self._game = PythonGame(static_game_info=static_environment,
                                initial_state=initial_state,
                                initial_instruction=None,
                                initial_num_moves=LEADER_MOVES_PER_TURN,
                                game_config=rollout.GameConfig(
                                    allow_player_intersections=False,
                                    keep_track_of_turns=True,
                                    auto_end_turn=False,
                                    check_valid_state=False,
                                    generate_new_cards=False,
                                    log_fn=logging.info,
                                    start_with_leader=True),
                                leader_actions=None,
                                expected_sets=None)

        self._current_observation = observation.create_first_observation(
            initial_state,
            self._experiment_configs[0].data_config.maximum_memory_age,
            fully_observable=False)

        game_data = RecordedGame(self._game_id, self._game_seed,
                                 static_environment, self._pretrained_path)
        self._game_dir = game_data.save(ROLLOUT_LOG_DIR)

    def move_leader(self, action: Action, received_action_id: int,
                    next_position: Position, next_rotation: Rotation):
        logging.info(
            'Executing leader %s; index %s; expecting position %s and rotation %s'
            % (action, received_action_id, next_position, next_rotation))

        try:
            if received_action_id != self._global_action_idx:
                raise ValueError(
                    'Expected next leader action to have index %s; got %s instead'
                    % (self._global_action_idx, received_action_id))

            self._game.execute_leader_action(action)

            leader_status = self._game.get_current_state().leader
            wrong_orientation = False
            if leader_status.position != next_position:
                logging.warning(
                    'Leader was not in expected position according to server. Executed to get %s; server '
                    'sent %s' % (leader_status.position, next_position))
                wrong_orientation = True
            if leader_status.rotation != next_rotation:
                logging.warning(
                    'Leader was not in expected rotation according to server. Executed to get %s; server '
                    'sent %s' % (leader_status.rotation, next_rotation))
                wrong_orientation = True
            if wrong_orientation:
                logging.warning(
                    'Got wrong leader configuration after executing action. Ending game.'
                )
                self.process_end_of_game(exception=True)
            self._global_action_idx += 1

            self._reset_current_observation()

        except Exception as e:
            logging.warning('Could not move leader. Exception: %s' % e)
            self.process_end_of_game(exception=True)

    def _fill_with_game_info(self, obj):
        obj.gameinfo.seed = ""
        obj.gameinfo.workerid = self.get_id()
        obj.gameinfo.assignmentid = self.get_id()
        obj.gameinfo.character = 'Follower'
        obj.gameinfo.gameid = self.get_game_id()
        obj.gameinfo.turnid = self._game.get_turn_index()

    def _emit_finish_command(self):
        logging.info('Finishing command #%s' %
                     self._game.get_instruction_index())
        instruction_idx: CerealBarProto_pb2.InstructionIndex = CerealBarProto_pb2.InstructionIndex(
        )
        instruction_idx.instructionindex = self._game.get_instruction_index(
        ) - 1
        self._fill_with_game_info(instruction_idx)

        self._socket.emit('finishedCommand',
                          instruction_idx.SerializeToString())

    def _emit_movement(self, pred_action: Action):
        logging.info('Emitting movement: %s' % pred_action)
        movement = CerealBarProto_pb2.Movement()
        movement.character = 'Follower'
        movement.type = str(pred_action)

        current_follower: Player = self._game.get_current_state().follower
        new_position: Position = current_follower.position
        new_rotation: Rotation = current_follower.rotation

        movement.nextposition.hexX = new_position.x
        movement.nextposition.hexZ = new_position.y
        movement.nextrotation = int(new_rotation)
        self._fill_with_game_info(movement)
        self._socket.emit('movement', movement.SerializeToString())

    def emit_your_turn(self, method):
        logging.info('Sending yourTurn to leader')
        your_turn = CerealBarProto_pb2.YourTurn()
        your_turn.method = method
        self._fill_with_game_info(your_turn)
        self._socket.emit('yourTurn', your_turn.SerializeToString())

    def reboot(self):
        try:
            if self._current_rollout:
                self._current_rollout.set_unfinished()
                self._end_instruction()
            self._game.reboot()
        except Exception as e:
            logging.warning('Could not reboot follower. Reason: %s' % e)
            self.process_end_of_game(exception=True)

    def _reset_current_observation(self):
        if self._current_tracker:
            current_observation: observation.Observation = self._current_tracker.get_current_observation(
            )
        else:
            current_observation: observation.Observation = self._current_observation

        self._current_observation = observation.update_observation(
            current_observation,
            self._game.get_current_state(),
            update_ages=False)

        if self._current_tracker:
            self._current_tracker.set_current_observation(
                self._current_observation)

    def add_new_cards(self, card_list: List[Card]):
        self._game.add_cards(card_list)
        self._reset_current_observation()

    def _end_instruction(self):
        self._current_rollout.save(self._game_dir)
        self._current_rollout = None
        self._current_tracker = None
        self._action_sequence = list()

    def _execute_action_with_target_voxel(
            self, predictions: Union[ActionPredictions, VoxelPredictions],
            predicted_action: Action,
            predicted_configuration: Optional[Player],
            directly_predicted: bool,
            ensemble_model_predictions: List[ActionPredictions]):
        assert self._current_tracker is not None

        previous_state: State = copy.deepcopy(self._game.get_current_state())
        previous_observation: Observation = copy.deepcopy(
            self._current_observation)

        self._current_tracker.execute_action(
            predicted_action,
            predicted_configuration,
            allow_no_config=directly_predicted)

        self._action_sequence.append(predicted_action)

        logging.info(
            f'Executed action (#{self._num_actions_in_turn} in turn): {predicted_action}'
        )

        self._current_observation = self._current_tracker.get_current_observation(
        )

        if predicted_action == Action.STOP:
            action_id = 'instruction_%s' % self._current_instruction_idx
        else:
            action_id = 'move_%s' % self._global_action_idx
            self._num_actions_in_turn += 1

        sampled_action: SampledAction = SampledAction(
            action_id,
            previous_state,
            previous_observation,
            predictions.off_graph(0),
            predicted_configuration,
            predicted_action,
            copy.deepcopy(self._current_tracker.get_current_state().follower),
            ensemble_model_predictions=ensemble_model_predictions)

        self._current_rollout.add_sample(sampled_action)

        if predicted_action == Action.STOP:
            # Reset everything
            logging.info('Resetting everything because the action was STOP')
            self._end_instruction()

    def _predict_and_emit_action(self):
        logging.info('Running inference')
        st = time.time()

        obstacle_positions: Set[Position] = self._static_game_collection.games[
            self._current_tracker.get_game_id(
            )].environment.get_obstacle_positions() | {
                self._current_tracker.get_current_state().leader.position
            }

        ensemble_predictions: Optional[List[ActionPredictions]] = None

        with torch.no_grad():
            if self._experiment_paths:
                predictions, each_ensemble_predictions = get_ensemble_model_predictions(
                    [self._current_tracker.get_current_step_example()],
                    self._static_game_collection,
                    self._models,
                    softmax_normalization=True,
                    use_voting=True,
                    allow_player_intersections=False,
                    current_rollouts=[self._current_tracker])
                ensemble_predictions = [
                    pred.off_graph(0) for pred in each_ensemble_predictions
                ]
            else:
                predictions: ActionPredictions = get_single_model_predictions(
                    [self._current_tracker.get_current_step_example()],
                    self._static_game_collection, self._models[0])

            action_to_take, probability = predictions.argmax(
                0,
                self._current_tracker.get_current_state().follower,
                obstacle_positions,
                sample=self._do_sampling)

        inference_time: float = time.time() - st
        logging.info('\tinference time\t%s' % inference_time)
        logging.info('Predicted action: %s' % action_to_take)

        # Execute the action
        self._execute_action_with_target_voxel(
            predictions,
            action_to_take,
            predicted_configuration=None,
            directly_predicted=True,
            ensemble_model_predictions=ensemble_predictions)

        # Send the action to the server
        time.sleep(max(0., 1.0 - inference_time))
        if action_to_take == Action.STOP:
            self._emit_finish_command()
        else:
            self._global_action_idx += 1
            self._emit_movement(action_to_take)

    def _initialize_tracker(self):
        # Update the current observation to include the newest leader actions
        # Return value indicates if the operation was successful (i.e., not canceled by the Leader)
        self._current_observation = observation.update_observation(
            self._current_observation, self._game.get_current_state(), True)

        instruction: str = self._game.get_current_instruction()
        logging.info(
            f'Creating a new example tracker with instruction: {instruction}')

        self._current_instruction_idx = copy.deepcopy(
            self._game.get_instruction_index())

        self._current_tracker = RolloutTracker(
            instruction,
            self._game_id,
            f'{self._game_id}-{self._current_instruction_idx}',
            self._current_observation,
            self._game.get_current_state(),
            self._game,
            copy_allowed=False)

        logging.info(
            f'Created a new example tracker for instruction: {instruction}')

        # Start the current rollout
        logging.info('Creating a rollout for current instruction index: %s' %
                     self._current_instruction_idx)
        self._current_rollout = RecordedRollout(
            self._game_id, self._current_instruction_idx,
            self._game.get_current_instruction())

    def make_one_move(self):
        # Initialize a tracker
        if not self._current_tracker:
            self._initialize_tracker()

        try:
            # Try predicting and emitting an action to the server.
            self._predict_and_emit_action()
        except Exception as e:
            logging.warning(
                'Could not predict and emit follower action. Reason: %s' % e)
            if 'None' not in str(e):
                self.process_end_of_game(exception=True)

    def end_game(self):
        if self._current_rollout:
            self._current_rollout.set_unfinished()
            self._current_rollout.save(self._game_dir)
            self._current_rollout = None

    def request_action(self):
        logging.info('Action requested from the server.')
        while len(self._game.get_current_state().cards) != 21:
            time.sleep(0.05)

        # Basic checks that an action can be taken.
        if self._game.is_leader_turn():
            logging.info(
                'It should be the follower\'s turn, not the leader\'s.')
            self._game.set_turn(is_leader=False)

        if not self._game.instruction_buffer_size():
            # Should never start a turn without instructions to follow.
            logging.warning(
                'Immediately ended turn after starting -- there are no more instructions to follow, '
                'so ending turn.')
            return

        self.make_one_move()

    def set_turn(self, leader: bool):
        self._game.set_turn(is_leader=leader)
        self._num_actions_in_turn = 0

    def process_end_of_game(self, exception=False):
        if self._current_rollout:
            self._current_rollout.set_unfinished()
            self._current_rollout.save(self._game_dir)

        gameinfo = CerealBarProto_pb2.StaticGameInfo()
        gameinfo.seed = ""
        gameinfo.workerid = self.get_id()
        gameinfo.assignmentid = self.get_id()
        gameinfo.character = 'Follower'
        gameinfo.gameid = self.get_game_id()

        message = 'killAgentDueToBug' if exception else 'canKillAgent'
        self._socket.emit(message, gameinfo.SerializeToString())
        exit()

    def process_map_info(self, map_info: CerealBarProto_pb2.MapInfo):
        static_environment: StaticEnvironment = static_from_proto(map_info)
        initial_state: State = state_from_proto(map_info)

        logging.info('leader/follower:')
        logging.info(initial_state.leader)
        logging.info(initial_state.follower)
        self.start_game(static_environment, initial_state)
