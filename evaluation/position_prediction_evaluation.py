"""Defines metrics, evaluation functions, and evaluation loop for position prediction."""
from __future__ import annotations

import logging
import numpy as np
import os
import pyperclip
import random
import torch
from tqdm import tqdm

from config.rollout import GameConfig
from config.training_configs import SupervisedTargetConfig
from data import bpe_tokenizer
from data.example import get_examples_for_games
from environment.action import Action
from environment.player import Player
from environment.position import EDGE_WIDTH
from environment.rotation import ROTATIONS
from environment.static_environment import StaticEnvironment
from evaluation import action_prediction_metrics, position_prediction_metrics, rollout_metrics
from evaluation.metric import Metric, PROP_METRICS
from inference.predicted_action_distribution import ActionPredictions, PREDICTABLE_ACTIONS
from inference.predicted_voxel import VoxelPredictions
from inference.rollout import get_argmax_configurations
from inference.rollout_tracker import RolloutTracker
from inference.top_k_sampling import get_top_k_vin_sample, NUM_TOP_K_SAMPLES
from inference.vin.vin_model import get_vin_predictions
from learning.batching import step_batch
from model.position_prediction import PositionPredictionModel
from simulation.server_socket import ServerSocket
from simulation.unity_game import UnityGame
from util import torch_util

NUM_SAMPLES = 1

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.evaluation import EvaluationConfig
    from config.rollout import RolloutConfig
    from data.dataset import Dataset, DatasetCollection, GamesCollection
    from data.example import Example
    from environment.position import Position
    from environment.state import State
    from inference.vin.vin_model import Cerealbar_VIN
    from typing import Dict, List, Optional, Tuple, Union, Set


def _visualize_inference_on_example(model: PositionPredictionModel,
                                    example: Example, games: GamesCollection,
                                    game_server: ServerSocket):
    if not example.step_examples:
        raise ValueError(
            'Example must have step examples computed ahead of time.')

    # Run inference on the model
    environment: StaticEnvironment = games.games[
        example.get_game_id()].environment
    example.construct_supervised_step_examples(
        SupervisedTargetConfig(True, True, False, False),
        environment.get_obstacle_positions())

    batched: step_batch.StepBatch = step_batch.batch_steps(
        example.step_examples,
        games,
        model.get_tokenizer(),
        model.get_environment_batcher(),
        directly_predict_actions=model.directly_predicts_actions(),
        batch_targets=False)

    batched.to_device()
    predictions: VoxelPredictions = model(batched)

    # Setup the game
    game: UnityGame = UnityGame(
        games.games[example.get_game_id()].environment,
        example.target_action_sequence[0].previous_state, example.instruction,
        example.num_first_turn_steps,
        GameConfig(allow_player_intersections=True), example.leader_actions,
        example.expected_sets, game_server)

    # Print / visualize the predictions
    print(f'---------- {example.example_id} ----------')
    print(example.instruction)
    for i, step in enumerate(example.target_action_sequence):
        current_follower: Player = step.previous_state.follower

        if model.directly_predicts_actions():
            argmax_option, argmax_prob = predictions.argmax(i)
        else:
            argmax_option, argmax_prob = predictions.argmax(i,
                                                            allow_copy=False)

        if step.feedback_annotation is not None:
            if argmax_option != Action.STOP and not model.directly_predicts_actions(
            ):
                dist_to_send = np.zeros((EDGE_WIDTH, EDGE_WIDTH))
                argmax_pos, argmax_rot = argmax_option
                dist_to_send[argmax_pos.x][argmax_pos.y] = 1
                game.send_map_probability_dist(dist_to_send, 1)

            if step.feedback_annotation.sampled_goal_voxel is None:
                if model.directly_predicts_actions():
                    original_probability = step.feedback_annotation.probability_dist.action_probabilities[
                        PREDICTABLE_ACTIONS.index(step.target_action)]
                else:
                    original_probability = step.feedback_annotation.probability_dist.stop_probabilities
                    argmax_option = Action.STOP
                    argmax_prob = predictions.stop_probabilities[i]
            else:
                voxel = step.feedback_annotation.sampled_goal_voxel
                original_probability = step.feedback_annotation.probability_dist.voxel_probabilities[
                    ROTATIONS.index(voxel.rotation), voxel.position.x,
                    voxel.position.y]
                argmax_option = (voxel.position, voxel.rotation)
                argmax_prob = predictions.voxel_probabilities[i][
                    ROTATIONS.index(
                        voxel.rotation)][voxel.position.x][voxel.position.y]
            print(
                f'\toriginal probability: {(100. * original_probability):.2f}%\t{step.target_action}'
                f'\tfeedback: {step.feedback_annotation.feedback.num_positive - step.feedback_annotation.feedback.num_negative}'
            )

        prefix: str = f'\t{current_follower.position}\t{current_follower.rotation.shorthand()}' \
                      f'\t{(100. * argmax_prob):.2f}%'

        dist_to_send: np.ndarray = np.zeros((EDGE_WIDTH, EDGE_WIDTH))
        if isinstance(argmax_option, Action):
            print(f'{prefix}\t{argmax_option}')
        else:
            argmax_pos, argmax_rot = argmax_option

            dist_to_send[argmax_pos.x][argmax_pos.y] = 1

            print(f'{prefix}\t{argmax_pos}\t{argmax_rot.shorthand()}')

        game.send_map_probability_dist(dist_to_send)

        input(f'{i}. {step.target_action}')
        game.execute_follower_action(step.target_action)

    input('Press enter to show next example.')
    game.finish_all_leader_actions()


def get_argmax_vin_prediction(predictions: VoxelPredictions,
                              vin: Cerealbar_VIN, current_state: State,
                              game: Optional[UnityGame],
                              batch: step_batch.StepBatch,
                              allow_player_intersections: bool,
                              log_fn) -> Tuple[Action, Optional[Player]]:
    default_obstacle_mask: torch.Tensor = batch.environment_state.static_info.obstacle_mask
    if not allow_player_intersections:
        # Also include the leader position
        default_obstacle_mask = batch.environment_state.get_all_obstacles()

    argmax_action, argmax_prob = predictions.argmax(
        0,
        allow_copy=batch.original_examples[0].can_copy,
        obstacle_mask=default_obstacle_mask[0])

    if game:
        dist_to_send: np.ndarray = torch.sum(
            predictions.voxel_probabilities[0], dim=0).numpy()
        game.send_map_probability_dist(dist_to_send)

    argmax_voxel: Optional[Player] = None

    if argmax_action == Action.STOP:
        argmax_action = Action.STOP
    else:
        if argmax_action == Action.COPY:
            raise NotImplementedError(
                'Need to grab the argmax player from the previous prediction!')
        else:
            argmax_players: List[Player] = get_argmax_configurations(
                predictions, batch, allow_player_intersections)

        vin_predictions: Set[Action] = get_vin_predictions(
            vin, [current_state], batch, allow_player_intersections,
            argmax_players)[0]

        ac_str: str = ', '.join([str(ac) for ac in vin_predictions])
        log_fn(f'Possible actions: {ac_str}')

        argmax_action: Action = random.sample(vin_predictions, 1)[0]

        argmax_voxel: Player = argmax_players[0]
        argmax_pos: Position = argmax_voxel.position

        log_fn(f'Argmax configuration: {argmax_pos}; {argmax_voxel.rotation}')

        if game:
            dist_to_send: np.ndarray = np.zeros((EDGE_WIDTH, EDGE_WIDTH))
            dist_to_send[argmax_pos.x][argmax_pos.y] = 1
            game.send_map_probability_dist(dist_to_send, 1)

        argmax_prob: float = predictions.voxel_probabilities[0][
            ROTATIONS.index(
                argmax_voxel.rotation)][argmax_pos.x][argmax_pos.y].item()

    text: str = f'Executing action {argmax_action} (probability: {(100. * argmax_prob):.1f}%)'
    if log_fn == print:
        input(text)
    else:
        log_fn(text)

    return argmax_action, argmax_voxel


def _visualize_rollout_on_example(model: PositionPredictionModel,
                                  example: Example, games: GamesCollection,
                                  game_server: ServerSocket,
                                  rollout_config: RolloutConfig):
    game: UnityGame = UnityGame(
        games.games[example.get_game_id()].environment,
        example.target_action_sequence[0].previous_state, example.instruction,
        example.num_first_turn_steps, rollout_config.game_config,
        example.leader_actions, example.expected_sets, game_server)

    print(f'---------- {example.example_id} ----------')
    print(example.instruction)
    pyperclip.copy(example.instruction)

    rollout: RolloutTracker = RolloutTracker(example.instruction,
                                             example.get_game_id(),
                                             example.example_id,
                                             example.get_initial_observation(),
                                             example.get_initial_state(),
                                             game,
                                             copy_allowed=False)
    num_steps: int = 0
    while num_steps <= rollout_config.max_num_steps and not rollout.has_stopped(
    ):
        forced_action: Optional[Action] = None
        if num_steps == rollout_config.max_num_steps - 1:
            forced_action = Action.STOP

        batch: step_batch.StepBatch = step_batch.batch_steps(
            [rollout.get_current_step_example()],
            games,
            model.get_tokenizer(),
            model.get_environment_batcher(),
            batch_targets=False,
            directly_predict_actions=model.directly_predicts_actions())
        predictions: Union[VoxelPredictions, ActionPredictions] = model(batch)

        target_voxel: Optional[Player] = None
        if forced_action:
            # Force the action instead.
            argmax_to_execute = forced_action
            input(f'Forcing action {argmax_to_execute}')
        else:
            if rollout_config.restrict_to_neighbors:
                argmax_to_execute, probability = predictions.argmax_neighbor_action(
                    0,
                    rollout.get_current_state().follower,
                    games.games[example.get_game_id(
                    )].environment.get_obstacle_positions())

                dist_to_send: np.ndarray = torch.sum(
                    predictions.voxel_probabilities[0], dim=0).numpy()
                game.send_map_probability_dist(dist_to_send)

                input(
                    f'Executing action {argmax_action} (probability: {(100. * probability):.1f}%)'
                )

                # TODO: Get the argmax voxel here too to pass into the rollout.

            else:
                if model.uses_copy():
                    # TODO: Allow copy here
                    raise NotImplementedError(
                        'Allow copy here! Make sure to set it in the step example used to create '
                        'the batch.')

                if rollout_config.use_sampling:
                    argmax_to_execute, target_voxel = get_top_k_vin_sample(
                        predictions,
                        model.get_vin(),
                        rollout.get_current_state(),
                        game,
                        batch,
                        rollout_config.game_config.allow_player_intersections,
                        print,
                        k=NUM_TOP_K_SAMPLES)
                else:
                    if model.directly_predicts_actions():
                        assert isinstance(predictions, ActionPredictions)
                        argmax_to_execute, probability = predictions.argmax(
                            0,
                            rollout.get_current_state().follower,
                            games.games[example.get_game_id(
                            )].environment.get_obstacle_positions()
                            | {rollout.get_current_state().leader.position})
                        input(
                            f'Executing {argmax_to_execute} with probability {(100. * probability):.1f}%'
                        )
                    else:
                        assert isinstance(predictions, VoxelPredictions)
                        argmax_to_execute, target_voxel = get_argmax_vin_prediction(
                            predictions, model.get_vin(),
                            rollout.get_current_state(), game, batch,
                            rollout_config.game_config.
                            allow_player_intersections, print)

        rollout.execute_action(
            argmax_to_execute,
            target_voxel,
            allow_no_config=model.directly_predicts_actions())

        num_steps += 1

    rollout.finish_all_leader_actions()


def _launch_prediction_browser(evaluation_config: EvaluationConfig,
                               model: PositionPredictionModel,
                               eval_data: Dataset, games: GamesCollection):
    game_server: ServerSocket = ServerSocket(
        evaluation_config.standalone_config.ip_address,
        evaluation_config.standalone_config.port)
    game_server.start_unity()

    examples: List[Example] = eval_data.instruction_examples
    if evaluation_config.randomize:
        r: random.Random = random.Random(72)
        r.shuffle(examples)

    with open('eval.txt') as infile:
        exids = [line.strip() for line in infile.readlines()]

    for example in examples:
        if example.example_id not in exids:
            continue
        if evaluation_config.gold_forcing_actions:
            _visualize_inference_on_example(model, example, games, game_server)
        else:
            _visualize_rollout_on_example(model, example, games, game_server,
                                          evaluation_config.rollout_config)

    game_server.close()


def _eval_model(models: List[PositionPredictionModel], dataset: Dataset,
                games: GamesCollection, evaluation_config: EvaluationConfig,
                logfile_path: str):
    return rollout_metrics.evaluate_position_predictor(
        models,
        dataset.instruction_examples,
        games,
        evaluation_config.rollout_config,
        16,
        sample_actions=evaluation_config.sampling,
        show_progress=True,
        compute_accuracy_metrics=not evaluation_config.online_dataset_name,
        logfile_path=logfile_path)[0]


def load_and_evaluate_model(evaluation_config: EvaluationConfig,
                            data: DatasetCollection):
    # Setup: load step examples, load tokenizer, load model
    if evaluation_config.gold_forcing_actions:
        data.construct_supervised_step_examples(
            evaluation_config.loaded_experiment_config.get_target_config())
        logging.info('Number of step examples per split:')
        for split, dataset in data.static_datasets.items():
            logging.info(f'\t{split}\t{len(dataset.step_examples)}')

    models: List[PositionPredictionModel] = list()
    for i, model_path in enumerate(evaluation_config.model_filepaths):
        logging.info(f'---- Loading model {model_path} ----')
        logging.info('Loading tokenizer')
        tokenizer: bpe_tokenizer.BPETokenizer = bpe_tokenizer.load_bpe_tokenizer(
            evaluation_config.get_model_directory(i))

        logging.info('Loading model')
        model: PositionPredictionModel = PositionPredictionModel(
            evaluation_config.loaded_experiment_configs[i].get_model_config(),
            tokenizer,
            vin_backprop_config=None)
        model.to(torch_util.DEVICE)
        model.load_state_dict(
            torch.load(model_path, map_location=torch_util.DEVICE))
        logging.info('Created position prediction model.')

        model.eval()
        models.append(model)

    first_model: PositionPredictionModel = models[0]

    with torch.no_grad():
        if evaluation_config.online_dataset_name:
            dataset: Dataset = data.online_datasets[
                evaluation_config.dataset_split][
                    evaluation_config.online_dataset_name]
        else:
            dataset: Dataset = data.static_datasets[
                evaluation_config.dataset_split]

        if evaluation_config.standalone_config:
            _launch_prediction_browser(evaluation_config, first_model, dataset,
                                       data.games)
        else:
            if evaluation_config.gold_forcing_actions:
                # Accuracies of position / stop distributions
                if evaluation_config.loaded_experiment_config.get_target_config(
                ).directly_predict_actions:
                    if evaluation_config.online_dataset_name:
                        evaluate_ids = {
                            (example.example_id, example.step_idx)
                            for example in dataset.step_examples if
                            example.action_annotation.feedback.polarity() > 0
                        }
                        exs = [
                            example for example in dataset.step_examples
                            if (example.example_id,
                                example.step_idx) in evaluate_ids
                        ]
                        exs = [
                            step for step in exs if
                            not step.action_annotation.feedback.is_neutral()
                        ]
                        metrics: Dict[
                            Metric,
                            float] = action_prediction_metrics.compute_expected_feedback(
                                exs, 16, data.games, first_model, False)
                    else:
                        metrics: Dict[
                            Metric,
                            float] = action_prediction_metrics.evaluate_action_predictor(
                                first_model,
                                dataset.step_examples,
                                data.games,
                                16,
                                show_progress=True)
                else:
                    metrics: Dict[
                        Metric,
                        float] = position_prediction_metrics.evaluate_position_predictor(
                            first_model,
                            dataset.step_examples,
                            data.games,
                            16,
                            allow_player_intersections=evaluation_config.
                            rollout_config.game_config.
                            allow_player_intersections,
                            show_progress=True)

                logging.info('gold-forcing results')
                for metric_name, value in metrics.items():
                    if metric_name in PROP_METRICS:
                        logging.info(f'\t{metric_name}\t{(100. * value):.1f}%')
                    else:
                        logging.info(f'\t{metric_name}\t{value}')
            elif evaluation_config.cascaded_evaluation:
                game_examples: Dict[str,
                                    List[Example]] = get_examples_for_games(
                                        dataset.instruction_examples)

                score_deltas: List[float] = list()
                for game_id, examples in tqdm(game_examples.items()):
                    prop_points_scored: Optional[
                        float] = rollout_metrics.cascaded_evaluation(
                            first_model, examples, game_id, data.games,
                            evaluation_config.rollout_config)
                    if prop_points_scored is not None:
                        score_deltas.append(prop_points_scored)
                print(
                    f'Average prop. points scored: {(100. * np.mean(np.array(prop_points_scored))):.1f}'
                )
            else:
                if evaluation_config.sampling:
                    each_run_metrics: Dict[Metric, List[float]] = dict()
                    for i in range(NUM_SAMPLES):
                        logging.info(f'Sampling run {i} of 5')

                        log_path: str = ''
                        if not evaluation_config.rollout_config.ensemble_inference:
                            log_path = os.path.join(
                                evaluation_config.loaded_experiment_config.
                                experiment_metadata.get_experiment_directory(),
                                f'{evaluation_config.dataset_split.shorthand()}_sampling_{i}.log'
                            )

                        this_run_metrics: Dict[Metric, float] = _eval_model(
                            models,
                            dataset,
                            data.games,
                            evaluation_config,
                            logfile_path=log_path)
                        for metric, value in this_run_metrics.items():
                            if metric not in each_run_metrics:
                                if i > 0:
                                    raise ValueError(
                                        f'Unrecognized metric: {metric}')
                                each_run_metrics[metric] = list()
                            each_run_metrics[metric].append(value)

                    for metric, values in each_run_metrics.items():
                        if len(values) != NUM_SAMPLES:
                            raise ValueError(
                                f'Only got {len(values)} values for metric {metric}.'
                            )
                        mean = np.mean(values)
                        std = np.std(values)

                        if metric in PROP_METRICS:
                            logging.info(
                                f'{metric}\t{(100. * mean):.1f}%\t+/-{(100. * std):.1f}%'
                            )
                            for value in values:
                                logging.info(f'\t{100. * value}')

                        else:
                            logging.info(f'{metric}\t{mean}\t+/-{std}')
                            for value in values:
                                logging.info(f'\t{value}')

                if evaluation_config.argmax:
                    agent_rollout_metrics: Dict[Metric, float] = _eval_model(
                        models, dataset, data.games, evaluation_config,
                        os.path.join(
                            evaluation_config.loaded_experiment_config.
                            experiment_metadata.get_experiment_directory(),
                            f'{evaluation_config.dataset_split.shorthand()}_argmax.log'
                        ))

                    logging.info('argmax rollout results')
                    for metric_name, value in agent_rollout_metrics.items():
                        if metric_name in PROP_METRICS:
                            logging.info(
                                f'\t{metric_name}\t{(100. * value):.1f}%')
                        else:
                            logging.info(f'\t{metric_name}\t{value}')
