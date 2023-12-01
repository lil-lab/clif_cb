"""Do a rollout with a model and a set of examples."""
from __future__ import annotations

import random
import torch

from environment.action import Action
from environment.card import CardSelection
from environment.player import Player
from environment.position import Position
from environment.rotation import Rotation
from inference.predicted_action_distribution import ActionPredictions, PREDICTABLE_ACTIONS
from inference.rollout_tracker import RolloutTracker
from inference.top_k_sampling import sample_from_top_k, Sample, NUM_TOP_K_SAMPLES
from inference.vin.vin_model import get_vin_predictions
from learning.batching.step_batch import StepBatch, batch_steps
from simulation.python_game import PythonGame
from util import torch_util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Set, Optional, Tuple, Union
    from config.rollout import RolloutConfig
    from data.dataset import GamesCollection
    from data.example import Example
    from data.step_example import StepExample
    from inference.predicted_voxel import VoxelPredictions
    from inference.vin.vin_model import Cerealbar_VIN
    from model.position_prediction import PositionPredictionModel


def _should_terminate_rollout(rollouts: List[RolloutTracker], num_steps: int,
                              max_num_steps: int):
    stopped: Set[bool] = set([rollout.has_stopped() for rollout in rollouts])
    if len(stopped) == 1 and (True in stopped):
        return True

    if max_num_steps < 0:
        return False

    if num_steps >= max_num_steps:
        return True


def get_argmax_configurations(
        predictions: VoxelPredictions, batch: StepBatch,
        allow_player_intersections: bool) -> List[Player]:
    default_obstacle_mask: torch.Tensor = batch.environment_state.static_info.obstacle_mask
    if not allow_player_intersections:
        # Also include the leader position
        default_obstacle_mask = batch.environment_state.get_all_obstacles()

    argmax_configs: List[Player] = list()

    # Set the goals by getting the argmax configuration.
    for i in range(predictions.get_batch_size()):
        argmax_item: Union[Action, Tuple[
            Position, Rotation]] = predictions.argmax(
                i,
                allow_stop=False,
                obstacle_mask=default_obstacle_mask[i],
                allow_copy=batch.original_examples[i].can_copy)[0]

        if argmax_item == Action.COPY:
            # Resolve this to the previous player configuration.
            argmax_configs.append(
                batch.original_examples[i].previous_targets[-1])
        else:
            argmax_pos, argmax_rot = argmax_item
            argmax_configs.append(Player(True, argmax_pos, argmax_rot))
    return argmax_configs


def get_ensemble_model_predictions(
    step_examples: List[StepExample], games: GamesCollection,
    models: List[PositionPredictionModel], softmax_normalization: bool,
    use_voting: bool, allow_player_intersections: bool,
    current_rollouts: List[RolloutTracker]
) -> Tuple[ActionPredictions, List[ActionPredictions]]:
    batches: List[StepBatch] = [
        batch_steps(step_examples,
                    games,
                    model.get_tokenizer(),
                    model.get_environment_batcher(),
                    batch_targets=False,
                    directly_predict_actions=True) for model in models
    ]

    # TODO: Use Ray for this
    predictions: List[ActionPredictions] = list()
    for i, model in enumerate(models):
        batches[i].to_device()
        predictions.append(model(batches[i]))

    if use_voting:
        # Argmax for each model.
        action_scores: torch.Tensor = torch.zeros(
            (len(current_rollouts), len(PREDICTABLE_ACTIONS)),
            device=torch_util.DEVICE)
        for i, prediction in enumerate(predictions):
            for j, rollout in enumerate(current_rollouts):
                obstacle_positions: Set[Position] = games.games[
                    rollout.get_game_id()].environment.get_obstacle_positions(
                    )

                if not allow_player_intersections:
                    obstacle_positions |= {
                        rollout.get_current_state().leader.position
                    }

                action = prediction.argmax(
                    j,
                    current_agent=rollout.get_current_state().follower,
                    obstacle_positions=obstacle_positions,
                    sample=False)[0]

                action_scores[j][PREDICTABLE_ACTIONS.index(action)] += 1.
    else:
        # Boltzman addition
        action_scores: torch.Tensor = torch.sum(torch.stack(
            [prediction.action_probabilities for prediction in predictions]),
                                                dim=0)

    if softmax_normalization:
        action_probs = torch.nn.Softmax(dim=1)(action_scores)
    else:
        normalizations = torch.sum(action_scores, dim=1)
        action_probs = action_scores / normalizations.unsqueeze(1)

    return ActionPredictions(action_probs), predictions


def get_single_model_predictions(
        step_examples: List[StepExample], games: GamesCollection,
        model: PositionPredictionModel) -> ActionPredictions:
    step_batch: StepBatch = batch_steps(step_examples,
                                        games,
                                        model.get_tokenizer(),
                                        model.get_environment_batcher(),
                                        batch_targets=False,
                                        directly_predict_actions=True)
    step_batch.to_device()
    return model(step_batch)


def _get_and_execute_action(
        step_examples: List[StepExample], games: GamesCollection,
        models: List[PositionPredictionModel], config: RolloutConfig,
        not_stopped_rollouts: List[RolloutTracker],
        forced_actions: List[Optional[Action]], sample_actions: bool):

    if config.ensemble_inference:
        predictions: ActionPredictions = get_ensemble_model_predictions(
            step_examples, games, models,
            config.ensemble_softmax_normalization, config.voting_ensemble,
            config.game_config.allow_player_intersections,
            not_stopped_rollouts)[0]
    else:
        predictions: ActionPredictions = get_single_model_predictions(
            step_examples, games, models[0])

    # Take the action in each example
    for i, rollout in enumerate(not_stopped_rollouts):
        if forced_actions[i]:
            # Force the action instead.
            action_to_execute = forced_actions[i]
            if action_to_execute == Action.STOP:
                rollout.set_stop_forced(True)
        else:
            obstacle_positions: Set[Position] = games.games[
                rollout.get_game_id()].environment.get_obstacle_positions()

            if not config.game_config.allow_player_intersections:
                obstacle_positions |= {
                    rollout.get_current_state().leader.position
                }

            action_to_execute = predictions.argmax(
                i,
                current_agent=rollout.get_current_state().follower,
                obstacle_positions=obstacle_positions,
                sample=sample_actions)[0]

        rollout.execute_action(action_to_execute,
                               target_config=None,
                               allow_no_config=True)


def _get_argmax_voxel(
        predictions: VoxelPredictions, step_batch: StepBatch,
        model: PositionPredictionModel, rollouts: List[RolloutTracker],
        forced_actions: List[Optional[Action]], games: GamesCollection,
        config: RolloutConfig) -> List[Tuple[Action, Optional[Player]]]:
    vin_predictions: Optional[torch.Tensor] = None
    argmax_players: Optional[List[Player]] = None
    if not config.restrict_to_neighbors:
        argmax_players = get_argmax_configurations(
            predictions, step_batch,
            config.game_config.allow_player_intersections)

        # Run the VIN.
        vin_predictions: List[Set[Action]] = get_vin_predictions(
            model.get_vin(),
            [rollout.get_current_state() for rollout in rollouts], step_batch,
            config.game_config.allow_player_intersections, argmax_players)

    obstacle_mask: torch.Tensor = step_batch.environment_state.static_info.obstacle_mask
    if not config.game_config.allow_player_intersections:
        # Also include the leader position
        obstacle_mask = step_batch.environment_state.get_all_obstacles()

    # Take the action in each example
    argmaxes: List[Tuple[Action, Optional[Player]]] = list()

    for i, rollout in enumerate(rollouts):
        argmax_voxel: Optional[Player] = None
        if forced_actions[i]:
            # Force the action instead.
            argmax_action = forced_actions[i]
            if argmax_action == Action.STOP:
                rollout.set_stop_forced(True)
        else:
            if config.restrict_to_neighbors:
                obstacle_pos: Set[Position] = games.games[rollout.get_game_id(
                )].environment.get_obstacle_positions() | {
                    rollout.get_current_state().leader.position
                }
                argmax_action, _, argmax_voxel = predictions.argmax_neighbor_action(
                    i,
                    rollout.get_current_state().follower,
                    obstacle_pos,
                    can_copy=False)
            else:
                argmax_action, argmax_prob = predictions.argmax(
                    i,
                    obstacle_mask=obstacle_mask[i],
                    allow_copy=rollout.can_copy())
                if argmax_action == Action.STOP:
                    argmax_action = Action.STOP
                elif argmax_action == Action.COPY:
                    # Grab previous target.
                    argmax_voxel = rollout.get_targets()[-1]

                    # Run the VIN again.
                    this_batch = batch_steps([step_batch.original_examples[i]],
                                             games,
                                             model.get_tokenizer(),
                                             model.get_environment_batcher(),
                                             batch_targets=False)
                    this_batch.to_device()
                    this_example_vin_predictions: Set[
                        Action] = get_vin_predictions(
                            model.get_vin(), [rollout.get_current_state()],
                            this_batch,
                            config.game_config.allow_player_intersections,
                            [argmax_voxel])[0]
                    argmax_action = random.choice(
                        list(this_example_vin_predictions))
                else:
                    argmax_action = random.choice(list(vin_predictions[i]))
                    argmax_voxel = argmax_players[i]
        argmaxes.append((argmax_action, argmax_voxel))

    return argmaxes


def _top_k_sample_voxels(
    predictions: VoxelPredictions, step_batch: StepBatch,
    config: RolloutConfig, vin: Cerealbar_VIN, rollouts: List[RolloutTracker],
    forced_actions: List[Optional[Action]]
) -> List[Tuple[Action, Optional[Player]]]:
    obstacle_mask: torch.Tensor = step_batch.environment_state.static_info.obstacle_mask
    if not config.game_config.allow_player_intersections:
        obstacle_mask = step_batch.environment_state.get_all_obstacles()

    samples: List[Sample] = sample_from_top_k(
        predictions, NUM_TOP_K_SAMPLES, obstacle_mask, vin,
        [rollout.get_current_state() for rollout in rollouts], step_batch,
        config.game_config.allow_player_intersections)

    results: List[Tuple[Action, Optional[Player]]] = list()
    for i in range(step_batch.get_batch_size()):
        if forced_actions[i]:
            if forced_actions[i] == Action.STOP:
                results.append((Action.STOP, None))
            else:
                raise ValueError(
                    f'Should not force non-stop action: {forced_actions[i]}')
        else:
            results.append((samples[i].action, samples[i].target_voxel))

    return results


def _sample_and_execute_voxel(step_examples: List[StepExample],
                              games: GamesCollection,
                              model: PositionPredictionModel,
                              config: RolloutConfig,
                              not_stopped_rollouts: List[RolloutTracker],
                              forced_actions: List[Optional[Action]]):
    step_batch: StepBatch = batch_steps(step_examples,
                                        games,
                                        model.get_tokenizer(),
                                        model.get_environment_batcher(),
                                        batch_targets=False)
    step_batch.to_device()
    predictions: VoxelPredictions = model(step_batch)

    if config.use_sampling:
        if config.restrict_to_neighbors:
            raise NotImplementedError(
                'Restricting to neighbors not supported with sampling yet.')
        if model.uses_copy():
            raise NotImplementedError(
                'Using copy is not supported with sampling yet.')

        argmax_results: List[Tuple[Action,
                                   Optional[Player]]] = _top_k_sample_voxels(
                                       predictions, step_batch, config,
                                       model.get_vin(), not_stopped_rollouts,
                                       forced_actions)
    else:
        argmax_results: List[Tuple[Action,
                                   Optional[Player]]] = _get_argmax_voxel(
                                       predictions, step_batch, model,
                                       not_stopped_rollouts, forced_actions,
                                       games, config)

    for i, rollout in enumerate(not_stopped_rollouts):
        argmax_action, argmax_voxel = argmax_results[i]
        rollout.execute_action(argmax_action, argmax_voxel)


def single_rollout(model: PositionPredictionModel, tracker: RolloutTracker,
                   games: GamesCollection, config: RolloutConfig):
    if model.directly_predicts_actions():
        raise NotImplementedError(
            'Need to implement this for direct action prediction.')

    num_steps: int = 0
    while not _should_terminate_rollout([tracker], num_steps,
                                        config.max_num_steps):
        forced_action: Optional[Action] = None
        if num_steps == config.max_num_steps - 1:
            forced_action: bool = Action.STOP

        step_example: StepExample = tracker.get_current_step_example()

        _sample_and_execute_voxel([step_example], games, model, config,
                                  [tracker], [forced_action])


def batch_rollout(models: List[PositionPredictionModel],
                  examples: List[Example], games: GamesCollection,
                  config: RolloutConfig,
                  sample_actions: bool) -> List[RolloutTracker]:
    uses_copy: bool = models[0].uses_copy()
    directly_predicts_actions: bool = models[0].directly_predicts_actions()
    for m in models:
        if m.uses_copy() != uses_copy:
            raise ValueError
        if m.directly_predicts_actions() != directly_predicts_actions:
            raise ValueError

    rollouts: List[RolloutTracker] = [
        RolloutTracker(
            example.instruction, example.get_game_id(), example.example_id,
            example.get_initial_observation(), example.get_initial_state(),
            PythonGame(games.games[example.get_game_id()].environment,
                       example.get_initial_state(), example.instruction,
                       example.num_first_turn_steps, config.game_config,
                       example.leader_actions, example.expected_sets),
            uses_copy) for example in examples
    ]

    num_steps: int = 0
    while not _should_terminate_rollout(rollouts, num_steps,
                                        config.max_num_steps):
        not_stopped_rollouts: List[RolloutTracker] = [
            rollout for rollout in rollouts if not rollout.has_stopped()
        ]

        forced_actions: Optional[List[Action]] = [
            None for _ in not_stopped_rollouts
        ]
        # Decide whether an action should be forced (e.g., end of action sequence)
        if num_steps == config.max_num_steps - 1:
            forced_actions = [Action.STOP for _ in not_stopped_rollouts]

        # Predict an action for each example
        step_examples: List[StepExample] = [
            rollout.get_current_step_example()
            for rollout in not_stopped_rollouts
        ]

        if directly_predicts_actions:
            _get_and_execute_action(step_examples, games, models, config,
                                    not_stopped_rollouts, forced_actions,
                                    sample_actions)
        else:
            if sample_actions:
                raise NotImplementedError(
                    'Sampling for voxel prediction model in evaluation is not yet supported.'
                )

            if config.ensemble_inference:
                raise NotImplementedError

            _sample_and_execute_voxel(step_examples, games, models[0], config,
                                      not_stopped_rollouts, forced_actions)

        num_steps += 1

    # Make sure all leaders have been executed for every rollout
    for rollout in rollouts:
        rollout.finish_all_leader_actions()

    return rollouts
