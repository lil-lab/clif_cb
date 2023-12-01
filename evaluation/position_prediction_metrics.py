from typing import Union, Tuple, List, Dict

import logging
import numpy as np
import time
import torch

from tqdm import tqdm

from data.dataset import GamesCollection
from data.step_example import StepExample
from environment.action import Action
from environment.player import Player
from environment.position import Position, out_of_bounds
from environment.rotation import ROTATIONS, Rotation
from evaluation.metric import Metric
from inference.predicted_voxel import VoxelPredictions
from inference.top_k_sampling import get_action_distributions_up_to_k
from inference.vin.vin_model import get_vin_predictions
from learning.batching import step_batch
from learning.optimizers import supervised_position_prediction
from model.position_prediction import PositionPredictionModel
from util import util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from environment.state import State
    from typing import List, Optional, Set

TOP_K_SAMPLE_SIZE: int = 3


def correct_prediction(prediction: Union[Action, Tuple[Position, Rotation]],
                       example_step: StepExample) -> bool:
    if prediction == Action.STOP:
        return example_step.target_action == Action.STOP
    elif prediction == Action.COPY:
        return example_step.should_copy
    else:
        pos, rot = prediction

        return Player(True, pos,
                      rot) in example_step.possible_target_configurations


def _voxel_accuracies(
    batch: step_batch.StepBatch, predictions: VoxelPredictions
) -> Tuple[List[int], List[Union[Action, Tuple[Position, Rotation]]]]:
    accuracies: List[int] = list()
    argmaxes: List[Union[Action, Tuple[Position, Rotation]]] = list()

    for i, example in enumerate(batch.original_examples):
        argmax_item: Union[Action,
                           Tuple[Position, Rotation]] = predictions.argmax(
                               i,
                               allow_copy=predictions.copy_probabilities
                               is not None)[0]
        accuracies.append(int(correct_prediction(argmax_item, example)))
        argmaxes.append(argmax_item)

    return accuracies, argmaxes


def _np_mean_of_list(l: List) -> float:
    return float(np.mean(np.array(l)))


def evaluate_position_predictor(
        model: PositionPredictionModel,
        data: List[StepExample],
        games: GamesCollection,
        batch_size: int,
        allow_player_intersections: bool,
        show_progress: bool = False) -> Dict[Metric, float]:
    model.eval()

    losses: List[float] = list()
    voxel_and_stop_accuracies: List[int] = list()
    position_accuracies_sum: List[int] = list()
    position_accuracies_voxel: List[int] = list()
    entropies: List[float] = list()

    # "Expected feedback": basically just the probabilities of the gold actions according to the model. Correlated
    # directly with loss, but interpretable as feedback (feedback is 1.0 for every action in supervised data.).
    expected_feedback: List[float] = list()
    good_actions: List[int] = list()

    correct_voxel_subsequent: List[int] = list()
    predicted_obstacle: List[int] = list()
    restr_correct_action: List[int] = list()

    num_correct_stops: int = 0
    num_target_stops: int = 0.00001  # Tiny number in case this is zero.
    num_stop_preds: float = 0.00001  # Tiny number in case this is zero.

    with torch.no_grad():
        chunks: List[List[StepExample]] = list(util.chunks(data, batch_size))

        if show_progress:
            chunks = tqdm(chunks)

        prev_time = 0
        for batch_idx, batch in enumerate(chunks):
            if not show_progress:
                logging.info(
                    f'{batch_idx + 1} / {len(chunks)} (prev. batch took {prev_time} s)'
                )

            st = time.time()
            batched: step_batch.StepBatch = step_batch.batch_steps(
                batch,
                games,
                model.get_tokenizer(),
                model.get_environment_batcher(),
                directly_predict_actions=model.directly_predicts_actions())

            batched.to_device()
            predictions: VoxelPredictions = model(batched)

            # Loss and entropy. Always set use_ips to false, even when the training uses IPS.
            losses.extend(
                supervised_position_prediction.voxel_prediction_loss(
                    batched, predictions, use_ips=False,
                    average=False).tolist())

            expected_feedback.extend(
                supervised_position_prediction.voxel_prediction_probabilities(
                    batched, predictions).tolist())

            entropies.extend(
                supervised_position_prediction.voxel_prediction_entropy(
                    predictions, reduce=False).tolist())

            # Prediction accuracy
            accuracies, argmaxes = _voxel_accuracies(batched, predictions)
            voxel_and_stop_accuracies.extend(accuracies)

            # Run the VIN
            target_players: List[Optional[Player]] = list()
            for argmax in argmaxes:
                if argmax == Action.STOP:
                    target_players.append(None)
                else:
                    pos, rot = argmax
                    target_players.append(Player(True, pos, rot))

            current_states: List[State] = [
                example.state for example in batched.original_examples
            ]
            argmax_actions: List[Optional[Set[Action]]] = get_vin_predictions(
                model.get_vin(), current_states, batched,
                allow_player_intersections, target_players)

            obstacle_mask: torch.Tensor = batched.environment_state.static_info.obstacle_mask
            if not allow_player_intersections:
                obstacle_mask = batched.environment_state.get_all_obstacles()
            """
            action_distribution_estimates: Dict[
                int, List[Dict[Action, Tuple[float, Optional[Tuple[Position, Rotation]]]]]] = \
                get_action_distributions_up_to_k(
                    predictions, TOP_K_SAMPLE_SIZE, obstacle_mask, model.get_vin(), current_states, batched,
                    allow_player_intersections, randomly_choose_action=False, max_probs_for_action_voxels=False)
            """

            # Accuracy of position prediction of the argmax example
            for i, example in enumerate(batched.original_examples):
                # Accuracy of the argmax action through the VIN
                if example.target_action == Action.STOP:
                    #                    logging.info(
                    #                        f'target was stop; prediction was {argmaxes[i]}')
                    good_actions.append(int(argmaxes[i] == Action.STOP))
                elif argmaxes[i] == Action.STOP:
                    #                    logging.info(f'argmax was stop but target was not')
                    # Target was not stop, but prediction was, so it's wrong.
                    good_actions.append(0)
                else:
                    #                    logging.info(
                    #                        f'is target in argmax actions? {example.target_action in argmax_actions[i]}'
                    #                    )
                    # If the argmax action is still viable, then it's a good prediction for a voxel.
                    good_actions.append(
                        int(example.target_action in argmax_actions[i]))

                # Probability of target action
                """
                if example.target_action in action_distribution_estimates[
                        TOP_K_SAMPLE_SIZE][i]:
                    action_probability: float = action_distribution_estimates[
                        TOP_K_SAMPLE_SIZE][i][example.target_action][0]
                else:
                    # Not in the beam anymore; conservatively, give it 0 points for this.
                    action_probability: float = 0.
                action_efs.append(action_probability)
                """

                # Only compute for examples whose target is not STOP.
                if example.target_action != Action.STOP:
                    if argmaxes[i] == Action.STOP:
                        position_accuracies_sum.append(0)
                        position_accuracies_voxel.append(0)
                    elif argmaxes[i] == Action.COPY:
                        position_accuracies_sum.append(int(
                            example.should_copy))
                        position_accuracies_voxel.append(
                            int(example.should_copy))
                    else:
                        pos, rot = argmaxes[i]

                        target_positions: Set[
                            Position] = example.get_possible_target_positions(
                            )

                        # Was the argmax voxel's position correct?
                        position_accuracies_voxel.append(
                            int(pos in target_positions))

                        # Was the argmax position (summing over rotation channels) correct?
                        position_accuracies_sum.append(
                            int(
                                predictions.argmax_position(
                                    i, predictions.copy_probabilities
                                    is not None) in target_positions))

                        argmax_voxel: Player = Player(True, pos, rot)
                        if argmax_voxel in example.possible_target_configurations:
                            correct_voxel_subsequent.append(
                                int(argmax_voxel ==
                                    example.next_target_configuration))

                        predicted_obstacle.append(
                            int(pos in games.games[example.game_id].
                                environment.get_obstacle_positions()
                                or pos == example.state.leader.position
                                or out_of_bounds(pos)))

                # Restricted output space: only reachable positions
                pred_action: Action = predictions.argmax_neighbor_action(
                    i, example.state.follower, set(),
                    predictions.copy_probabilities is not None)[0]
                restr_correct_action.append(
                    int(pred_action == example.target_action))

            # Accuracy of STOP (precision and recall)
            for i, example in enumerate(batched.original_examples):
                if argmaxes[i] == Action.STOP:
                    num_stop_preds += 1
                    if example.target_action == Action.STOP:
                        num_correct_stops += 1

            num_target_stops += torch.sum(
                batched.target.action_type_labels).item()

            prev_time = time.time() - st

    return {
        Metric.VOXEL_LOSS: _np_mean_of_list(losses),
        Metric.VOXEL_ACCURACY: _np_mean_of_list(voxel_and_stop_accuracies),
        Metric.STOP_RECALL: num_correct_stops / num_target_stops,
        Metric.STOP_PRECISION: num_correct_stops / num_stop_preds,
        Metric.POSITION_ACCURACY_SUM:
        _np_mean_of_list(position_accuracies_sum),
        Metric.POSITION_ACCURACY_VOXEL:
        _np_mean_of_list(position_accuracies_voxel),
        Metric.ENTROPY: _np_mean_of_list(entropies),
        Metric.PROP_ARGMAX_SUBSEQUENT:
        _np_mean_of_list(correct_voxel_subsequent),
        Metric.PROP_PRED_OBSTACLE: _np_mean_of_list(predicted_obstacle),
        Metric.NEIGHBOR_ACTION_ACC: _np_mean_of_list(restr_correct_action),
        Metric.EXPECTED_FEEDBACK: _np_mean_of_list(expected_feedback),
        Metric.PROP_GOOD_ACTION: _np_mean_of_list(good_actions)
    }


def compute_expected_feedback(
        model: PositionPredictionModel, step_examples: List[StepExample],
        games: GamesCollection, evaluation_batch_size: int,
        allow_player_intersections: bool) -> Dict[Metric, float]:
    """
    Gets probability of voxel predictions over all examples. For negative examples, it uses the inverse predicted 
    probability.
    """
    pos_expected_fbs: List[float] = list()
    neg_expected_fbs: List[float] = list()

    prop_good_actions: List[int] = list()
    prop_neg_switch: List[int] = list()
    prop_pos_same: List[int] = list()

    for batch in util.chunks(step_examples, evaluation_batch_size):
        batched: step_batch.StepBatch = step_batch.batch_steps(
            batch,
            games,
            model.get_tokenizer(),
            model.get_environment_batcher(),
            batch_targets=False,
            batch_feedback=True)

        batched.to_device()
        predictions: VoxelPredictions = model(batched)

        # Get argmaxes and VIN predictions and give points depending on whether the argmax action is good or bad
        target_players: List[Optional[Player]] = list()

        for i, example in enumerate(batched.original_examples):
            argmax_item: Union[Action,
                               Tuple[Position, Rotation]] = predictions.argmax(
                                   i,
                                   allow_copy=predictions.copy_probabilities
                                   is not None)[0]
            if argmax_item == Action.STOP:
                target_players.append(None)
            else:
                pos, rot = argmax_item
                target_players.append(Player(True, pos, rot))

        current_states: List[State] = [
            example.state for example in batched.original_examples
        ]

        argmax_actions: List[Optional[Set[Action]]] = get_vin_predictions(
            model.get_vin(), current_states, batched,
            allow_player_intersections, target_players)
        for i, target in enumerate(target_players):
            if target is None:
                argmax_actions[i] = {Action.STOP}

        # Action distributions
        obstacle_mask: torch.Tensor = batched.environment_state.static_info.obstacle_mask
        if not allow_player_intersections:
            obstacle_mask = batched.environment_state.get_all_obstacles()
        """
        all_action_distribution_estimates:  Dict[int, List[Dict[Action, Tuple[float, Optional[Tuple[Position,
                                                                                                 Rotation]]]]]] = \
            get_action_distributions_up_to_k(
                predictions, TOP_K_SAMPLE_SIZE, obstacle_mask, model.get_vin(), current_states, batched,
                allow_player_intersections, randomly_choose_action=False, max_probs_for_action_voxels=False)
        action_distribution_estimates = all_action_distribution_estimates[
            TOP_K_SAMPLE_SIZE]
        """

        for i in range(batched.get_batch_size()):
            # Only considers examples that are from the online feedback set.
            if batched.original_examples[i].is_converted_feedback_example:
                continue

            sample: Union[Action,
                          Player] = batched.feedbacks.sampled_configurations[i]

            if sample == Action.STOP:
                voxel_probability: torch.Tensor = predictions.stop_probabilities[
                    i]
            else:
                pos: Position = sample.position
                rot: Rotation = sample.rotation

                voxel_probability: torch.Tensor = predictions.voxel_probabilities[
                    i][ROTATIONS.index(rot)][pos.x][pos.y]

            feedback: float = batched.feedbacks.feedbacks[i].item()
            original_action: Action = batched.original_examples[
                i].sampled_action

            if batched.feedbacks.weights[i].item() == 1:
                # Only include feedbacks where the weight is 1 (i.e., not in the COACH trace).
                expected_fb: float = feedback * voxel_probability.item()
                """
                action_probability: float = 0.
                if original_action in action_distribution_estimates[i]:
                    action_probability: float = action_distribution_estimates[
                        i][original_action][0]

                expected_action_fb: float = feedback * action_probability
                if feedback < 0:
                    neg_action_efs.append(expected_action_fb)
                elif feedback > 0:
                    pos_action_efs.append(expected_action_fb)
                """

                if feedback < 0:
                    neg_expected_fbs.append(expected_fb)
                    if original_action in argmax_actions[i]:
                        # Not good: should not still be possible to take this action.
                        #                        logging.info(
                        #                            f'negative feedback: original action is still an argmax action ({original_action} vs. {argmax_actions[i]})'
                        #                        )
                        prop_good_actions.append(0)
                        prop_neg_switch.append(0)
                    else:
                        # Good: no longer possible to take this action.
                        #                        logging.info(
                        #                            f'negative feedback: original action is no longer an argmax action({original_action} vs. {argmax_actions[i]}).'
                        #                        )
                        prop_good_actions.append(1)
                        prop_neg_switch.append(1)
                elif feedback > 0:
                    pos_expected_fbs.append(expected_fb)
                    if original_action in argmax_actions[i]:
                        # Good: should still be able to take this action.
                        #                        logging.info(
                        #                            f'positive feedback: original action is still an argmax action({original_action} vs. {argmax_actions[i]}).'
                        #                        )
                        prop_good_actions.append(1)
                        prop_pos_same.append(1)
                    else:
                        # Not good: no longer possible to take this action.
                        #                        logging.info(
                        #                            f'positive feedback: original action is no longer an argmax action({original_action} vs. {argmax_actions[i]}).'
                        #                        )
                        prop_good_actions.append(0)
                        prop_pos_same.append(0)

    results_dict: Dict[Metric, float] = dict()
    if pos_expected_fbs + neg_expected_fbs:
        results_dict[Metric.EXPECTED_FEEDBACK] = np.mean(
            np.array(pos_expected_fbs + neg_expected_fbs))
        results_dict[Metric.PROP_GOOD_ACTION] = np.mean(
            np.array(prop_good_actions))
    if pos_expected_fbs:
        results_dict[Metric.EXPECTED_FEEDBACK_POS] = np.mean(
            np.array(pos_expected_fbs))
        results_dict[Metric.PROP_POS_FB_SAME] = np.mean(
            np.array(prop_pos_same))
    if neg_expected_fbs:
        results_dict[Metric.EXPECTED_FEEDBACK_NEG] = np.mean(
            np.array(neg_expected_fbs))
        results_dict[Metric.PROP_NEG_FB_SWITCH] = np.mean(
            np.array(prop_neg_switch))

    return results_dict
