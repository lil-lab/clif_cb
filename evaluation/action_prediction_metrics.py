"""Evaluates a model that predicts a distribution over actions."""
from __future__ import annotations

import logging
import numpy as np
import torch

from tqdm import tqdm

from data.dataset import GamesCollection
from data.step_example import StepExample
from evaluation.metric import Metric
from inference.predicted_action_distribution import ActionPredictions, PREDICTABLE_ACTIONS
from learning.batching import step_batch
from learning.optimizers import supervised_action_prediction, ips
from model.position_prediction import PositionPredictionModel
from util import util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, List, Set, Tuple
    from environment.position import Position
    from environment.action import Action


def _np_mean_of_list(l: List) -> float:
    return float(np.mean(np.array(l)))


def _action_accuracies(batch: step_batch.StepBatch,
                       predictions: ActionPredictions) -> List[int]:
    accuracies: List[int] = list()

    for i, example in enumerate(batch.original_examples):
        argmax_action: Action = predictions.argmax(i)[0]

        accuracies.append(int(argmax_action == example.target_action))

    return accuracies


def evaluate_action_predictor(
        model: PositionPredictionModel,
        data: List[StepExample],
        games: GamesCollection,
        batch_size: int,
        show_progress: bool = False) -> Dict[Metric, float]:
    model.eval()

    losses: List[float] = list()
    accuracies: List[float] = list()
    entropies: List[float] = list()
    expected_feedback: List[float] = list()

    with torch.no_grad():
        chunks: List[List[StepExample]] = list(util.chunks(data, batch_size))

        if show_progress:
            chunks = tqdm(chunks)

        for batch_idx, batch in enumerate(chunks):
            if not show_progress:
                logging.info(f'{batch_idx + 1} / {len(chunks)}')

            batched: step_batch.StepBatch = step_batch.batch_steps(
                batch,
                games,
                model.get_tokenizer(),
                model.get_environment_batcher(),
                directly_predict_actions=model.directly_predicts_actions())

            batched.to_device()
            predictions: ActionPredictions = model(batched)

            losses.extend(
                supervised_action_prediction.action_prediction_loss(
                    batched, predictions, average=False).tolist())

            expected_feedback.extend(
                supervised_action_prediction.target_action_probabilities(
                    batched, predictions).tolist())

            entropies.extend(
                supervised_action_prediction.action_prediction_entropy(
                    predictions, reduce=False).tolist())

            # Prediction accuracy
            accuracies.extend(_action_accuracies(batched, predictions))

    return {
        Metric.ACTION_LOSS: _np_mean_of_list(losses),
        Metric.ACTION_ACCURACY: _np_mean_of_list(accuracies),
        Metric.ENTROPY: _np_mean_of_list(entropies),
        Metric.EXPECTED_FEEDBACK: _np_mean_of_list(expected_feedback),
        Metric.PROP_GOOD_ACTION: _np_mean_of_list(accuracies)
    }


def compute_expected_feedback(
        step_examples: List[StepExample], evaluation_batch_size: int,
        games: GamesCollection, model: PositionPredictionModel,
        allow_player_intersections: bool) -> Dict[Metric, float]:

    pos_expected_fbs: List[float] = list()
    neg_expected_fbs: List[float] = list()

    prop_neg_switch: List[int] = list()
    prop_pos_same: List[int] = list()

    logging.info(
        f'Evaluating on {len(step_examples)} step examples '
        f'({len([step for step in step_examples if not step.action_annotation.feedback.is_neutral()])} '
        f'have nonzero feedback)')

    ipses: List[float] = list()

    chunks = list(util.chunks(step_examples, evaluation_batch_size))
    for batch in tqdm(chunks):
        batched: step_batch.StepBatch = step_batch.batch_steps(
            batch,
            games,
            model.get_tokenizer(),
            model.get_environment_batcher(),
            batch_targets=False,
            batch_feedback=True,
            directly_predict_actions=True)
        batched.to_device()
        predictions: ActionPredictions = model(batched)

        ips_list: List[float] = ips.get_ips(batched, predictions, True, True,
                                            True)
        ipses.extend(ips_list)

        for i, example in enumerate(batched.original_examples):
            if batched.feedbacks.weights[i].item() == 1:
                sampled_action: Action = batched.feedbacks.sampled_configurations[
                    i]
                current_action_probability: torch.Tensor = predictions.action_probabilities[
                    i][PREDICTABLE_ACTIONS.index(sampled_action)]
                feedback: float = batched.feedbacks.feedbacks[i].item()
                expected_fb: float = feedback * current_action_probability.item(
                )

                if current_action_probability.item() >= 0.5:
                    print(f'{example.example_id},{example.step_idx}')

                obstacles: Set[Position] = games.games[
                    example.game_id].environment.get_obstacle_positions()
                if not allow_player_intersections:
                    obstacles |= {example.state.leader}
                argmax: Action = predictions.argmax(i, example.state.follower,
                                                    obstacles)[0]

                if feedback < 0:
                    neg_expected_fbs.append(expected_fb)
                    if argmax == sampled_action:
                        prop_neg_switch.append(0)
                    else:
                        prop_neg_switch.append(1)
                elif feedback > 0:
                    pos_expected_fbs.append(expected_fb)
                    if argmax == sampled_action:
                        prop_pos_same.append(1)
                    else:
                        prop_pos_same.append(0)

    results_dict: Dict[Metric, float] = dict()

    if pos_expected_fbs + neg_expected_fbs:
        results_dict[Metric.EXPECTED_FEEDBACK] = np.mean(
            np.array(pos_expected_fbs + neg_expected_fbs))
        results_dict[Metric.PROP_GOOD_ACTION] = np.mean(
            np.array(prop_pos_same + prop_neg_switch))
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
    results_dict[Metric.IPS] = np.mean(np.array(ipses))
    results_dict[Metric.NUM_PASSING_IPS] = len([i for i in ipses if i >= 0.5])

    return results_dict
