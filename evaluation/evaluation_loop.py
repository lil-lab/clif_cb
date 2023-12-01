import logging
import os
import time
from typing import Dict, Optional, List, Tuple

import numpy as np
import ray

from datetime import datetime
from torch import nn

from config.rollout import RolloutConfig
from data.dataset import Dataset, GamesCollection
from data.dataset_split import DatasetSplit
from data.example import Example
from data.step_example import StepExample
from evaluation import action_prediction_metrics, position_prediction_metrics, rollout_metrics
from evaluation.metric import Metric, InstructionFollowingErrorType
from learning.parameter_server import ParameterServer

from learning.patience_tracker import PatienceTracker, EF_NAME, PG_ACTION, COMBINED_ACTION_EF, PG_ACTION_POS, \
    PG_ACTION_NEG
from model.position_prediction import PositionPredictionModel
from util import torch_util, log, wandb_util

NUM_SAMPLES: int = 5


def _check_improvements_and_save(metrics_to_improve: Dict[str, float],
                                 patience_tracker: PatienceTracker,
                                 epoch_idx: int,
                                 model: PositionPredictionModel,
                                 save_directory: str, always_save: bool,
                                 run) -> Optional[str]:
    best_save_file: Optional[str] = None

    improved_metrics: List[str] = ray.get(
        patience_tracker.improved_metric_after_epoch.remote(
            metrics_to_improve))

    if epoch_idx == 0 or improved_metrics or always_save:
        model.save(save_directory, improved_metrics, epoch_idx)
        logging.info('Saved model -- improved metrics:')
        for metric in improved_metrics:
            logging.info(f'\t{metric}\t{metrics_to_improve[metric]}')
        logging.info('All results:')
        for metric in metrics_to_improve:
            logging.info(f'\t{metric}\t{metrics_to_improve[metric]}')

    logging.info(f'Done with eval for epoch #{epoch_idx}')

    for metric_name, countdown in ray.get(
            patience_tracker.get_countdowns.remote()).items():
        run.log({f'countdowns/{metric_name}': countdown}, step=epoch_idx)

    patience_tracker.increment_epochs.remote()

    return best_save_file


def _do_sampling_rollouts(
    model: PositionPredictionModel, instructions: List[Example],
    games: GamesCollection, rollout_config: RolloutConfig,
    evaluation_batch_size: int
) -> Tuple[Dict[Metric, float], Dict[InstructionFollowingErrorType, float]]:
    sampling_results: List[Tuple[Dict[Metric, float],
                                 Dict[InstructionFollowingErrorType,
                                      float]]] = list()
    for i in range(NUM_SAMPLES):
        sampling_results.append(
            rollout_metrics.evaluate_position_predictor([model],
                                                        instructions,
                                                        games,
                                                        rollout_config,
                                                        evaluation_batch_size,
                                                        sample_actions=True))
    rollout_eval: Dict[Metric, float] = dict()
    errors: Dict[InstructionFollowingErrorType, float] = dict()

    for metric_name in sampling_results[0][0]:
        metric_results: List[float] = list()
        for iter_results in sampling_results:
            result = iter_results[0][metric_name]
            metric_results.append(result)
        rollout_eval[metric_name] = np.mean(np.array(metric_results))

    for metric_name in sampling_results[0][1]:
        metric_results: List[float] = list()
        for iter_results in sampling_results:
            result = iter_results[1][metric_name]
            metric_results.append(result)
        errors[metric_name] = np.mean(np.array(metric_results))

    return rollout_eval, errors


def _evaluation_round(model: PositionPredictionModel,
                      patience_tracker: PatienceTracker,
                      validation_data: Dict[DatasetSplit, Dataset],
                      online_validation_data: Optional[Dict[str, Dataset]],
                      games: GamesCollection, evaluation_batch_size: int,
                      save_directory: str, rollout_config: RolloutConfig,
                      always_save: bool, ordered_online_datasets: List[str],
                      run):
    epoch_idx = ray.get(patience_tracker.get_num_epochs.remote())

    # Evaluate the model
    logging.info(f'Evaluating model at epoch {epoch_idx}.')

    metrics_to_improve: Dict[str, float] = dict()

    val_expected_feedback: Optional[float] = None
    val_prop_good: Optional[float] = None
    val_action_ef: Optional[float] = None

    # Accuracies of position / stop distributions
    for split, dataset in validation_data.items():
        logging.info(f'Evaluating position predictor for split {split}')

        st = time.time()
        if model.directly_predicts_actions():
            gold_forcing_metrics: Dict[
                Metric,
                float] = action_prediction_metrics.evaluate_action_predictor(
                    model, dataset.step_examples, games, evaluation_batch_size)
            metrics_to_improve[
                f'{split}_{Metric.ACTION_LOSS}'] = -gold_forcing_metrics[
                    Metric.ACTION_LOSS]
            metrics_to_improve[
                f'{split}_{Metric.ACTION_ACCURACY}'] = gold_forcing_metrics[
                    Metric.ACTION_ACCURACY]
        else:
            gold_forcing_metrics: Dict[
                Metric,
                float] = position_prediction_metrics.evaluate_position_predictor(
                    model, dataset.step_examples, games, evaluation_batch_size,
                    rollout_config.game_config.allow_player_intersections)
            metrics_to_improve[
                f'{split}_{Metric.VOXEL_LOSS}'] = -gold_forcing_metrics[
                    Metric.VOXEL_LOSS]
            metrics_to_improve[
                f'{split}_{Metric.VOXEL_ACCURACY}'] = gold_forcing_metrics[
                    Metric.VOXEL_ACCURACY]
        gold_forcing_time = time.time() - st

        # Accuracies of rollouts
        logging.info(f'Evaluating rollouts for split {split}')
        st = time.time()
        argmax_agent_rollout_metrics, argmax_error_categories = rollout_metrics.evaluate_position_predictor(
            [model], dataset.instruction_examples, games, rollout_config,
            evaluation_batch_size)
        argmax_rollout_time = time.time() - st

        st = time.time()
        sampling_rollout_results, sampling_rollout_errors = _do_sampling_rollouts(
            model, dataset.instruction_examples, games, rollout_config,
            evaluation_batch_size)
        sampling_val_time = time.time() - st

        run.log({
            f'validation_errors/{split}_{error_name}':
            100. * value / len(dataset.instruction_examples)
            for error_name, value in argmax_error_categories.items()
        }, step=epoch_idx)

        run.log(
            {
                f'validation/{split}_{metric}': value
                for metric, value in gold_forcing_metrics.items()
            },
            step=epoch_idx)
        run.log(
            {
                'timing/argmax_val_time': argmax_rollout_time,
                'timing/gold_forcing_val_time': gold_forcing_time,
                'timing/sampling_val_time': sampling_val_time
            },
            step=epoch_idx)
        run.log(
            {
                f'rollout_validation/{split}_{metric}': value
                for metric, value in argmax_agent_rollout_metrics.items()
            },
            step=epoch_idx)
        run.log(
            {
                f'sampling_validation/{split}_{metric}': value
                for metric, value in sampling_rollout_results.items()
            },
            step=epoch_idx)
        run.log(
            {
                f'sampling_validation_errors/{split}_{error_name}':
                value / len(dataset.instruction_examples)
                for error_name, value in sampling_rollout_errors.items()
            },
            step=epoch_idx)

        metrics_to_improve[
            f'{split}_{Metric.EXPECTED_FEEDBACK}'] = gold_forcing_metrics[
                Metric.EXPECTED_FEEDBACK]
        metrics_to_improve[
            f'{split}_{Metric.CARD_ACCURACY}'] = argmax_agent_rollout_metrics[
                Metric.CARD_ACCURACY]
        metrics_to_improve[
            f'{split}_{Metric.SUCCESS_STOP_DISTANCE}'] = argmax_agent_rollout_metrics[
                Metric.SUCCESS_STOP_DISTANCE]

        metrics_to_improve[
            f'{split}_sampling_{Metric.SUCCESS_STOP_DISTANCE}'] = sampling_rollout_results[
                Metric.SUCCESS_STOP_DISTANCE]

        metrics_to_improve[
            f'{split}_{Metric.PROP_GOOD_ACTION}'] = gold_forcing_metrics[
                Metric.PROP_GOOD_ACTION]
        if Metric.ACTION_EF in gold_forcing_metrics:
            metrics_to_improve[
                f'{split}_{Metric.ACTION_EF}'] = gold_forcing_metrics[
                    Metric.ACTION_EF]

        if split in {DatasetSplit.VALIDATION, DatasetSplit.TRAIN}:
            val_expected_feedback = gold_forcing_metrics[
                Metric.EXPECTED_FEEDBACK]

            val_prop_good = gold_forcing_metrics[Metric.PROP_GOOD_ACTION]

            if Metric.ACTION_EF in gold_forcing_metrics:
                val_action_ef = gold_forcing_metrics[Metric.ACTION_EF]

    last_round_expected_feedback: Optional[float] = None
    prev_round_feedbacks: List[float] = list()

    last_round_prop_good: Optional[float] = None
    prev_round_prop_good: List[float] = list()

    last_round_action_ef: Optional[float] = None
    prev_round_action_efs: List[float] = list()

    last_round_pos_good: Optional[float] = None
    prev_round_pos_good: List[float] = list()
    last_round_neg_good: Optional[float] = None
    prev_round_neg_good: List[float] = list()

    if online_validation_data:
        logging.info('Evaluating on held-out online game data.')
        for did, dataset in online_validation_data.items():
            logging.info(f'Evaluating dataset {did}.')

            results: Dict[Metric,
                          float] = rollout_metrics.evaluate_position_predictor(
                              [model],
                              dataset.instruction_examples,
                              games,
                              rollout_config,
                              evaluation_batch_size,
                              compute_accuracy_metrics=False)[0]

            # Also compute loss
            nonzero_examples: List[StepExample] = [
                example for example in dataset.step_examples
                if not example.action_annotation.feedback.is_neutral()
            ]

            if model.directly_predicts_actions():
                feedback_results: Dict[
                    Metric,
                    float] = action_prediction_metrics.compute_expected_feedback(
                        nonzero_examples,
                        evaluation_batch_size,
                        games,
                        model,
                        allow_player_intersections=False)
            else:
                feedback_results: Dict[
                    Metric,
                    float] = position_prediction_metrics.compute_expected_feedback(
                        model,
                        nonzero_examples,
                        games,
                        evaluation_batch_size,
                        allow_player_intersections=False)

            results.update(feedback_results)

            run.log(
                {
                    f'online_validation_{did}/{metric}': value
                    for metric, value in results.items()
                },
                step=epoch_idx)

            metrics_to_improve[f'{did}_{Metric.EXPECTED_FEEDBACK}'] = results[
                Metric.EXPECTED_FEEDBACK]
            metrics_to_improve[f'{did}_{Metric.PROP_GOOD_ACTION}'] = results[
                Metric.PROP_GOOD_ACTION]
            metrics_to_improve[f'{did}_{Metric.PROP_POS_FB_SAME}'] = results[
                Metric.PROP_POS_FB_SAME]
            metrics_to_improve[f'{did}_{Metric.PROP_NEG_FB_SWITCH}'] = results[
                Metric.PROP_NEG_FB_SWITCH]

            if Metric.ACTION_EF in results:
                metrics_to_improve[f'{did}_{Metric.ACTION_EF}'] = results[
                    Metric.ACTION_EF]

            if did == ordered_online_datasets[-1]:
                last_round_expected_feedback = results[
                    Metric.EXPECTED_FEEDBACK]
                last_round_prop_good = results[Metric.PROP_GOOD_ACTION]
                last_round_pos_good = results[Metric.PROP_POS_FB_SAME]
                last_round_neg_good = results[Metric.PROP_NEG_FB_SWITCH]
                if Metric.ACTION_EF in results:
                    last_round_action_ef = results[Metric.ACTION_EF]
            else:
                # TODO: This just uses all of them: instead, should maybe have a horizon for how far back to order.
                prev_round_feedbacks.append(results[Metric.EXPECTED_FEEDBACK])
                prev_round_prop_good.append(results[Metric.PROP_GOOD_ACTION])

                prev_round_pos_good.append(results[Metric.PROP_POS_FB_SAME])
                prev_round_neg_good.append(results[Metric.PROP_NEG_FB_SWITCH])

                if Metric.ACTION_EF in results:
                    prev_round_action_efs.append(results[Metric.ACTION_EF])

    if last_round_expected_feedback is not None:
        feedbacks: List[float] = [
            last_round_expected_feedback,
            float(
                np.mean(
                    np.array([val_expected_feedback] + prev_round_feedbacks)))
        ]

        combined_feedback: float = float(np.mean(np.array(feedbacks)))
        run.log({f'combined_validation/expected_feedback': combined_feedback},
                step=epoch_idx)
        metrics_to_improve[EF_NAME] = combined_feedback
    if last_round_prop_good is not None:
        avg_prop_good: List[float] = [
            last_round_prop_good,
            float(np.mean(np.array([val_prop_good] + prev_round_prop_good)))
        ]

        combined_prop_good: float = float(np.mean(np.array(avg_prop_good)))
        metrics_to_improve[PG_ACTION] = combined_prop_good

        # TODO: This includes both static and online games.
        avg_prop_pos_good: List[float] = [
            last_round_pos_good,
            float(np.mean(np.array([val_prop_good] + prev_round_pos_good)))
        ]
        combined_pos_good: float = float(np.mean(np.array(avg_prop_pos_good)))

        metrics_to_improve[PG_ACTION_POS] = combined_pos_good

        if prev_round_neg_good:
            avg_prop_neg_good: List[float] = [
                last_round_neg_good,
                float(np.mean(np.array(prev_round_neg_good)))
            ]
        else:
            avg_prop_neg_good = last_round_neg_good
        combined_neg_good: float = float(np.mean(np.array(avg_prop_neg_good)))
        metrics_to_improve[PG_ACTION_NEG] = combined_neg_good

        run.log(
            {
                'combined_validation/prop_good_action': combined_prop_good,
                'combined_validation/prop_good_pos_feedback':
                combined_pos_good,
                'combined_validation/prop_good_neg_feedback': combined_neg_good
            },
            step=epoch_idx)
    if last_round_action_ef:
        all_action_efs: List[float] = [
            last_round_action_ef,
            float(np.mean(np.array([val_action_ef] + prev_round_action_efs)))
        ]
        combined_action_ef: float = float(np.mean(np.array(all_action_efs)))
        run.log({f'combined_validation/action_ef': combined_action_ef},
                step=epoch_idx)
        metrics_to_improve[f'{COMBINED_ACTION_EF}'] = combined_action_ef

    logging.info(f'Done evaluating model at epoch {epoch_idx}.')

    if not metrics_to_improve:
        raise ValueError('Did not get any metrics to check improvement on!')

    return _check_improvements_and_save(metrics_to_improve, patience_tracker,
                                        epoch_idx, model, save_directory,
                                        always_save, run)


@ray.remote(num_gpus=torch_util.NUM_GPUS / 2)
def loop(parameter_server: ParameterServer,
         patience_tracker: PatienceTracker,
         save_directory: str,
         project_name: str,
         experiment_name: str,
         validation_data: Dict[DatasetSplit, Dataset],
         games: GamesCollection,
         rollout_config: RolloutConfig,
         evaluation_batch_size: int,
         always_save: bool = False,
         online_val_data: Optional[Dict[str, Dataset]] = None,
         ordered_online_datasets: Optional[List[str]] = None):

    if (online_val_data is None) != (ordered_online_datasets is None):
        raise ValueError(
            'Online val data must be provided iff ordered online datasets are provided'
        )

    print(f'Started evaluation worker at {datetime.now()}')
    log.setup(save_directory, 'val_eval.log')
    logging.info(f'Starting evaluating epoch loop at {datetime.now()}')
    logging.info("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    logging.info("CUDA_VISIBLE_DEVICES: {}".format(
        os.environ["CUDA_VISIBLE_DEVICES"]))

    run = wandb_util.initialize_wandb_with_name(project_name, experiment_name)

    best_save_file: str = ''

    while not ray.get(patience_tracker.done_training.remote()):
        # Check for a new model to evaluate
        model: nn.Module = ray.get(
            parameter_server.get_model_to_evaluate.remote())

        if model is not None:
            if not isinstance(model, PositionPredictionModel):
                raise ValueError(
                    f'Model must be of type PositionPredictionModel; was {type(model)}'
                )

            save_file = _evaluation_round(model, patience_tracker,
                                          validation_data, online_val_data,
                                          games, evaluation_batch_size,
                                          save_directory, rollout_config,
                                          always_save, ordered_online_datasets,
                                          run)
            if save_file:
                best_save_file = save_file

        time.sleep(1.0)

    log.teardown_logger()
    run.finish()
    return best_save_file
