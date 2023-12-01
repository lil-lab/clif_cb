"""Endpoint for all training experiments."""
from __future__ import annotations

import copy
import logging

from config import experiment, evaluation
from data import loading
from data.dataset import DatasetCollection, GamesCollection
from data.dataset_split import DatasetSplit
from evaluation import evaluate
from util import torch_util, log

from learning import feedback_training
from learning import position_supervision_training

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.data_config import DataConfig
    from data.dataset import Dataset
    from data.game import Game
    from typing import Dict, List


def load_training_data(data_config: DataConfig,
                       debug: bool,
                       val_only: bool = False) -> DatasetCollection:
    splits: List[DatasetSplit] = [DatasetSplit.VALIDATION]
    if not val_only:
        splits.append(DatasetSplit.TRAIN)

    logging.info('Loading data')
    data: Dict[DatasetSplit, Dataset] = loading.load_instructions(data_config,
                                                                  splits,
                                                                  debug=debug)

    logging.info('Finished loading data. Dataset sizes:')
    all_game_ids: List[str] = list()
    for split, dataset in data.items():
        logging.info(f'\t{split}\t{dataset.get_num_instructions()}')
        all_game_ids.extend([
            example.get_game_id() for example in dataset.instruction_examples
        ])

    games: Dict[str, Game] = loading.load_games(set(all_game_ids))

    logging.info(f'Finished loading {len(games)} games.')

    return DatasetCollection(data, GamesCollection(games, dict()))


def _load_and_log_recorded_data(
        experiment_config: experiment.ExperimentConfig) -> DatasetCollection:
    ips_dict: Dict[str, bool] = dict()
    for did in experiment_config.feedback_finetuning_config.dataset_ids:
        if experiment_config.feedback_finetuning_config.only_ips_passing_examples:
            if did == experiment_config.feedback_finetuning_config.main_online_dataset:
                if experiment_config.feedback_finetuning_config.use_only_ips_passing_for_main_dataset:
                    ips_dict[did] = False
                else:
                    ips_dict[did] = True
            else:
                ips_dict[did] = False
        else:
            ips_dict[
                did] = experiment_config.feedback_finetuning_config.use_ips

    online_data: DatasetCollection = loading.load_recorded_data(
        experiment_config.data_config,
        experiment_config.feedback_finetuning_config.dataset_ids,
        experiment_config.feedback_finetuning_config.recorded_data_directory,
        ips_dict)

    train_ids: List[str] = list()
    val_ids: List[str] = list()

    for did in experiment_config.feedback_finetuning_config.dataset_ids:
        logging.info(
            f'For dataset ID {did} (is main dataset? '
            f'{did == experiment_config.feedback_finetuning_config.main_online_dataset}): '
        )
        train_dataset: Dataset = online_data.online_datasets[
            DatasetSplit.TRAIN][did]
        val_dataset: Dataset = online_data.online_datasets[
            DatasetSplit.VALIDATION][did]

        logging.info(
            f'\tTrain: {train_dataset.get_num_instructions()} instr / {len(train_dataset.step_examples)} steps'
        )
        logging.info(
            f'\tValidation: {val_dataset.get_num_instructions()} instr / {len(val_dataset.step_examples)} steps'
        )

        for step in train_dataset.step_examples:
            train_ids.append(f'{step.example_id}/{step.step_idx}')
        for step in val_dataset.step_examples:
            val_ids.append(f'{step.example_id}/{step.step_idx}')

    if len(set(train_ids)) != len(train_ids):
        raise ValueError('Duplicate train IDs in online data!')
    if len(set(val_ids)) != len(val_ids):
        raise ValueError('Duplicate validation IDs in online data!')
    if len(set(train_ids) & set(val_ids)) > 0:
        print(set(train_ids) & set(val_ids))
        raise ValueError('Overlap in training and validation online step IDs!')
    return online_data


def _load_and_log_original_data(
        experiment_config: experiment.ExperimentConfig) -> DatasetCollection:
    original_data: DatasetCollection = load_training_data(
        experiment_config.data_config,
        experiment_config.experiment_metadata.debug,
        val_only=not experiment_config.feedback_finetuning_config.
        use_original_training_data)

    if (experiment_config.experiment_metadata.debug and experiment_config.
            feedback_finetuning_config.use_original_training_data):
        original_data.static_datasets[DatasetSplit.VALIDATION] = copy.deepcopy(
            original_data.static_datasets[DatasetSplit.TRAIN])
        original_data.static_datasets[
            DatasetSplit.HELD_OUT_TRAIN] = copy.deepcopy(
                original_data.static_datasets[DatasetSplit.TRAIN])

    original_data.construct_supervised_step_examples(
        experiment_config.feedback_finetuning_config.loaded_pretraining_config.
        target_config)

    logging.info(
        f'Loaded {original_data.static_datasets[DatasetSplit.VALIDATION].get_num_instructions()} '
        f'validation examples ({len(original_data.games.games)} games; may include original training data) from '
        f'original human-human games.')

    # Original training data (if relevant)
    if experiment_config.feedback_finetuning_config.use_original_training_data:
        logging.info(
            f'Loaded {original_data.static_datasets[DatasetSplit.TRAIN].get_num_instructions()} instructions '
            f'from the original training set.')
        if DatasetSplit.HELD_OUT_TRAIN in original_data.static_datasets:
            logging.info(
                f'({original_data.static_datasets[DatasetSplit.HELD_OUT_TRAIN].get_num_instructions()} '
                f'instructions from the held-out training split.)')

        # Convert just these examples to feedback examples.
        original_data.static_datasets[
            DatasetSplit.TRAIN].convert_step_examples_to_feedback()

    return original_data


def _load_training_data_and_train(
        experiment_config: experiment.ExperimentConfig) -> str:
    if experiment_config.supervised_position_prediction_config:
        data: DatasetCollection = load_training_data(
            experiment_config.data_config,
            experiment_config.experiment_metadata.debug)

        return position_supervision_training.train(experiment_config, data)
    if experiment_config.feedback_finetuning_config:
        # Recorded training data.
        online_data: DatasetCollection = _load_and_log_recorded_data(
            experiment_config)

        # Static validation data.
        original_data: DatasetCollection = _load_and_log_original_data(
            experiment_config)

        # Combine all the games and data into one dataset.
        combined_games: Dict[str, Game] = online_data.games.games
        for gid, game in original_data.games.games.items():
            if gid in combined_games:
                raise ValueError(
                    f'Game ID {gid} is in both original and online data!')
            combined_games[gid] = game

        all_games: GamesCollection = GamesCollection(combined_games, dict())
        all_data: DatasetCollection = DatasetCollection(
            static_datasets={
                DatasetSplit.VALIDATION:
                original_data.static_datasets[DatasetSplit.VALIDATION]
            },
            games=all_games,
            online_datasets=online_data.online_datasets)

        if experiment_config.feedback_finetuning_config.use_original_training_data:
            all_data.static_datasets[
                DatasetSplit.TRAIN] = original_data.static_datasets[
                    DatasetSplit.TRAIN]

        return feedback_training.train(experiment_config, all_data)


def train(config_filepath: str):
    experiment_config: experiment.ExperimentConfig = experiment.load_experiment_config_from_json(
        config_filepath)
    experiment_config.save()

    log.setup(experiment_config.experiment_metadata.get_experiment_directory())
    logging.info(f'Parsed experiment config {config_filepath}:')
    logging.info(experiment_config)
    logging.info('Using device: %s' % str(torch_util.DEVICE))

    best_save_file: str = _load_training_data_and_train(experiment_config)

    logging.info(f'Got best save file {best_save_file}')

    if experiment_config.supervised_position_prediction_config:
        logging.info(f'Evaluating on dev set with save file {best_save_file}')

        # Evaluate with gold-forcing
        evaluate.evaluate_with_config(
            evaluation.EvaluationConfig(
                DatasetSplit.DEV,
                best_save_file,
                debug=experiment_config.experiment_metadata.debug,
                loaded_experiment_config=experiment_config,
                gold_forcing_actions=True,
                rollout_config=None,
                randomize=False))

        # Evaluate without gold-forcing
        evaluate.evaluate_with_config(
            evaluation.EvaluationConfig(
                DatasetSplit.DEV,
                best_save_file,
                debug=experiment_config.experiment_metadata.debug,
                loaded_experiment_config=experiment_config,
                gold_forcing_actions=False,
                rollout_config=experiment_config.
                supervised_position_prediction_config.rollout_config,
                randomize=False))
