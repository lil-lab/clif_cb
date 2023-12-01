"""Endpoint for evaluating a model."""
from __future__ import annotations

import logging
from util import log

from config import evaluation
from data import loading
from data.dataset import DatasetCollection, GamesCollection
from evaluation import position_prediction_evaluation
from util import torch_util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.data_config import DataConfig
    from data.dataset import Dataset
    from data.dataset_split import DatasetSplit
    from data.game import Game
    from typing import Dict, List


def _load_data(data_config: DataConfig, split: DatasetSplit,
               debug: bool) -> DatasetCollection:
    logging.info('Loading data')
    data: Dict[DatasetSplit, Dataset] = loading.load_instructions(data_config,
                                                                  [split],
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


def evaluate_with_config(evaluation_config: evaluation.EvaluationConfig):
    if evaluation_config.online_dataset_name:
        data: DatasetCollection = loading.load_recorded_data(
            evaluation_config.loaded_experiment_config.data_config,
            [evaluation_config.online_dataset_name],
            'game_recordings/',
            val_only=False,
            use_ips={evaluation_config.online_dataset_name: False})
    else:
        data: DatasetCollection = _load_data(
            evaluation_config.loaded_experiment_configs[0].data_config,
            evaluation_config.dataset_split, evaluation_config.debug)

    position_prediction_evaluation.load_and_evaluate_model(
        evaluation_config, data)


def load_config_and_evaluate(config_filepath: str):
    evaluation_config: evaluation.EvaluationConfig = evaluation.load_evaluation_config_from_json(
        config_filepath)

    logging.info('Loading the following models for evaluation:')
    for model_path in evaluation_config.model_filepaths:
        print(f'\t{model_path}')
    logging.info('Using device: %s' % str(torch_util.DEVICE))

    evaluate_with_config(evaluation_config)
