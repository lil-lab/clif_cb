"""
Trains a model to map instructions and observations to a distribution over positions in hex-voxel space by using 
supervision on gold positions.
"""
from __future__ import annotations

import copy
import logging
import random

from typing import TYPE_CHECKING

from data import bpe_tokenizer
from data.dataset_split import DatasetSplit
from environment.action import Action
from evaluation import evaluation_loop
from learning.abstract_trainer import Trainer
from learning.batching import step_batch
from learning.optimizers.supervised_position_prediction import SupervisedPositionPredictionOptimizer
from learning.optimizers.supervised_action_prediction import SupervisedActionPredictionOptimizer
from learning.optimizers.q_value_optimization import QValueOptimizer
from model.position_prediction import PositionPredictionModel
from util import torch_util, util

if TYPE_CHECKING:
    from config.experiment import ExperimentConfig
    from data.dataset import DatasetCollection
    from data.example import Example
    from data.step_example import StepExample
    from environment.player import Player
    from inference.predicted_action_distribution import ActionPredictions
    from inference.predicted_voxel import VoxelPredictions
    from learning.parameter_server import ParameterServer
    from learning.patience_tracker import PatienceTracker
    from torch import nn
    from typing import Dict, List, Optional, Union


def _batch_examples_sharing_target(all_training_examples: List[Example],
                                   batch_size: int) -> List[List[StepExample]]:
    shared_target_steps: Dict[str, List[StepExample]] = dict()
    total_num_steps: int = 0
    for example in all_training_examples:
        example_id: str = example.example_id
        target_idx: int = 0
        previous_target: Optional[Player] = None

        for i, step in enumerate(example.step_examples):
            if step.target_action == Action.STOP:
                # Put it in its own target ID
                target_id = f'{example_id}_STOP'
            else:
                if step.final_target != previous_target:
                    # If the target is different than the previous target, increment the target idx.
                    target_idx += 1
                target_id: str = f'{example_id}_{target_idx}'
                previous_target = step.final_target

            if step.unique_target_id is None:
                step.unique_target_id = target_id

            if target_id not in shared_target_steps:
                shared_target_steps[target_id] = list()
            shared_target_steps[target_id].append(step)
            total_num_steps += 1
    max_group_size: int = max(
        [len(exs) for exs in shared_target_steps.values()])

    logging.info(
        f'Organized {len(all_training_examples)} examples ({total_num_steps} steps) into {len(shared_target_steps)} '
        f'groups of steps with shared targets, with an average of {total_num_steps / len(shared_target_steps)} steps '
        f'per group; {len(shared_target_steps) / len(all_training_examples)} groups per example. (Max group size is '
        f'{max_group_size}.)')

    all_target_ids: List[str] = list(shared_target_steps.keys())
    random.shuffle(all_target_ids)

    chunked_target_ids: List[List[str]] = list(
        util.chunks(all_target_ids, batch_size))

    chunked_examples: List[List[StepExample]] = list()
    for id_list in chunked_target_ids:
        examples: List[StepExample] = list()
        for target_id in id_list:
            examples.extend(shared_target_steps[target_id])
        chunked_examples.append(examples)

    return chunked_examples


class SupervisedPositionTrainer(Trainer):
    def __init__(self, config: ExperimentConfig, data: DatasetCollection):
        super(SupervisedPositionTrainer, self).__init__(config, data)

        self._data.construct_supervised_step_examples(
            config.supervised_position_prediction_config.target_config)
        logging.info('Number of step examples per split:')
        for split, dataset in self._data.static_datasets.items():
            logging.info(f'\t{split}\t{len(dataset.step_examples)}')

        if self._config.experiment_metadata.debug:
            self._data.static_datasets[
                DatasetSplit.VALIDATION] = copy.deepcopy(
                    self._data.static_datasets[DatasetSplit.TRAIN])
        if config.supervised_position_prediction_config.vin_backprop_config is not None:
            # Re-annotate to be feedback examples.
            if self._config.supervised_position_prediction_config.use_ips:
                raise NotImplementedError(
                    'Need to make sure the feedback examples created here use IPS.'
                )
            self._data.static_datasets[
                DatasetSplit.TRAIN].convert_step_examples_to_feedback()

        # Train a tokenizer
        training_sentences: List[str] = [
            ex.instruction for ex in self._data.static_datasets[
                DatasetSplit.TRAIN].instruction_examples
        ]
        tokenizer: bpe_tokenizer.BPETokenizer = bpe_tokenizer.BPETokenizer(
            bpe_tokenizer.train_bpe_tokenizer(
                training_sentences, config.data_config.tokenizer_config),
            self._config.experiment_metadata.get_experiment_directory())
        tokenizer.save()
        tokenizer.log_info()

        # Create the model
        self._model: PositionPredictionModel = PositionPredictionModel(
            config.supervised_position_prediction_config.model_config,
            tokenizer,
            config.supervised_position_prediction_config.vin_backprop_config)
        self._model.to(torch_util.DEVICE)
        logging.info('Created position prediction model with parameters:')
        logging.info(self._model)

        # Create the optimizer
        if self._config.get_target_config().directly_predict_actions:
            self._optimizer: SupervisedActionPredictionOptimizer = SupervisedActionPredictionOptimizer(
                self._model,
                self._config.supervised_position_prediction_config.optimizer)
        elif self._config.supervised_position_prediction_config.vin_backprop_config is not None:
            self._optimizer: QValueOptimizer = QValueOptimizer(
                self._model,
                self._config.supervised_position_prediction_config.optimizer,
                self._config.supervised_position_prediction_config.use_ips,
                # Clipping here doesn't matter, since it's dividing by 1 anyway.
                clip_max_ips=False,
                voxel_loss_coeff=self._config.
                supervised_position_prediction_config.vin_backprop_config.
                voxel_loss_coeff,
                always_use_ips=self._config.
                supervised_position_prediction_config.use_ips)
        else:
            self._optimizer: SupervisedPositionPredictionOptimizer = SupervisedPositionPredictionOptimizer(
                self._model,
                self._config.supervised_position_prediction_config.optimizer,
                self._config.supervised_position_prediction_config.
                same_final_target_consistency_coefficient, self._config.
                supervised_position_prediction_config.entropy_coefficient,
                self._config.supervised_position_prediction_config.use_ips)

    def _train_batch(self, data: List[StepExample]):
        batched: step_batch.StepBatch = step_batch.batch_steps(
            data,
            self._data.games,
            self._model.get_tokenizer(),
            self._model.get_environment_batcher(),
            batch_targets=self._config.supervised_position_prediction_config.
            vin_backprop_config is None,
            batch_feedback=self._config.supervised_position_prediction_config.
            vin_backprop_config is not None,
            directly_predict_actions=self._config.get_target_config(
            ).directly_predict_actions,
            allow_player_intersections=True)

        batched.to_device()
        predictions: Union[VoxelPredictions,
                           ActionPredictions] = self._model(batched)

        loss_dict: Dict[str, float] = self._optimizer.compute_and_apply_loss(
            batched, predictions)[0]
        self.wandb_log(loss_dict)

        self._log_idx += 1

    def _train_epoch(self):
        """Carries out training over all training examples."""
        self._model.train()
        batch_size: int = self._config.supervised_position_prediction_config.optimizer.batch_size

        if self._config.supervised_position_prediction_config.same_final_target_consistency_coefficient > 0:
            # Need to organize batches so that contiguous examples with the same final target appear in the same batch.
            batches: List[List[StepExample]] = _batch_examples_sharing_target(
                self._data.static_datasets[
                    DatasetSplit.TRAIN].instruction_examples, batch_size)
        else:
            training_examples: List[StepExample] = self._data.static_datasets[
                DatasetSplit.TRAIN].step_examples

            random.shuffle(training_examples)

            batches: List[List[StepExample]] = list(
                util.chunks(training_examples, batch_size))

        logging.info('Starting epoch')
        for batch_idx, batch in enumerate(batches):
            logging.info(f'{batch_idx + 1} / {len(batches)}')
            self._train_batch(batch)

    def get_model(self) -> nn.Module:
        return self._model

    def launch_evaluation(self, param_server: ParameterServer,
                          patience_tracker: PatienceTracker):
        return evaluation_loop.loop.remote(
            param_server,
            patience_tracker,
            self._config.experiment_metadata.get_experiment_directory(),
            self._config.get_project_name(),
            self._config.experiment_metadata.experiment_name, {
                DatasetSplit.VALIDATION:
                self._data.static_datasets[DatasetSplit.VALIDATION]
            },
            self._data.games,
            self._config.supervised_position_prediction_config.rollout_config,
            self._config.supervised_position_prediction_config.optimizer.
            batch_size,
            always_save=True)


def train(config: ExperimentConfig, data: DatasetCollection) -> str:
    trainer: SupervisedPositionTrainer = SupervisedPositionTrainer(
        config, data)

    return trainer.train_loop()
