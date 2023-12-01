"""Fine-tune a pretrained model using feedback signals from human users."""
from __future__ import annotations

import copy
import os
import logging
import numpy as np
import random
import torch

from typing import TYPE_CHECKING
from tqdm import tqdm

from config.data_config import FeedbackHeuristicsConfig
from config.training_configs import FeedbackFinetuningConfig
from data import bpe_tokenizer
from data.dataset_split import DatasetSplit
from evaluation import evaluation_loop
from inference.predicted_action_distribution import PREDICTABLE_ACTIONS
from learning.abstract_trainer import Trainer
from learning.batching import step_batch
from learning.optimizers.feedback import FeedbackOptimizer, compute_expected_feedback
from learning.optimizers.q_value_optimization import QValueOptimizer
from model.position_prediction import PositionPredictionModel
from util import torch_util, util

if TYPE_CHECKING:
    from config.experiment import ExperimentConfig
    from data.dataset import DatasetCollection
    from data.step_example import StepExample
    from inference.predicted_action_distribution import ActionPredictions
    from inference.predicted_voxel import VoxelPredictions
    from learning.parameter_server import ParameterServer
    from learning.patience_tracker import PatienceTracker
    from torch import nn
    from typing import Dict, List, Optional, Set, Union


class FeedbackTrainer(Trainer):
    def __init__(self, config: ExperimentConfig, data: DatasetCollection):
        super(FeedbackTrainer, self).__init__(config, data)

        self._feedback_training_config: FeedbackFinetuningConfig = self._config.feedback_finetuning_config

        feedback_heuristics: Optional[FeedbackHeuristicsConfig] = \
            self._feedback_training_config.feedback_heuristics_config
        if feedback_heuristics:
            # Reannotate original training dataset feedback signals in order to use it for training.
            if DatasetSplit.TRAIN in self._data.static_datasets:
                self._data.static_datasets[
                    DatasetSplit.TRAIN].reannotate_feedback_with_heuristics(
                        feedback_heuristics)

            # Also reannotate all of the online data.
            for split, datasets in self._data.online_datasets.items():
                for did, dataset in datasets.items():
                    if dataset.instruction_examples:
                        dataset.reannotate_feedback_with_heuristics(
                            feedback_heuristics)
        training_sentences: List[str] = [
            ex.instruction for ex in self._data.static_datasets[
                DatasetSplit.TRAIN].instruction_examples
        ]

        self._online_example_ids = set()
        if self._feedback_training_config.positive_instructions_only:
            if self._feedback_training_config.only_ips_passing_examples:
                raise NotImplementedError

            logging.info(
                'Keeping instructions which only have positive feedback.')
            for did, dat in data.online_datasets[DatasetSplit.TRAIN].items():
                instr_with_pos = set()
                instr_with_neg = set()

                instr_with_pos |= {
                    example.example_id
                    for example in dat.step_examples
                    if example.action_annotation.feedback.polarity() > 0
                }
                instr_with_neg |= {
                    example.example_id
                    for example in dat.step_examples
                    if example.action_annotation.feedback.polarity() < 0
                }
                pos_only_instr = instr_with_pos - instr_with_neg
                logging.info(
                    f'Found {len(pos_only_instr)} instructions which only have positive feedback in dataset '
                    f'{did}.')

                self._online_example_ids |= {
                    (example.example_id, example.step_idx)
                    for example in dat.step_examples
                    if example.example_id in pos_only_instr
                }
        elif self._feedback_training_config.positive_actions_only:
            for did, dat in data.online_datasets[DatasetSplit.TRAIN].items():
                this_dataset_ids = {
                    (example.example_id, example.step_idx)
                    for example in dat.step_examples
                    if example.action_annotation.feedback.polarity() > 0
                }
                logging.info(
                    f'Dataset {did} has {len(this_dataset_ids)} total examples'
                )

                if self._feedback_training_config.only_ips_passing_examples and (
                        did !=
                        self._feedback_training_config.main_online_dataset
                        or self._feedback_training_config.
                        use_only_ips_passing_for_main_dataset):
                    with open(f'game_recordings/{did}_passing_exs.txt'
                              ) as infile:
                        passed_ids = set()
                        for line in infile.readlines():
                            if line.strip():
                                exid, step = line.strip().split(',')
                                passed_ids.add((exid, int(step)))
                        this_dataset_ids &= passed_ids
                    logging.info(
                        f'Loaded {len(this_dataset_ids)} passing IDs from dataset {did}'
                    )

                self._online_example_ids |= this_dataset_ids
        else:
            if self._feedback_training_config.only_ips_passing_examples:
                raise NotImplementedError

            for did, dat in data.online_datasets[DatasetSplit.TRAIN].items():
                self._online_example_ids |= {(example.example_id,
                                              example.step_idx)
                                             for example in dat.step_examples}

        self._log_data_stats()

        # Load the tokenizer.
        online_train_exs: Set[str] = {
            exid
            for exid, step in self._online_example_ids
        }
        for did, dat in data.online_datasets[DatasetSplit.TRAIN].items():
            for example in dat.instruction_examples:
                if example.example_id in online_train_exs:
                    training_sentences.append(example.instruction)

        tokenizer: bpe_tokenizer.BPETokenizer = bpe_tokenizer.BPETokenizer(
            bpe_tokenizer.train_bpe_tokenizer(
                training_sentences, config.data_config.tokenizer_config),
            self._config.experiment_metadata.get_experiment_directory())
        tokenizer.save(config.experiment_metadata.get_experiment_directory())
        tokenizer.log_info()

        # Create the model
        self._model: PositionPredictionModel = PositionPredictionModel(
            self._feedback_training_config.loaded_pretraining_config.
            model_config, tokenizer, config.feedback_finetuning_config.
            loaded_pretraining_config.vin_backprop_config)
        self._model.to(torch_util.DEVICE)
        if self._model.uses_copy():
            raise NotImplementedError(
                'Copy for feedback training not yet supported or tested.')

        # Load the model parameters
        if self._feedback_training_config.load_pretrained_model:
            logging.info(
                f'Loading pretrained model from {self._feedback_training_config.pretrained_experiment_directory}'
            )
            self._model.load_state_dict(
                torch.load(os.path.join(
                    self._feedback_training_config.
                    pretrained_experiment_directory,
                    self._feedback_training_config.pretrained_model_filename),
                           map_location=torch_util.DEVICE))
        logging.info('Created position prediction model with parameters:')
        logging.info(self._model)

        # Create the optimizer
        if self._feedback_training_config.loaded_pretraining_config.vin_backprop_config:
            if self._feedback_training_config.counterfactual_negative_examples:
                raise ValueError(
                    'Counterfactual negative examples not supported with VIN backprop.'
                )
            self._optimizer: QValueOptimizer = QValueOptimizer(
                self._model, self._feedback_training_config.optimizer,
                self._feedback_training_config.use_ips,
                self._feedback_training_config.clip_ips_max,
                self._feedback_training_config.loaded_pretraining_config.
                vin_backprop_config.voxel_loss_coeff, self.
                _feedback_training_config.use_ips_in_original_training_data,
                self._feedback_training_config.entropy_coefficient)
        else:
            self._optimizer: FeedbackOptimizer = FeedbackOptimizer(
                self._model, self._feedback_training_config.optimizer,
                self._feedback_training_config.clip_ips_max, self.
                _feedback_training_config.use_ips_in_original_training_data,
                self._feedback_training_config.use_ips,
                self._feedback_training_config.use_original_probability_ips,
                self._feedback_training_config.entropy_coefficient, self.
                _feedback_training_config.counterfactual_negative_examples)

    def _log_data_stats(self):
        num_original_step_examples = 0
        if DatasetSplit.TRAIN in self._data.static_datasets:
            num_original_step_examples: int = len(
                self._data.static_datasets[DatasetSplit.TRAIN].step_examples)

        num_online_step_examples: Dict[str, int] = dict()
        for did, dataset in self._data.online_datasets[
                DatasetSplit.TRAIN].items():
            num_online_step_examples[did] = len([
                step for step in dataset.step_examples
                if not step.action_annotation.feedback.is_neutral() and (
                    step.example_id, step.step_idx) in self._online_example_ids
            ])

        base_steps: int = num_original_step_examples + sum(
            num_online_step_examples.values()) - num_online_step_examples[
                self._feedback_training_config.main_online_dataset]
        if self._feedback_training_config.use_rehearsal:
            # Everything except main steps.
            if base_steps:
                logging.info(
                    f'Main dataset {self._feedback_training_config.main_online_dataset} (50% of training steps) has '
                    f'{num_online_step_examples[self._feedback_training_config.main_online_dataset]} total steps.'
                )
                logging.info('For rehearsal (other 50% of training steps):')
                logging.info(
                    f'\tOriginal data: {(100. * num_original_step_examples / base_steps):.1f}% '
                    f'/ {num_original_step_examples} steps')
                for did, amount in num_online_step_examples.items():
                    if did != self._feedback_training_config.main_online_dataset:
                        logging.info(
                            f'\t{did}: {(100. * amount / base_steps):.1f}% / {amount} steps'
                        )
        else:
            base_steps += num_online_step_examples[
                self._feedback_training_config.main_online_dataset]
            logging.info(
                'Not using rehearsal; all data is randomly sampled from original dataset size.'
            )
            logging.info(
                f'Sampling probabilities for each dataset (in each batch):')
            logging.info(
                f'\tOriginal data: {(100. * num_original_step_examples / base_steps):.1f} / '
                f'{num_original_step_examples} steps')
            for did, amount in num_online_step_examples.items():
                logging.info(
                    f'\t{did}: {(100. * amount / base_steps):.1f}% / {amount} steps'
                )

    def _train_batch(self, data: List[StepExample]):
        batched: step_batch.StepBatch = step_batch.batch_steps(
            data,
            self._data.games,
            self._model.get_tokenizer(),
            self._model.get_environment_batcher(),
            batch_targets=False,
            batch_feedback=True,
            directly_predict_actions=self._model.directly_predicts_actions())

        batched.to_device()
        predictions: VoxelPredictions = self._model(batched)

        loss_dict, item_losses = self._optimizer.compute_and_apply_loss(
            batched, predictions)

        batch_size: int = self._feedback_training_config.optimizer.batch_size
        if self._feedback_training_config.rehearsal_in_3_parts:
            batch_size //= 3
            main_loss = item_losses[:batch_size]
            supervised_loss = item_losses[batch_size:batch_size * 2]
            previous_loss = item_losses[batch_size * 2:]

            self.wandb_log({
                'loss/current_round':
                torch.mean(main_loss).item(),
                'loss/supervised':
                torch.mean(supervised_loss).item(),
                'loss/previous_rounds':
                torch.mean(previous_loss).item()
            })
        elif self._feedback_training_config.use_rehearsal:
            batch_size //= 2
            main_loss = item_losses[:batch_size]
            previous_loss = item_losses[batch_size:]

            self.wandb_log({
                'loss/current_round':
                torch.mean(main_loss).item(),
                'loss/previous_and_supervised':
                torch.mean(previous_loss).item()
            })

        assert isinstance(item_losses, torch.Tensor)
        pos_losses: List[float] = [
            item for item in item_losses[:batch_size].tolist() if item > 0
        ]
        neg_losses: List[float] = [
            item for item in item_losses[:batch_size].tolist() if item < 0
        ]
        self.wandb_log({
            'loss/positive_feedback_current_round':
            np.mean(pos_losses),
            'loss/negative_feedback_current_round':
            np.mean(neg_losses)
        })

        self.wandb_log(loss_dict)

        self.wandb_log(compute_expected_feedback(batched, predictions))

        self._log_idx += 1

    def _train_epoch(self):
        """Carries out training over all training examples."""
        self._model.train()

        # Get the main training examples.
        main_training_examples: List[StepExample] = [
            example
            for example in self._data.online_datasets[DatasetSplit.TRAIN]
            [self._feedback_training_config.main_online_dataset].step_examples
            if (example.example_id,
                example.step_idx) in self._online_example_ids
        ]
        main_training_examples = [
            step for step in main_training_examples
            if not step.action_annotation.feedback.is_neutral()
        ]

        # Get the other (original data, other online datasets) examples.
        additional_training_examples = list()
        supervised_training_examples: List[StepExample] = list()
        if DatasetSplit.TRAIN in self._data.static_datasets:
            additional_training_examples: List[StepExample] = copy.copy(
                self._data.static_datasets[DatasetSplit.TRAIN].step_examples)
            supervised_training_examples = copy.copy(
                self._data.static_datasets[DatasetSplit.TRAIN].step_examples)
        elif self._feedback_training_config.rehearsal_in_3_parts:
            raise ValueError(
                'Must provide static data train set if training batches in 3 parts.'
            )

        previous_round_examples: List[StepExample] = list()
        if self._feedback_training_config.use_new_data:
            # Only include old feedback data in additional training examples if new data is being used.
            for did, dataset in self._data.online_datasets[
                    DatasetSplit.TRAIN].items():
                if did != self._feedback_training_config.main_online_dataset:
                    prev_round_ex = [
                        step for step in dataset.step_examples
                        if (step.example_id,
                            step.step_idx) in self._online_example_ids
                    ]
                    additional_training_examples.extend(prev_round_ex)

                    previous_round_examples.extend(prev_round_ex)

        additional_training_examples = [
            step for step in additional_training_examples
            if not step.action_annotation.feedback.is_neutral()
        ]

        # Get a new batch size (if using rehearsal), or include the old examples in the main training examples.
        batch_size: int = self._feedback_training_config.optimizer.batch_size

        if self._feedback_training_config.use_rehearsal and additional_training_examples:
            if self._feedback_training_config.rehearsal_in_3_parts:
                batch_size //= 3
            else:
                batch_size //= 2
        elif self._feedback_training_config.upsample_original_data:
            batch_size = batch_size - round(
                self._feedback_training_config.original_upsampling_rate *
                batch_size)
            main_training_examples.extend(previous_round_examples)
        else:
            main_training_examples.extend(additional_training_examples)

        if self._feedback_training_config.use_new_data:
            logging.info(
                f'Training with {len(main_training_examples)} total examples.')
        else:
            logging.info(
                f'Not using new data to train; using only old (supervised) data, comprising {len(additional_training_examples)} step examples.'
            )

        # Iterate throughout batches:
        if not self._feedback_training_config.use_new_data:
            main_training_examples = additional_training_examples

        random.shuffle(main_training_examples)

        for i, batch in enumerate(
                util.chunks(main_training_examples, batch_size)):
            batch = list(batch)
            if self._feedback_training_config.use_rehearsal and additional_training_examples:
                # Add an equal number of original training examples
                if self._feedback_training_config.use_new_data:
                    if self._feedback_training_config.rehearsal_in_3_parts:
                        batch.extend(
                            random.sample(supervised_training_examples,
                                          batch_size))
                        batch.extend(
                            random.sample(previous_round_examples, batch_size))
                    else:
                        batch.extend(
                            random.sample(additional_training_examples,
                                          batch_size))
            if self._feedback_training_config.original_upsampling_rate:
                batch.extend(
                    random.sample(
                        supervised_training_examples,
                        self._feedback_training_config.optimizer.batch_size -
                        batch_size))

            self._train_batch(batch)

    def get_model(self) -> nn.Module:
        return self._model

    def launch_evaluation(self, param_server: ParameterServer,
                          patience_tracker: PatienceTracker):
        split: DatasetSplit = DatasetSplit.VALIDATION

        return evaluation_loop.loop.remote(
            param_server,
            patience_tracker,
            self._config.experiment_metadata.get_experiment_directory(),
            self._config.get_project_name(),
            self._config.experiment_metadata.experiment_name,
            {split: self._data.static_datasets[split]},
            self._data.games,
            self._feedback_training_config.loaded_pretraining_config.
            rollout_config,
            self._feedback_training_config.optimizer.batch_size,
            always_save=True,
            online_val_data=self._data.online_datasets[
                DatasetSplit.VALIDATION],
            ordered_online_datasets=self._feedback_training_config.dataset_ids)

    def verify_model_inference(self):
        """
        Verifies that the model's inference is the same as it was during the games (i.e., that it predicts the 
        same probability distributions as before).
        """
        self._model.eval()
        logging.info('Verifying model inference...')
        with torch.no_grad():
            main_training_examples: List[
                StepExample] = self._data.online_datasets[DatasetSplit.TRAIN][
                    self._feedback_training_config.
                    main_online_dataset].step_examples + self._data.online_datasets[
                        DatasetSplit.VALIDATION][
                            self._feedback_training_config.
                            main_online_dataset].step_examples

            chunks: List[List[StepExample]] = list(
                util.chunks(
                    main_training_examples,
                    self._feedback_training_config.optimizer.batch_size))
            for batch in tqdm(chunks):
                batched: step_batch.StepBatch = step_batch.batch_steps(
                    batch,
                    self._data.games,
                    self._model.get_tokenizer(),
                    self._model.get_environment_batcher(),
                    batch_targets=False,
                    batch_feedback=not self._model.directly_predicts_actions(),
                    directly_predict_actions=self._model.
                    directly_predicts_actions())

                batched.to_device()
                predictions: Union[VoxelPredictions,
                                   ActionPredictions] = self._model(batched)

                for i, example in enumerate(batched.original_examples):
                    if self._model.directly_predicts_actions():
                        print(f'{example.example_id}\t{example.step_idx}')
                        for j, action in enumerate(PREDICTABLE_ACTIONS):
                            print(
                                f'{action}'
                                f'\t{(100. * batch[i].action_annotation.probability_dist.action_probabilities[j]):.1f}%'
                                f'\t{(100. * predictions.action_probabilities[i][j].item()):.1f}%'
                            )
                        if not torch.allclose(
                                torch.tensor(
                                    batch[i].action_annotation.
                                    probability_dist.action_probabilities),
                                predictions.action_probabilities[i]):
                            print('actions incorrect')
                    else:
                        if not torch.allclose(
                                batched.feedbacks.original_distributions.
                                voxel_probabilities[i],
                                predictions.voxel_probabilities[i]):
                            print('voxels incorrect')
                            #                        print(
                            #                            f'Inference voxel probabilities not the same as recorded ones for example '
                            #                            f'{example.example_id} step {example.step_idx} '
                            #                            f'({batched.feedbacks.original_distributions.voxel_probabilities[i]} '
                            #                            f'vs. {predictions.voxel_probabilities[i]}).')
                            print(batched.feedbacks.original_distributions.
                                  voxel_probabilities[i].size())
                            for j in range(6):
                                for x in range(25):
                                    for y in range(25):
                                        v1 = batched.feedbacks.original_distributions.voxel_probabilities[
                                            i][j][x][y]
                                        v2 = predictions.voxel_probabilities[
                                            i][j][x][y]

                                        if not torch.allclose(v1, v2):
                                            print(
                                                f'{j, x, y}\t{v1.item()}\t{v2.item()}'
                                            )

                        if not torch.allclose(
                                batched.feedbacks.original_distributions.
                                stop_probabilities[i],
                                predictions.stop_probabilities[i]):
                            print(
                                f'Inference stop probabilities not the same as recorded ones for example '
                                f'{example.example_id} step {example.step_idx} '
                                f'({batched.feedbacks.original_distributions.stop_probabilities[i]} vs. '
                                f'{predictions.stop_probabilities[i]}).')
        logging.info('Done verifying model inference.')


def train(config: ExperimentConfig, data: DatasetCollection) -> str:
    trainer: FeedbackTrainer = FeedbackTrainer(config, data)

    #    trainer.verify_model_inference()
    #    exit()

    return trainer.train_loop()
