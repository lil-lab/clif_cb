"""Configuration for an experiment."""
from __future__ import annotations

import json
import os

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from config import data_config, experiment_metadata, training_configs, training_util_configs

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from config.model_configs import PositionPredictionModelConfig

CONFIG_FILE_NAME: str = 'config.json'


@dataclass_json
@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment.
    """
    experiment_metadata: experiment_metadata.ExperimentMetadata

    data_config: data_config.DataConfig

    patience_schedule: training_util_configs.PatienceSchedule

    supervised_position_prediction_config: Optional[
        training_configs.SupervisedPositionPredictionConfig] = None

    feedback_finetuning_config: Optional[
        training_configs.FeedbackFinetuningConfig] = None

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def get_target_config(
            self) -> Optional[training_configs.SupervisedTargetConfig]:
        if self.supervised_position_prediction_config:
            return self.supervised_position_prediction_config.target_config
        if self.feedback_finetuning_config:
            return self.feedback_finetuning_config.loaded_pretraining_config.target_config

    def get_model_config(self) -> Optional[PositionPredictionModelConfig]:
        if self.supervised_position_prediction_config:
            return self.supervised_position_prediction_config.model_config
        if self.feedback_finetuning_config:
            return self.feedback_finetuning_config.loaded_pretraining_config.model_config

    def validate(self):
        self.experiment_metadata.validate()
        self.data_config.validate()
        self.patience_schedule.validate()

        num_configs: int = 0
        if self.supervised_position_prediction_config is not None:
            self.supervised_position_prediction_config.validate()
            num_configs += 1
        if self.feedback_finetuning_config is not None:
            pretrained_exp_config: ExperimentConfig = load_experiment_config_from_json(
                os.path.join(
                    self.feedback_finetuning_config.
                    pretrained_experiment_directory, CONFIG_FILE_NAME))
            if pretrained_exp_config.supervised_position_prediction_config:
                self.feedback_finetuning_config.loaded_pretraining_config = pretrained_exp_config.supervised_position_prediction_config
            elif pretrained_exp_config.feedback_finetuning_config:
                self.feedback_finetuning_config.loaded_pretraining_config = pretrained_exp_config.feedback_finetuning_config.loaded_pretraining_config

            self.feedback_finetuning_config.validate()
            num_configs += 1

        if num_configs != 1:
            raise ValueError(
                f'Exactly one experiment config must be specified. {num_configs} were specified.'
            )

    def save(self):
        directory: str = self.experiment_metadata.get_experiment_directory()
        if not os.path.exists(directory):
            os.mkdir(directory)

        filepath: str = os.path.join(directory, CONFIG_FILE_NAME)

        if os.path.exists(filepath):
            raise FileExistsError(f'Config file already exists: {filepath}')

        with open(filepath, 'w') as ofile:
            ofile.write(str(self))

    def get_project_name(self) -> str:
        """ Gets the project name for weights and biases. Should be a unique namne for each type of training (not 
        each experiment)."""
        if self.supervised_position_prediction_config:
            return self.supervised_position_prediction_config.project_name
        elif self.feedback_finetuning_config:
            return self.feedback_finetuning_config.project_name


def load_experiment_config_from_json(filename: str):
    """Loads an ExperimentConfig from a given file."""
    with open(filename) as infile:
        str_data: str = infile.read()
    config: ExperimentConfig = ExperimentConfig.from_json(str_data)

    config.validate()

    return config
