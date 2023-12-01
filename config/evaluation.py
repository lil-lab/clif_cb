"""Configurations for evaluating a model (from main.py)."""
from __future__ import annotations

import os

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from config import experiment
from config.rollout import RolloutConfig
from data.dataset_split import DatasetSplit

from typing import List, Optional


@dataclass
class UnityStandaloneConfig:
    ip_address: str = 'localhost'

    port: int = 3706

    def validate(self):
        if not self.ip_address:
            raise ValueError('IP address must be provided.')
        if self.port <= 0:
            raise ValueError('Port must be a positive integer.')


@dataclass_json
@dataclass
class EvaluationConfig:
    """
    Configuration for a single experiment.
    
    Attributes:
        self.dataset_split
            The split of data to evaluate on.
        self.model_filepath
            The filepath pointing to the model that should be saved.
        self.standalone_config
            Optional configuration for a Unity standalone.
        self.debug
            If True, only one evaluation example will be loaded.
        self.loaded_experiment_config
            The experiment config, loaded from disk given the model path and directory. This should not be set 
            explicitly, but loaded.
    """
    dataset_split: DatasetSplit

    model_filepaths: List[str]

    online_dataset_name: Optional[str] = None

    standalone_config: Optional[UnityStandaloneConfig] = None

    debug: bool = False

    loaded_experiment_configs: Optional[List[
        experiment.ExperimentConfig]] = None

    gold_forcing_actions: bool = True

    cascaded_evaluation: bool = False

    rollout_config: Optional[RolloutConfig] = None

    randomize: bool = True

    sampling: bool = False

    argmax: bool = False

    def get_model_filename(self, index: int):
        return self.model_filepaths[index].split('/')[-1]

    def get_model_directory(self, index: int):
        return '/'.join(self.model_filepaths[index].split('/')[:-1])

    def load_experiment_config(self, index: int):
        if not self.loaded_experiment_configs:
            self.loaded_experiment_configs = [
                None for _ in range(len(self.model_filepaths))
            ]

        if self.loaded_experiment_configs[index]:
            raise ValueError('Experiment config is already loaded.')

        self.loaded_experiment_configs[
            index] = experiment.load_experiment_config_from_json(
                os.path.join(self.get_model_directory(index),
                             experiment.CONFIG_FILE_NAME))

    def validate(self):
        if not self.model_filepaths:
            raise ValueError('Model filepath must be set.')
        for i in range(len(self.model_filepaths)):
            if not self.get_model_filename(i).endswith('.pt'):
                raise ValueError('Must provide a .pt file to evaluate.')
        if not self.loaded_experiment_configs:
            raise ValueError(
                'Experiment config must be loaded before validating.')

        for config in self.loaded_experiment_configs:
            config.validate()

        if not self.gold_forcing_actions and not self.rollout_config:
            raise ValueError(
                'Must provide a rollout config when not gold-forcing actions.')

        if self.cascaded_evaluation and self.gold_forcing_actions:
            raise ValueError(
                'Cannot have both cascaded evaluation and gold-forcing set.')

        if self.standalone_config:
            self.standalone_config.validate()

        if self.sampling:
            if self.gold_forcing_actions:
                raise ValueError(
                    'Cannot perform sampling when gold-forcing actions.')
            if self.standalone_config:
                raise NotImplementedError(
                    'Sampling with visualization is not supported.')

        if not self.sampling and not self.argmax:
            raise ValueError('At least one of argmax/sampling must be set.')

        if len(self.model_filepaths
               ) > 1 != self.rollout_config.ensemble_inference:
            raise ValueError(
                'If more than one model filepath is provided, ensemble inference must be used.'
            )

        if self.rollout_config.ensemble_inference:
            if self.standalone_config:
                raise NotImplementedError(
                    'Ensemble inference with standalone is not yet supported.')
            if self.gold_forcing_actions:
                raise NotImplementedError(
                    'Gold-forcing not supported with ensemble inference.')


def load_evaluation_config_from_json(filename: str):
    """Loads an ExperimentConfig from a given file."""
    with open(filename) as infile:
        str_data: str = infile.read()

    config: EvaluationConfig = EvaluationConfig.from_json(str_data)

    for i, model_path in enumerate(config.model_filepaths):
        config.load_experiment_config(i)

    config.validate()

    return config
