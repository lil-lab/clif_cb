"""A trainer class: trains a model."""
from __future__ import annotations

import logging
import ray
import time

from abc import ABC, abstractmethod
from datetime import datetime

from learning import parameter_server
from learning.patience_tracker import PatienceTracker
from util import torch_util, wandb_util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.experiment import ExperimentConfig
    from data.dataset import DatasetCollection
    from torch import nn
    from typing import Dict


class Trainer(ABC):
    def __init__(self, config: ExperimentConfig, dataset: DatasetCollection):
        self._data: DatasetCollection = dataset
        self._config: ExperimentConfig = config

        self._wandb_run = None
        self._log_idx = 0

    @abstractmethod
    def _train_epoch(self):
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    @abstractmethod
    def launch_evaluation(self, param_server: parameter_server.ParameterServer,
                          patience_tracker: PatienceTracker):
        # TODO: Return type must be a ray object. What is that type exactly?
        pass

    def wandb_log(self, data: Dict):
        if not self._wandb_run:
            raise ValueError('WANDB was never initialized.')
        self._wandb_run.log(data, step=self._log_idx)

    def train_loop(self) -> str:
        """Trains a model, and returns the filepath of the best model during training."""
        ray.init(num_gpus=torch_util.NUM_GPUS)

        logging.info(f'Spawning training tracker at {datetime.now()}')
        training_tracker: PatienceTracker = PatienceTracker.remote(
            self._config.patience_schedule)
        logging.info(f'Spawning parameter server at {datetime.now()}')
        param_server: parameter_server.ParameterServer = parameter_server.ParameterServer.remote(
        )
        logging.info(f'Spawning evaluation loop at {datetime.now()}')
        best_model_pointer = self.launch_evaluation(param_server,
                                                    training_tracker)

        experiment_name: str = self._config.experiment_metadata.experiment_name

        print(f'Starting experiment: {experiment_name}')
        logging.info(f'Starting experiment {experiment_name}')

        with wandb_util.initialize_wandb_with_name(
                self._config.get_project_name(), experiment_name) as run:
            self._wandb_run = run

            num_epochs: int = 0
            # Send model to evaluation server.
            logging.info('Sending initial parameters to eval')
            parameter_server.send_params_to_eval(param_server,
                                                 self.get_model())
            while not ray.get(training_tracker.done_training.remote()):
                # Perform an epoch of training.
                num_epochs += 1
                logging.info(f'Starting epoch {num_epochs}')

                st = time.time()
                self._train_epoch()
                epoch_time = time.time() - st
                self.wandb_log({'timing/epoch': epoch_time})

                logging.info(
                    f'Finished epoch {num_epochs} in {epoch_time} seconds; sending params to eval'
                )
                parameter_server.send_params_to_eval(param_server,
                                                     self.get_model())

        logging.info('Training loop is over.')
        return ray.get(best_model_pointer)
