"""
Keeps track of model parameters between various processes. E.g., can be used to send model parameters to 
evaluation.
"""
from __future__ import annotations

import logging
import ray
import threading
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
    from torch import nn

# 30 min.
MAXIMUM_EVALUATION_TIME: int = 60 * 60


@ray.remote(num_gpus=0.5)
class ParameterServer:
    def __init__(self):
        self._current_model: nn.Module = None

        self._lock: threading.Lock = threading.Lock()

    def set_updated_model(self, new_model: nn.Module):
        with self._lock:
            if self._current_model is not None:
                raise ValueError(
                    'Trying to set updated model but the current model in parameter server is not None'
                )
            self._current_model = new_model

    def get_model_to_evaluate(self) -> Optional[nn.Module]:
        with self._lock:
            if self._current_model is not None:
                model: nn.Module = self._current_model
                self._current_model = None
                return model

    def waiting_on_eval(self) -> bool:
        return self._current_model is not None


def send_params_to_eval(param_server: ParameterServer, model: nn.Module):
    # Wait until the param server can take a new set of parameters
    wait_start_time = time.time()

    while ray.get(param_server.waiting_on_eval.remote()):
        logging.info('Waiting for current evaluation to finish...')
        time.sleep(60.0)
        if time.time() - wait_start_time > MAXIMUM_EVALUATION_TIME:
            logging.info('Waited for too long; returning.')
            return

    try:
        logging.info(
            f'Sending model parameters (model type {type(model)}) to the to eval process.'
        )
        ray.get(param_server.set_updated_model.remote(model))
    except ValueError as e:
        print('Param server crashed with error:')
        print(e)
        raise e
