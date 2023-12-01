# python -m inference.vin.vin_test
import logging

import torch
import numpy as np

from learning.training import load_training_data
from learning.batching import environment_batch, environment_batcher
from inference.vin.vin_model import Cerealbar_VIN, _get_cerealbar_axial_2d_kernels
from util import torch_util
from data.dataset import DatasetCollection
from config.experiment import load_experiment_config_from_json
from environment.rotation import ROTATIONS

rad_to_rotidx = {}
for i, r in enumerate(ROTATIONS):
    rad_to_rotidx[np.round(r.to_radians(), 4)] = i

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.data_config import DataConfig


def load_examples() -> DatasetCollection:
    data_config: DataConfig = load_experiment_config_from_json(
        "json_configs/train.json")
    data_config.maximum_memory_age = 12
    data_config.debug_num_examples = 1
    data: DatasetCollection = load_training_data(data_config, True)
    data.construct_supervised_step_examples(
        include_future_configs_in_target=True)
    logging.info('Number of step examples per split:')
    for split, dataset in data.static_datasets.items():
        logging.info(f'\t{split}\t{len(dataset.step_examples)}')

    training_examples = list(data.static_datasets.values())[0].step_examples
    environment_batcher_obj = environment_batcher.EnvironmentBatcher()
    envs = environment_batcher_obj.batch_environments(training_examples,
                                                      data.games)
    return envs


def construct_batch(envs):
    obstacles = envs.get_all_obstacles(
    )  # only object locations (not including cards to touch)
    current_pos = envs.dynamic_info.current_positions
    current_rot = [
        rad_to_rotidx[np.round(c.item(), 4)]
        for c in envs.dynamic_info.current_rotations
    ]
    current_rot = torch.tensor(current_rot).unsqueeze(1)
    current_state = torch.cat([current_rot, current_pos], 1)

    bs, h, w = obstacles.shape
    r = 6

    goals = torch.zeros([bs, r, h, w])
    alpha_idx = current_rot[-1][0].item()
    h_idx, w_idx = current_pos[-1][0], current_pos[-1][1]
    goals[:, alpha_idx, h_idx, w_idx] = 1

    goals = goals.to(torch_util.DEVICE)
    current_state = current_state.to(torch_util.DEVICE)
    obstacles = obstacles.to(torch_util.DEVICE)
    batch = (goals, current_state, obstacles)
    return batch


if __name__ == "__main__":
    # set-up a model
    model = Cerealbar_VIN()
    transition_kernels = _get_cerealbar_axial_2d_kernels()
    print(transition_kernels)
    model._set_kernels(transition_kernels, is_axial=True)

    # get_example
    envs = load_examples()
    batch = construct_batch(envs)

    # inference
    logits, actions = model(*batch)
    print(actions)
