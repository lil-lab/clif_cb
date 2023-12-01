from __future__ import annotations

import random

from config.data_config import DataConfig, TokenizerConfig
from data.dataset_split import DatasetSplit
from data.loading import load_recorded_data
from environment.action import Action

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.dataset import DatasetCollection
    from data.example import Example
    from typing import List

NUM_EXAMPLES_PER_ROUND = 400

# Load the examples to analyze
dids: List[str] = [
    '1_29', '1_30', '1_31', '2_1', '2_2', '2_3', '2_5', '2_7', '2_9', '2_12',
    '2_16'
]

# Load the dataset
data: DatasetCollection = load_recorded_data(DataConfig(TokenizerConfig()),
                                             dids,
                                             'game_recordings/',
                                             {did: True
                                              for did in dids},
                                             val_only=False)

all_examples: List[Example] = list()

ex_to_did = dict()

for did in dids:
    this_did_examples: List[Example] = data.online_datasets[
        DatasetSplit.TRAIN][did].instruction_examples
    examples_with_stop: List[Example] = [
        example for example in this_did_examples
        if example.target_action_sequence[-1].target_action == Action.STOP
    ]
    print(f'{did}\t{len(this_did_examples)}\t{len(examples_with_stop)}')

    r = random.Random(72)
    r.shuffle(examples_with_stop)

    all_examples.extend(examples_with_stop[:NUM_EXAMPLES_PER_ROUND])

with open('annotation/sampled_examples.txt', 'w') as ofile:
    for example in all_examples:
        ofile.write(f'{example.example_id}\n')
