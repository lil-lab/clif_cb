"""Defines a dataset for CerealBar."""
from __future__ import annotations

from dataclasses import dataclass
from tqdm import tqdm

from config.data_config import FeedbackHeuristicsConfig
from config.training_configs import SupervisedTargetConfig
from data.dataset_split import DatasetSplit
from data.example import Example
from data.step_example import StepExample
from data.game import Game

from learning.batching.environment_batch import StaticEnvironmentBatch

from typing import Dict, List, Optional


@dataclass
class Dataset:
    split: DatasetSplit

    instruction_examples: List[Example]

    step_examples: Optional[List[StepExample]] = None

    def get_num_instructions(self) -> int:
        return len(self.instruction_examples)

    def construct_supervised_step_examples(
            self, target_config: SupervisedTargetConfig, games: Dict[str,
                                                                     Game]):
        """Creates step examples (one example per step) for each loaded example."""
        if not self.step_examples:
            self.step_examples = list()
            for example in tqdm(self.instruction_examples):
                self.step_examples.extend(
                    example.construct_supervised_step_examples(
                        target_config, games[example.get_game_id()].
                        environment.get_obstacle_positions()))

    def construct_feedback_step_examples(self, source_name: str,
                                         use_ips: bool):
        if not self.step_examples:
            self.step_examples = list()
            for example in self.instruction_examples:
                self.step_examples.extend(
                    example.construct_feedback_step_examples(
                        source_name, use_ips))

    def convert_step_examples_to_feedback(self):
        if not self.step_examples:
            raise ValueError('No step examples were set!')

        self.step_examples = list()
        for example in self.instruction_examples:
            self.step_examples.extend(
                example.convert_step_examples_to_feedback())

    def reannotate_feedback_with_heuristics(self,
                                            config: FeedbackHeuristicsConfig):
        if not self.step_examples:
            raise ValueError('No step examples were set!')

        self.step_examples = list()
        for example in self.instruction_examples:
            self.step_examples.extend(
                example.reannotate_feedback_with_heuristics(config))


@dataclass
class GamesCollection:
    games: Dict[str, Game]

    cached_indices: Dict[str, StaticEnvironmentBatch]


@dataclass
class DatasetCollection:
    # Static data: original CB data with annotation of correct actions.
    static_datasets: Dict[DatasetSplit, Dataset]

    games: GamesCollection

    # Online games, for training and validation. No annotation of correct actions.
    online_datasets: Optional[Dict[DatasetSplit, Dict[str, Dataset]]] = None

    def construct_supervised_step_examples(
            self, target_config: SupervisedTargetConfig):
        """Creates step examples (one example per step) for each loaded example."""
        for name, dataset in self.static_datasets.items():
            print(f'constructing step examples for dataset {name}')
            dataset.construct_supervised_step_examples(target_config,
                                                       self.games.games)

    def construct_feedback_step_examples(self):
        """Creates step examples (one example per step) for each loaded example."""
        for split, datasets in self.online_datasets.items():
            for source_name, dataset in datasets.items():
                dataset.construct_feedback_step_examples(source_name)

    def reannotate_feedback_with_heuristics(
            self, heuristics_config: FeedbackHeuristicsConfig):
        for _, dataset in self.static_datasets.items():
            dataset.reannotate_feedback_with_heuristics(heuristics_config)
