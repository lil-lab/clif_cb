"""Configuration for data."""
from __future__ import annotations

import os

from dataclasses import dataclass

from typing import Optional


@dataclass
class TokenizerConfig:
    case_sensitive: bool = False

    maximum_vocabulary_size: int = 4096

    minimum_wordtype_occurrence: int = 2

    def validate(self):
        if self.maximum_vocabulary_size <= 0:
            raise ValueError('Maximum vocabulary size must be at least one.')


@dataclass
class DataConfig:
    tokenizer_config: TokenizerConfig

    maximum_memory_age: int = 12

    debug_num_examples: int = 1

    prop_training_data: float = 1.

    additional_num_steps: int = 0

    game_id_filename: Optional[str] = None

    full_observability: bool = False

    def validate(self):
        self.tokenizer_config.validate()

        if self.maximum_memory_age < 0 and not self.full_observability:
            raise ValueError('Maximum memory age must be nonnegative.')
        if self.debug_num_examples <= 0:
            raise ValueError('Number of debug examples must be at least one.')
        if not 0 <= self.prop_training_data <= 1.:
            raise ValueError(
                'Proportion of training data used must be in [0, 1].')
        if self.additional_num_steps < 0:
            raise ValueError(
                'Additional number of training steps must be at least 0.')
        if not self.prop_training_data and not self.additional_num_steps:
            raise ValueError(
                '0% of training data used and no additional steps are used.')
        if self.game_id_filename is not None and not os.path.exists(
                self.game_id_filename):
            raise ValueError(
                f'Tried to specify a game ID filename, but it did not exist: {self.game_id_filename}'
            )
        if self.game_id_filename is not None and self.prop_training_data < 1:
            raise ValueError(
                'Tried setting proportion of training data < 1 and also setting game ID filename; '
                'conflicting settings (only one should be specified at most).')


@dataclass
class FeedbackHeuristicsConfig:
    """Different options for feedback heuristics.
    
    Attributes:
        self.fill_in_the_blank
            If True, neutral feedback will be assigned non-neutral signals according to its neighbors. 
        self.same_targets
            If True, neutral feedback will be assigned any non-neutral signals that consecutive actions with the same 
            target received.
    
    """
    fill_in_the_blank: bool = False

    same_targets: bool = False

    coach: bool = False

    new_fitb: bool = False

    coach_decay_rate: float = 0.

    coach_horizon: int = 0

    def validate(self):
        num_settings: int = 0
        if self.fill_in_the_blank:
            num_settings += 1
        if self.same_targets:
            num_settings += 1
        if self.new_fitb:
            num_settings += 1
        if self.coach:
            if not (0 <= self.coach_decay_rate <= 1):
                raise ValueError('Coach decay rate must be in [0, 1].')

            num_settings += 1

        if num_settings != 1:
            raise ValueError('Exactly one setting must be set.')
