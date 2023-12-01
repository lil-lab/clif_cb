"""Defines kinds of dataset splits for the dataset."""
from __future__ import annotations

from enum import Enum


class DatasetSplit(str, Enum):
    TRAIN: str = 'TRAIN'
    DEV: str = 'DEV'
    TEST: str = 'TEST'
    VALIDATION: str = 'VALIDATION'

    # Original training data that it not used during training; can safely be used for evaluation. E.g., may be the
    # second half of data.
    HELD_OUT_TRAIN: str = 'HELD_OUT_TRAIN'

    def __str__(self):
        return self.value

    def shorthand(self) -> str:
        if self == DatasetSplit.TRAIN:
            return 'train'
        if self == DatasetSplit.DEV:
            return 'dev'
        if self == DatasetSplit.VALIDATION:
            return 'val'
        if self == DatasetSplit.TEST:
            return 'test'
        else:
            raise ValueError(f'No shorthand for dataset split {self}')
