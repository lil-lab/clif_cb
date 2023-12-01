"""Utility configurations for training, e.g., for an optimizer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatienceSchedule:
    initial_patience: int = 10

    patience_update_factor: float = 1.0

    def validate(self):
        if self.initial_patience < 0:
            raise ValueError('Initial patience must be nonnegative.')
        if self.patience_update_factor < 1:
            raise ValueError('Patience update factor must be at least one!')


@dataclass
class OptimizerConfig:
    batch_size: int

    learning_rate: float

    l2_coefficient: float

    def validate(self):
        if self.batch_size <= 0:
            raise ValueError('Batch size should be at least one.')

        if self.l2_coefficient < 0:
            raise ValueError('L2 coefficient should not be negative: %s' %
                             self.l2_coefficient)
        if self.learning_rate <= 0.:
            raise ValueError('Learning rate must be positive: %s' %
                             self.learning_rate)


@dataclass
class VINBackpropConfig:
    """Config for how to backpropagate through the VIN.
    
    Attributes:
        self.normalize_before
            Whether to normalize the input probabilities before passing into the VIN. If False, raw logits will be 
            passed in.
        self.divide_by_probability
            Whether to scale the input probability by the policy probability, so that all inputs to the VIN have a 
            magnitude of 1.
        self.fixed_num_iters
            A fixed number of iterations to run the VIN before grabbing a Q-value. If the value here is -1, 
            then the VIN will be ran until there is a nonzero Q-value.
    """
    normalize_before: bool
    divide_by_probability: bool

    fixed_num_iters: int
    voxel_loss_coeff: float

    def validate(self):
        if self.divide_by_probability and not self.normalize_before:
            raise ValueError(
                'If dividing by probability, must normalize before going into VIN.'
            )
        if self.voxel_loss_coeff <= 0:
            raise ValueError('Voxel loss coefficient must be positive.')
