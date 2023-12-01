"""Configuration for training experiments."""
from __future__ import annotations

import os

from dataclasses import dataclass

from config.data_config import FeedbackHeuristicsConfig
from config.training_util_configs import OptimizerConfig, VINBackpropConfig
from config.model_configs import PositionPredictionModelConfig
from config.rollout import RolloutConfig

from typing import List, Optional


@dataclass
class SupervisedTargetConfig:
    """
    Configuration for specifying targets for step examples.
    
    Attributes:
        self.include_future_configs_in_target
            Whether to include future target positions in the set of targets for training. If False, 
            only the subsequent target will be considered the target.
        self.restrict_targets_to_shortest_path
            Whether to restrict future targets only to ones which lie upon the shortest path to that target. I.e., 
            for an example at step j with N future targets, this will only include targets <p_i, ... p_I> such that for 
            all j < i <= I, (i - j) = len(shortest_path(p_j, p_i)) where p is an agent configuration.
        self.use_only_final_target
            If True, only the final target will be set as the plausible next configurations.
    """
    include_future_configs_in_target: bool
    restrict_targets_to_shortest_path: bool
    use_only_final_target: bool

    allow_copy: bool
    directly_predict_actions: bool = False

    def validate(self):
        if not self.directly_predict_actions:
            if self.use_only_final_target and not self.include_future_configs_in_target:
                raise ValueError(
                    'Cannot use a final target when not including future configs in target.'
                )
        if self.directly_predict_actions and self.allow_copy:
            raise ValueError(
                'Copy is not supported when directly predicting actions.')


@dataclass
class SupervisedPositionPredictionConfig:
    """ 
    A training experiment for mapping instructions and observations to a distribution over positions in the 
    environment (including a STOP action) by using supervision on the gold positions.
    
    Attributes:
        self.optimizer
            The optimizer configuration (including batch size, learning rate, etc.)
        self.model_config
            The configuration for the model architecture
    """
    optimizer: OptimizerConfig

    model_config: PositionPredictionModelConfig

    rollout_config: RolloutConfig

    target_config: SupervisedTargetConfig

    same_final_target_consistency_coefficient: float = 0.

    entropy_coefficient: float = 0.

    use_ips: bool = False

    clip_max_ips: bool = False

    vin_backprop_config: Optional[VINBackpropConfig] = None

    project_name: str = 'CB_POS_SUP'

    def validate(self):
        self.optimizer.validate()
        self.model_config.validate()
        self.rollout_config.validate()
        self.target_config.validate()

        if not self.target_config.include_future_configs_in_target and not self.rollout_config.restrict_to_neighbors:
            raise ValueError(
                'Training without including future targets, but evaluating with a VIN.'
            )
        if self.rollout_config.restrict_to_neighbors and self.target_config.use_only_final_target:
            raise ValueError(
                'Cannot evaluate with neighbors only when restricting targets to final future '
                'configurations.')
        if self.same_final_target_consistency_coefficient < 0:
            raise ValueError(
                'Consistency coefficient for steps with the same final target should not be negative.'
            )
        if self.same_final_target_consistency_coefficient > 0 and self.target_config.use_only_final_target:
            raise ValueError(
                'If only training for final target, consistency loss will not be useful.'
            )
        if (self.target_config.include_future_configs_in_target
                and self.same_final_target_consistency_coefficient == 0
                and not self.target_config.use_only_final_target):
            raise ValueError(
                'If including future configs in target, you may want to train using consistency loss, '
                'otherwise the agent is likely to learn to predict neighbors only.'
            )
        if self.model_config.use_previous_targets_in_input:
            # Need to make sure that the targets only include a single voxel.
            if self.target_config.include_future_configs_in_target and not self.target_config.use_only_final_target:
                raise ValueError(
                    'If including previous targets in input to the model, must make sure there is only '
                    'one gold target. Currently, all future targets are possible gold targets.'
                )
        if self.model_config.copy_action != self.target_config.allow_copy:
            raise ValueError(
                'Model should predict copy action iff targets are annotated with copy actions.'
            )
        if self.model_config.directly_predict_actions != self.target_config.directly_predict_actions:
            raise ValueError(
                'Model should directly predict actions iff targets are annotated with direct action '
                'prediction.')
        if self.model_config.directly_predict_actions and self.same_final_target_consistency_coefficient > 0:
            raise ValueError(
                'Cannot have same-target consistency loss when directly predicting actions.'
            )

        if self.vin_backprop_config is not None:
            self.vin_backprop_config.validate()
            if self.model_config.directly_predict_actions:
                raise ValueError(
                    'Cannot directly predict actions when backpropagating through the VIN.'
                )
            if not self.target_config.use_only_final_target:
                raise ValueError(
                    'Only a single target can be used when backpropagating through the VIN.'
                )
            if self.entropy_coefficient > 0:
                raise NotImplementedError(
                    'Entropy coefficient not yet supported when backpropagating through the VIN.'
                )
            if self.same_final_target_consistency_coefficient > 0:
                raise ValueError(
                    'Same target consisteny coefficient not supported when backpropagating through the '
                    'VIN.')


@dataclass
class FeedbackFinetuningConfig:
    """
    Experiment for training from human feedback.
    """
    optimizer: OptimizerConfig

    pretrained_experiment_directory: str
    pretrained_model_filename: str

    dataset_ids: List[str]

    main_online_dataset: str

    # Whether to reannotate feedback using heuristics.
    feedback_heuristics_config: Optional[FeedbackHeuristicsConfig] = None

    # If True, IPS is clipped such that its maximum value is 1.
    use_ips: bool = True
    clip_ips_max: bool = True
    use_original_probability_ips: bool = True

    load_pretrained_model: bool = True

    # Whether to load original training data; and if so, whether to sample it in equal amounts as the new feedback data.
    use_original_training_data: bool = True
    use_rehearsal: bool = True
    use_ips_in_original_training_data: bool = False
    rehearsal_in_3_parts: bool = False
    upsample_original_data: bool = False
    original_upsampling_rate: float = 0.

    positive_actions_only: bool = False
    positive_instructions_only: bool = False

    # Debugging only: test whether we see similar improvements *without* using new data.
    use_new_data: bool = True

    project_name: str = 'CB_POS_FINETUNE'

    loaded_pretraining_config: Optional[
        SupervisedPositionPredictionConfig] = None

    recorded_data_directory: str = 'game_recordings/'

    entropy_coefficient: float = 0.

    counterfactual_negative_examples: bool = False

    only_ips_passing_examples: str = False

    use_only_ips_passing_for_main_dataset: str = False

    def validate(self):
        self.optimizer.validate()

        if not self.loaded_pretraining_config:
            raise ValueError(
                'Pretraining config must be loaded before validating.')
        self.loaded_pretraining_config.validate()

        if not self.pretrained_experiment_directory:
            raise ValueError('Model filepath must be set.')
        if not self.pretrained_model_filename.endswith('.pt'):
            raise ValueError('Must provide a .pt file to evaluate.')

        if not self.dataset_ids:
            raise ValueError('Must provide dataset IDs for training.')

        if self.main_online_dataset not in self.dataset_ids:
            raise ValueError(
                f'Did not find specified main online dataset in provided IDs. '
                f'Specified dataset {self.main_online_dataset}; provided IDs {self.dataset_ids}'
            )

        if not self.use_original_training_data and not self.use_new_data:
            raise ValueError(
                'Must use at least one of original and new training data!')

        if self.feedback_heuristics_config:
            self.feedback_heuristics_config.validate()

        if not self.use_ips:
            if self.use_ips_in_original_training_data or self.clip_ips_max:
                raise ValueError(
                    'IPS is not used; cannot use in original training data or clip its value.'
                )

        if self.entropy_coefficient < 0:
            raise ValueError('Entropy coeffient must be nonnegative.')

        if self.positive_actions_only and self.positive_instructions_only:
            raise ValueError(
                'At most one of positive actions or positive instructions only must be set.'
            )

        if self.counterfactual_negative_examples and (
                self.positive_actions_only or self.positive_instructions_only):
            raise ValueError(
                'Must use negative examples in order to do counterfactual learning.'
            )

        if self.upsample_original_data and not 0 <= self.original_upsampling_rate < 1.:
            raise ValueError(
                f'Upsampling rate must be in [0, 1); was {self.original_upsampling_rate}'
            )
