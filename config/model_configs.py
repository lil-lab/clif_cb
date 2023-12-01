"""Configurations of models."""
from __future__ import annotations

from dataclasses import dataclass

from config import model_util_configs


@dataclass
class PositionPredictionModelConfig:
    """Position prediction model, which maps an instruction and observation to a distribution over voxels.
    
    Attributes:
        self.instruction_encoder_config: the encoder for the instruction.
        self.environment_embedder_config: the environment embedder.
    """
    instruction_encoder_config: model_util_configs.RNNConfig

    environment_embedder_config: model_util_configs.EnvironmentEmbedderConfig

    grounding_map_channels: int

    lingunet_config: model_util_configs.LingunetConfig

    interpret_lingunet_chanels_as_egocentric: bool = False

    use_previous_targets_in_input: bool = False

    gating_with_previous_targets: bool = False

    copy_action: bool = False

    directly_predict_actions: bool = False

    def validate(self):
        self.instruction_encoder_config.validate()
        self.environment_embedder_config.validate()
        self.lingunet_config.validate()

        if self.grounding_map_channels <= 0:
            raise ValueError(
                'Number of grounding map channels must be at least one.')

        if self.gating_with_previous_targets and not self.use_previous_targets_in_input:
            raise ValueError(
                'If using a gating mechanism with previous targets, must provide previous targets in the input.'
            )
        if self.copy_action and self.gating_with_previous_targets:
            raise ValueError(
                'If using a copy action, cannot gate with previous targets.')

        if self.directly_predict_actions:
            if self.use_previous_targets_in_input:
                raise ValueError(
                    'Cannot use previous targets in input of model when directly predicting actions.'
                )
            if self.copy_action:
                raise ValueError(
                    'Cannot use copy action when directly predicting actions.')
            if self.gating_with_previous_targets:
                raise ValueError(
                    'Cannot gate with previous targets when directly predicting actions.'
                )

        if self.use_previous_targets_in_input and not (
                0 <= self.environment_embedder_config.
                previous_target_dropout_rate < 1):
            raise ValueError('Previous target dropout rate must be in [0, 1)')
