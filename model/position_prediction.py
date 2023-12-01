"""
A position prediction model: maps instructions and observations to a distribution over positions in hex-voxel 
space, plus an additional prediction for the STOP action.
"""
from __future__ import annotations

import os
import torch

from torch import nn

from config.training_util_configs import VINBackpropConfig
from environment.action import Action, MOVEMENT_ACTIONS
from environment.player import Player
from environment.position import EDGE_WIDTH
from environment.rotation import ROTATIONS
from inference.predicted_action_distribution import ActionPredictions, PREDICTABLE_ACTIONS
from inference.predicted_voxel import VoxelPredictions
from inference.vin.vin_model import Cerealbar_VIN
from model.hex_space import hex_util
from model.hex_space.pose import Pose
from model.modules.sentence_encoder import SentenceEncoder
from model.modules.environment_embedder import EnvironmentEmbedder, shuffle_voxels_to_egocentric
from model.modules.lingunet import LingUNet, LingUNetOutput
from util import torch_util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.model_configs import PositionPredictionModelConfig
    from data.bpe_tokenizer import BPETokenizer
    from learning.batching.environment_batcher import EnvironmentBatcher
    from learning.batching.instruction_batch import InstructionBatch
    from learning.batching.step_batch import StepBatch
    from typing import List, Optional, Union


def _normalize_lingunet_action_outputs(action_logits: torch.Tensor):
    batch_size, num_actions = action_logits.size()

    assert num_actions == len(PREDICTABLE_ACTIONS)

    normalized_actions: torch.Tensor = torch.softmax(action_logits, dim=1)

    return ActionPredictions(normalized_actions)


def _normalize_lingunet_map_outputs(raw_predictions: LingUNetOutput,
                                    global_view: bool,
                                    use_copy_action: bool) -> VoxelPredictions:
    # Normalize across all voxels and the STOP action.
    batch_size: int = raw_predictions.images.size(0)

    # [1] Grab the logits
    # Voxel logits
    # B x 6 x 25 x 25, unnormalized
    voxel_logits: torch.Tensor = raw_predictions.images

    # Stop logits
    if raw_predictions.additional_pixels is None:
        raise ValueError(
            'LingUNet should predict an additional pixel for STOP.')

    # B, unnormalized
    stop_logits: torch.Tensor = raw_predictions.additional_pixels[:, 0]

    # [2] Flatten and normalize
    flat_logits: torch.Tensor = voxel_logits.view(batch_size, -1)

    # B x (6 * 25 * 25 + 1)

    copy_probabilities: Optional[torch.Tensor] = None
    if use_copy_action:
        copy_logits: torch.Tensor = raw_predictions.additional_pixels[:, 1]
        all_logits: torch.Tensor = torch.cat(
            (flat_logits, stop_logits.view(
                batch_size, 1), copy_logits.view(batch_size, 1)),
            dim=1)
        distributions: torch.Tensor = torch.softmax(all_logits, dim=1)

        voxel_probabilities: torch.Tensor = distributions[:, :-2].view(
            batch_size, len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH)
        stop_probabilities: torch.Tensor = distributions[:,
                                                         -2].view(batch_size)
        copy_probabilities: torch.Tensor = distributions[:,
                                                         -1].view(batch_size)

    else:
        all_logits: torch.Tensor = torch.cat(
            (flat_logits, stop_logits.view(batch_size, 1)), dim=1)
        distributions: torch.Tensor = torch.softmax(all_logits, dim=1)

        voxel_probabilities: torch.Tensor = distributions[:, :-1].view(
            batch_size, len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH)
        stop_probabilities: torch.Tensor = distributions[:,
                                                         -1].view(batch_size)

    return VoxelPredictions(voxel_probabilities, stop_probabilities,
                            copy_probabilities, global_view)


class PositionPredictionModel(nn.Module):
    def __init__(self, config: PositionPredictionModelConfig,
                 tokenizer: BPETokenizer,
                 vin_backprop_config: Optional[VINBackpropConfig]):
        super(PositionPredictionModel, self).__init__()

        self._config: PositionPredictionModelConfig = config

        # Instructions
        self._instruction_encoder: SentenceEncoder = SentenceEncoder(
            self._config.instruction_encoder_config, tokenizer)

        # Environments
        embed_prev_targets: bool = self._config.use_previous_targets_in_input
        if self._config.gating_with_previous_targets:
            embed_prev_targets = False

        self._environment_embedder: EnvironmentEmbedder = EnvironmentEmbedder(
            self._config.environment_embedder_config,
            self._config.interpret_lingunet_chanels_as_egocentric,
            embed_prev_targets)

        # Text kernel
        self._initial_text_kernel_ll: nn.Linear = nn.Linear(
            self._config.instruction_encoder_config.hidden_size,
            (self._environment_embedder.get_entire_embedding_size() *
             self._config.grounding_map_channels))

        # LingUNet
        lingunet_input_channels: int = self._environment_embedder.get_entire_embedding_size(
        ) + self._config.grounding_map_channels
        lingunet_output_channels: int = len(ROTATIONS)

        # Transformers into and out of LingUNet.
        self._offset_to_axial_converter: hex_util.OffsetToAxialConverter = hex_util.OffsetToAxialConverter(
            EDGE_WIDTH)
        self._into_lingunet_transformer: hex_util.AxialTranslatorRotator = hex_util.AxialTranslatorRotator(
            EDGE_WIDTH, lingunet_input_channels)

        self._after_lingunet_transformer: hex_util.AxialUntranslatorUnrotator = hex_util.AxialUntranslatorUnrotator(
            EDGE_WIDTH, lingunet_output_channels)
        self._axial_to_offset_converter: hex_util.AxialToOffsetConverter = hex_util.AxialToOffsetConverter(
        )

        num_additional_pixels: int = 1
        if self._config.gating_with_previous_targets or self._config.copy_action:
            num_additional_pixels = 2
        elif self._config.directly_predict_actions:
            num_additional_pixels = len(PREDICTABLE_ACTIONS)
            lingunet_output_channels = 0

        self._lingunet: LingUNet = LingUNet(
            lingunet_config=self._config.lingunet_config,
            input_channels=lingunet_input_channels,
            text_hidden_size=self._config.instruction_encoder_config.
            hidden_size,
            output_channels=lingunet_output_channels,
            additional_scalar_outputs=num_additional_pixels)

        # Relevant only for backpropagation, where the absolute magnitudes of the Q-values in the VIN matter.
        self._vin_backprop_config: VINBackpropConfig = vin_backprop_config
        self._vin: Cerealbar_VIN = Cerealbar_VIN()

    def _encode_instructions(
            self, instruction_batch: InstructionBatch) -> torch.Tensor:
        # All hidden states for every token. We don't do any kind of attention over this, just grab a single state.
        # So grab the last one.
        encoded: torch.Tensor = self._instruction_encoder(instruction_batch)

        return encoded[torch.arange(encoded.size(0)),
                       instruction_batch.sequence_lengths - 1, :]

    def _postprocess_environment_tensor(
            self, instruction: torch.Tensor,
            environment_tensor: torch.Tensor) -> torch.Tensor:
        # [1] Get text kernels
        batch_size = instruction.size(0)
        text_kernel_shape = (
            batch_size, self._config.grounding_map_channels,
            self._environment_embedder.get_entire_embedding_size(), 1, 1)
        text_kernels = self._initial_text_kernel_ll(instruction).view(
            text_kernel_shape)

        # [2] Apply text kernels to the input
        initial_text_outputs: List[torch.Tensor] = []
        for emb_env, text_kernel in zip(environment_tensor, text_kernels):
            initial_text_outputs.append(
                nn.functional.conv2d(emb_env.unsqueeze(0), text_kernel))

        grounding_map: torch.Tensor = torch.cat(tuple(initial_text_outputs),
                                                dim=0)

        return torch.cat((environment_tensor, grounding_map), dim=1)

    def _run_vin_for_backprop(self, batch: StepBatch,
                              prediction_map: torch.Tensor) -> torch.Tensor:
        # Gets the Q-value predictions for specific actions in the batch given predicted inputs.
        batch_size: int = batch.get_batch_size()
        all_q_values: torch.Tensor = torch.zeros(batch_size,
                                                 device=torch_util.DEVICE)

        for i in range(batch_size):
            voxel: Union[Action,
                         Player] = batch.feedbacks.sampled_configurations[i]
            goal_tensor: torch.Tensor = torch.zeros(
                (len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH),
                device=torch_util.DEVICE)

            if isinstance(voxel, Player):
                goal_reward: torch.Tensor = prediction_map[i][ROTATIONS.index(
                    voxel.rotation)][voxel.position.x][voxel.position.y]
                if self._vin_backprop_config.divide_by_probability:
                    goal_reward = goal_reward / goal_reward.item()

                goal_tensor[ROTATIONS.index(voxel.rotation)][voxel.position.x][
                    voxel.position.y] = goal_reward
                q_values: torch.Tensor = self._vin(
                    goal_tensor.unsqueeze(0),
                    torch.cat((batch.environment_state.dynamic_info.
                               current_rotations[i].unsqueeze(0).long(),
                               batch.environment_state.dynamic_info.
                               current_positions[i].unsqueeze(0)),
                              dim=1),
                    batch.feedbacks.vin_obstacles[i].unsqueeze(0),
                    num_iterations=self._vin_backprop_config.fixed_num_iters)

                assert q_values.size(0) == 1

                # If goal_reward <= 0, then this may be zero.
                all_q_values[i] = q_values[0][MOVEMENT_ACTIONS.index(
                    batch.feedbacks.executed_actions[i])]

        return all_q_values

    def _predict_outputs(
        self, batch: StepBatch, environment_tensor: torch.Tensor,
        instruction_state: torch.Tensor, current_positions: torch.Tensor,
        current_rotations: torch.Tensor, previous_targets: torch.Tensor
    ) -> Union[VoxelPredictions, ActionPredictions]:
        flat_rots: torch.Tensor = current_rotations.view((-1, ))

        # [1] translate/rotate into the correct space and so that the agent is facing a consistent
        # rotation/direction.
        axial_tensor = self._offset_to_axial_converter(environment_tensor)
        transformed_state, bounds = self._into_lingunet_transformer(
            axial_tensor, Pose(current_positions, flat_rots))

        # [2] Run LingUNet
        raw_lingunet_outputs: LingUNetOutput = self._lingunet(
            transformed_state, instruction_state)

        # [3] Translate it back to offset coordinates in global view
        if raw_lingunet_outputs.images is not None:
            retransformed_state: torch.Tensor = self._axial_to_offset_converter(
                self._after_lingunet_transformer(raw_lingunet_outputs.images,
                                                 flat_rots, bounds))

            if self._config.gating_with_previous_targets:
                gating_logits: torch.Tensor = raw_lingunet_outputs.additional_pixels[:,
                                                                                     1]

                # Put through a sigmoid.
                gating_probabilities: torch.Tensor = torch.sigmoid(
                    gating_logits)

                # Gate the inputs.
                previous_targets = shuffle_voxels_to_egocentric(
                    previous_targets, current_rotations)

                batch_size: int = gating_probabilities.size(0)
                unsqueezed_gating_probs: torch.Tensor = gating_probabilities.view(
                    (batch_size, 1, 1, 1))

                retransformed_state = (
                    unsqueezed_gating_probs * retransformed_state +
                    (1 - unsqueezed_gating_probs) * previous_targets)

            # [4] Process outputs to return a PositionPrediction object
            normalized_outputs: VoxelPredictions = _normalize_lingunet_map_outputs(
                LingUNetOutput(retransformed_state,
                               raw_lingunet_outputs.additional_pixels,
                               raw_lingunet_outputs.per_layer_pixels),
                not self._config.interpret_lingunet_chanels_as_egocentric,
                self._config.copy_action)

            if self._config.interpret_lingunet_chanels_as_egocentric:
                # Interpreting the lingunet output as egocentric rotations: need to convert to a global view.
                normalized_outputs = normalized_outputs.convert_channels_egocentric_to_global(
                    current_rotations)

            assert normalized_outputs.global_interpretation

            if self._vin_backprop_config and batch.feedbacks is not None:
                # Need to run the VIN.
                if self._vin_backprop_config.normalize_before:
                    prediction_map: torch.Tensor = normalized_outputs.voxel_probabilities
                else:
                    prediction_map: torch.Tensor = raw_lingunet_outputs.images
                normalized_outputs.q_values = self._run_vin_for_backprop(
                    batch, prediction_map)
        else:
            # Just normalize the actions.
            normalized_outputs: ActionPredictions = _normalize_lingunet_action_outputs(
                raw_lingunet_outputs.additional_pixels)

        return normalized_outputs

    def save(self, directory: str, improved_metrics: List[str],
             epoch_idx: int) -> str:
        path = os.path.join(directory, f'model_{epoch_idx}.pt')
        torch.save(self.state_dict(), path)
        return path

    def get_tokenizer(self) -> BPETokenizer:
        return self._instruction_encoder.get_tokenizer()

    def get_environment_batcher(self) -> EnvironmentBatcher:
        return self._environment_embedder.get_environment_batcher()

    def get_vin(self) -> Cerealbar_VIN:
        return self._vin

    def uses_copy(self) -> bool:
        return self._config.copy_action

    def directly_predicts_actions(self) -> bool:
        return self._config.directly_predict_actions

    def uses_vin_backprop(self) -> bool:
        return self._vin_backprop_config is not None

    def forward(self, batch: StepBatch) -> VoxelPredictions:
        # [1] Encode the instructions
        # Size: B x H (H = size of instruction vector representation).
        instruction_state: torch.Tensor = self._encode_instructions(
            batch.instructions)

        # [2] Encode the observation, including static and dynamic information
        environment_tensor: torch.Tensor = self._environment_embedder(
            batch.environment_state)

        # [3] Postprocess the environment tensor, e.g., by using text kernel and adding special utility channels on top.
        environment_tensor = self._postprocess_environment_tensor(
            instruction_state, environment_tensor)

        # [4] Apply LingUNet to the environment tensor and instruction embedding to get distributions over voxel space
        return self._predict_outputs(
            batch, environment_tensor, instruction_state,
            batch.environment_state.dynamic_info.current_positions,
            batch.environment_state.dynamic_info.current_rotations,
            batch.environment_state.previous_target)
