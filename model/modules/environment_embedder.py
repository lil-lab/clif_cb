"""Used for embedding static and dynamic information about an environment into a tensor."""
from __future__ import annotations

import torch
from torch import nn

from environment.position import EDGE_WIDTH
from environment.rotation import ROTATIONS
from learning.batching.environment_batcher import EnvironmentBatcher
from model.modules.dynamic_embedder import DynamicEmbedder
from model.modules.static_embedder import StaticEmbedder
from util import torch_util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.model_util_configs import EnvironmentEmbedderConfig
    from learning.batching.environment_batch import DynamicEnvironmentBatch, EnvironmentBatch
    from typing import List, Optional

VISITATION_SCALE: float = 0.5

# Extra channels includes utility channels, obstacle mask, observability masks, player locations, previous visitation
NUM_EXTRA_CHANNELS: int = 19

# Extra channels when using previous targets: 6 rotations for all prev + prev and 1 for summing over rotations
PREV_TARGET_EXTRA_CHANNELS: int = 14

NUM_HIER_SETS: int = 5


def shuffle_voxels_to_egocentric(previous_visitations: torch.Tensor, rotations: torch.Tensor) -> \
        torch.Tensor:
    shifted_tensors: List[torch.Tensor] = list()

    for i, rotation in enumerate(rotations.long().tolist()):
        item_tensor: torch.tensor = previous_visitations[i]

        shifted_tensor: torch.Tensor = torch.zeros(
            (len(ROTATIONS), EDGE_WIDTH, EDGE_WIDTH), device=torch_util.DEVICE)

        for j in range(len(ROTATIONS)):
            shifted_tensor[j, :, :] = item_tensor[(j - rotation[0]) %
                                                  len(ROTATIONS), :, :]
        shifted_tensors.append(shifted_tensor.unsqueeze(0))

    return torch.cat(shifted_tensors, dim=0)


class EnvironmentEmbedder(nn.Module):
    def __init__(self, config: EnvironmentEmbedderConfig,
                 shuffle_visitation_channels_to_egocentric: bool,
                 use_previous_targets_in_input: bool):
        super(EnvironmentEmbedder, self).__init__()

        self._shuffle_visitation_channels_to_egocentric: bool = shuffle_visitation_channels_to_egocentric
        self._use_previous_targets_in_input: bool = use_previous_targets_in_input

        self._hierarchical: bool = config.hierarchical

        # Batching utils
        self._environment_batcher: EnvironmentBatcher = EnvironmentBatcher(
            config.use_previous_cards_in_input)

        # Components
        self._property_embedding_size: int = config.embedding_size
        self._static_embedder: StaticEmbedder = StaticEmbedder(
            self._property_embedding_size, self._environment_batcher,
            self._hierarchical,
            config.zero_out_nonexistent_property_embeddings)
        self._dynamic_embedder: DynamicEmbedder = DynamicEmbedder(
            self._property_embedding_size, self._environment_batcher,
            self._hierarchical,
            config.zero_out_nonexistent_property_embeddings,
            config.share_card_embeddings_with_previous)

        self._previous_target_dropout_rate: float = config.previous_target_dropout_rate

        # Utility channels, precomputed
        self._environment_channel: nn.Parameter = nn.Parameter(
            torch.ones(1, 1, EDGE_WIDTH, EDGE_WIDTH), requires_grad=False)
        self._compass_placeholder: nn.Parameter = nn.Parameter(
            torch.zeros(1, 6, EDGE_WIDTH, EDGE_WIDTH), requires_grad=False)
        self._compass_ones_placeholder: nn.Parameter = nn.Parameter(
            torch.ones(1, 1, EDGE_WIDTH, EDGE_WIDTH), requires_grad=False)

    def get_environment_batcher(self) -> EnvironmentBatcher:
        return self._environment_batcher

    def _add_utility_channels(
            self, environment_tensor: torch.Tensor,
            follower_rotations: torch.Tensor) -> torch.Tensor:
        batch_size: int = environment_tensor.size(0)

        # Add environment mask
        with_environment: torch.Tensor = torch.cat(
            (environment_tensor,
             self._environment_channel.repeat(batch_size, 1, 1, 1)),
            dim=1)

        # Add compass channel
        compass: torch.Tensor = self._compass_placeholder.repeat(
            batch_size, 1, 1, 1)

        # Gets the index of the rotation to use (rounds down to nearest integer, between 0 -- 5)
        # TODO: Could probably be more efficient
        for i, rot in enumerate(follower_rotations.long().tolist()):
            compass[i, rot] = self._compass_ones_placeholder

        expected_sum: int = EDGE_WIDTH * EDGE_WIDTH * batch_size
        actual_sum: int = torch.sum(compass)
        assert actual_sum == expected_sum, 'Expected vs. actual sum of compass: %s vs. %s' % (
            expected_sum, actual_sum)
        return torch.cat((with_environment, compass), dim=1)

    def get_entire_embedding_size(self) -> int:
        base_size: int = self._property_embedding_size
        if self._hierarchical:
            base_size *= NUM_HIER_SETS

        emb_size: int = base_size + NUM_EXTRA_CHANNELS
        if self._use_previous_targets_in_input:
            emb_size += PREV_TARGET_EXTRA_CHANNELS
        return emb_size

    def forward(self, environment_batch: EnvironmentBatch) -> torch.Tensor:
        # [1] Embed the static and dynamic properties (not including obstacles or observability) here
        # Size should be B x Hn x 25 x 25, where H is the property embedding size and n is the number of hierarchical
        #  sets (n = 1 when using a flat representation).
        embedded_static: torch.Tensor = self._static_embedder(
            environment_batch.static_info)

        dynamic_batch: DynamicEnvironmentBatch = environment_batch.dynamic_info
        embedded_dynamic: torch.Tensor = self._dynamic_embedder(dynamic_batch)

        # Embedding of environment information: B x N x 25 x 25; N is the embedding size
        if self._hierarchical:
            embedded_env: torch.Tensor = torch.cat(
                (embedded_static, embedded_dynamic), dim=1)
        else:
            embedded_env: torch.Tensor = torch.sum(torch.stack(
                (embedded_dynamic, embedded_static)),
                                                   dim=0)

        # All obstacles, including the leader's current position: B x 1 x 25 x 25
        obstacle_mask: torch.Tensor = environment_batch.get_all_obstacles(
        ).unsqueeze(1)

        # Concatenate everything together. Should be size: B x (N + 7 + 5) x 25 x 25, where 7 is from 6 rotations for
        # the previous visitations (plus one for the sum of it), and 5 includes the obstacles, both observability
        # masks, and the player locations.
        observability_in_memory: torch.Tensor = dynamic_batch.observability_in_memory.unsqueeze(
            1)

        # Scale the previous visitations so that the value more closely matches the other values in the tensor.
        previous_visitations: torch.Tensor = environment_batch.previous_visitations * VISITATION_SCALE

        all_previous_targets: Optional[torch.Tensor] = None
        previous_target: Optional[torch.Tensor] = None

        if self._use_previous_targets_in_input:
            all_previous_targets = environment_batch.all_previous_targets * VISITATION_SCALE
            previous_target = environment_batch.previous_target

            # Apply dropout.
            if self.training:
                dropout_mask: torch.Tensor = (
                    torch.randn((all_previous_targets.size()),
                                device=torch_util.DEVICE) >=
                    self._previous_target_dropout_rate).float()
                all_previous_targets = all_previous_targets * dropout_mask
                previous_target = previous_target * dropout_mask

        if self._shuffle_visitation_channels_to_egocentric:
            previous_visitations = shuffle_voxels_to_egocentric(
                previous_visitations,
                environment_batch.dynamic_info.current_rotations)

            if self._use_previous_targets_in_input:
                all_previous_targets = shuffle_voxels_to_egocentric(
                    all_previous_targets,
                    environment_batch.dynamic_info.current_rotations)
                previous_target = shuffle_voxels_to_egocentric(
                    previous_target,
                    environment_batch.dynamic_info.current_rotations)

        all_visitations: torch.Tensor = torch.sum(
            environment_batch.previous_visitations, dim=1).unsqueeze(1)

        embedded: torch.Tensor = torch.cat(
            (embedded_env, obstacle_mask,
             dynamic_batch.observability_current.unsqueeze(1),
             observability_in_memory, previous_visitations, all_visitations,
             environment_batch.dynamic_info.leader_location.unsqueeze(1),
             environment_batch.dynamic_info.follower_location.unsqueeze(1)),
            dim=1)

        if self._use_previous_targets_in_input:
            # Get targets on position only (sum out rotations)
            all_pos_prev_targets: torch.Tensor = torch.sum(
                all_previous_targets, dim=1).unsqueeze(1)
            pos_prev_target: torch.Tensor = torch.sum(previous_target,
                                                      dim=1).unsqueeze(1)

            embedded = torch.cat(
                (embedded, all_previous_targets, previous_target,
                 all_pos_prev_targets, pos_prev_target),
                dim=1)

        # Mask it by the observability history.
        environment_tensor: torch.Tensor = embedded * observability_in_memory

        # Add the utility channels.
        environment_tensor = self._add_utility_channels(
            environment_tensor,
            environment_batch.dynamic_info.current_rotations)

        return environment_tensor
