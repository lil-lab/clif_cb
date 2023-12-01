"""Embeds dynamic properties about an environment."""
from __future__ import annotations

import torch
from torch import nn

from util.torch_util import TensorType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from learning.batching.environment_batch import DynamicEnvironmentBatch
    from learning.batching.environment_batcher import EnvironmentBatcher
    from typing import List


class DynamicEmbedder(nn.Module):
    def __init__(self, embedding_size: int, batcher: EnvironmentBatcher,
                 hierarchical: bool,
                 zero_out_nonexistent_property_embeddings: bool,
                 share_card_embeddings_with_previous: bool):
        super(DynamicEmbedder, self).__init__()

        self._batcher: EnvironmentBatcher = batcher
        self._embedding_size: int = embedding_size
        self._num_embeddings: int = batcher.get_num_dynamic_embeddings()

        self._hierarchical: bool = hierarchical
        self._zero_out_nonexistent_property_embeddings: bool = zero_out_nonexistent_property_embeddings
        self._share_card_embeddings_with_previous: bool = share_card_embeddings_with_previous

        self._embeddings: nn.Embedding = nn.Embedding(self._num_embeddings,
                                                      self._embedding_size)

        nn.init.xavier_normal_(self._embeddings.weight)

    def forward(self, batch: DynamicEnvironmentBatch) -> torch.Tensor:
        if batch.format == TensorType.NUMPY:
            raise ValueError('Input batch should be torch tensor, not numpy.')
        if not batch.batched:
            raise ValueError('Input should be batched.')

        # Get offset indices for every property type.
        prefix_length: int = 0

        card_counts: torch.Tensor = batch.card_counts + prefix_length
        prefix_length += len(self._batcher.get_card_counts())

        card_colors: torch.Tensor = batch.card_colors + prefix_length
        prefix_length += len(self._batcher.get_card_colors())

        card_shapes: torch.Tensor = batch.card_shapes + prefix_length
        prefix_length += len(self._batcher.get_card_shapes())

        card_selections: torch.Tensor = batch.card_selections + prefix_length
        prefix_length += len(self._batcher.get_card_selections())

        leader_rotation: torch.Tensor = batch.leader_rotation + prefix_length
        prefix_length += len(self._batcher.get_leader_rotations())

        follower_rotation: torch.Tensor = batch.follower_rotation + prefix_length
        prefix_length += len(self._batcher.get_follower_rotations())

        inputs_list: List[torch.Tensor] = [
            card_counts, card_colors, card_shapes, card_selections,
            leader_rotation, follower_rotation
        ]

        if self._batcher.use_previous_cards_in_input():
            if self._share_card_embeddings_with_previous:
                # Reset prefix length to start from where the original card embeddings start
                prefix_length = 0

            prev_visited_card_counts: torch.Tensor = batch.prev_visited_card_counts + prefix_length
            prefix_length += len(self._batcher.get_card_counts())

            prev_visited_card_colors: torch.Tensor = batch.prev_visited_card_colors + prefix_length
            prefix_length += len(self._batcher.get_card_colors())

            prev_visited_card_shapes: torch.Tensor = batch.prev_visited_card_shapes + prefix_length
            prefix_length += len(self._batcher.get_card_shapes())

            prev_visited_card_selections: torch.Tensor = batch.prev_visited_card_selections + prefix_length
            prefix_length += len(self._batcher.get_card_selections())

            inputs_list.extend([
                prev_visited_card_counts, prev_visited_card_colors,
                prev_visited_card_shapes, prev_visited_card_selections
            ])

        # Stack these into a single tensor.
        # Size should be N x B x 25 x 25, where N is the number of properties.
        stacked_properties: torch.Tensor = torch.stack(tuple(inputs_list))

        # Embed. Embedder embeds everything in this, so no need to reshape/permute yet.
        # Size should be N x B x 25 x 25 x H, where H is the embedding size.
        all_embeddings: torch.Tensor = self._embeddings(stacked_properties)

        # Permute so it is B x N x H x 25 x 25
        all_embeddings = all_embeddings.permute(1, 0, 4, 2, 3)

        if self._zero_out_nonexistent_property_embeddings:
            mask: torch.Tensor = batch.nonempty_property_mask.unsqueeze(
                2).repeat(1, 1, self._embedding_size, 1, 1)

            if not self._batcher.use_previous_cards_in_input():
                mask = mask[:, :-4, :, :]

            all_embeddings = all_embeddings * mask

        if self._hierarchical:
            # Split into hierarchical sets including:
            # - current card information
            # - player information
            # - previous card information
            current_card_info = torch.sum(all_embeddings[:, :4, :, :, :],
                                          dim=1)
            player_info = torch.sum(all_embeddings[:, 4:6, :, :, :], dim=1)

            concat_value = torch.cat((current_card_info, player_info), dim=1)

            if self._batcher.use_previous_cards_in_input():
                previous_card_info = torch.sum(all_embeddings[:, 6:, :, :, :],
                                               dim=1)
                concat_value = torch.cat((concat_value, previous_card_info),
                                         dim=1)
            return concat_value
        else:
            # Sum across the property dimension.
            return torch.sum(all_embeddings, dim=1)
