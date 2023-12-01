"""Embeds static properties about an environment."""
from __future__ import annotations

import torch
from torch import nn

from util.torch_util import TensorType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from learning.batching.environment_batch import StaticEnvironmentBatch
    from learning.batching.environment_batcher import EnvironmentBatcher


class StaticEmbedder(nn.Module):
    def __init__(self, embedding_size: int, batcher: EnvironmentBatcher,
                 hierarchical: bool,
                 zero_out_nonexistent_property_embeddings: bool):
        super(StaticEmbedder, self).__init__()

        self._batcher: EnvironmentBatcher = batcher
        self._embedding_size: int = embedding_size

        self._num_embeddings: int = batcher.get_num_static_embeddings()

        self._hierarchical: bool = hierarchical
        self._zero_out_nonexistent_property_embeddings: bool = zero_out_nonexistent_property_embeddings

        self._embeddings: nn.Embedding = nn.Embedding(self._num_embeddings,
                                                      self._embedding_size)

        nn.init.xavier_normal_(self._embeddings.weight)

    def forward(self, batch: StaticEnvironmentBatch) -> torch.Tensor:
        if batch.format == TensorType.NUMPY:
            raise ValueError('Input batch should be torch tensor, not numpy.')
        if not batch.batched:
            raise ValueError('Input should be batched.')

        # Get offset indices for every property type.
        prefix_length: int = 0

        prop_types: torch.Tensor = batch.prop_types + prefix_length
        prefix_length += len(self._batcher.get_prop_types())

        hut_colors: torch.Tensor = batch.hut_colors + prefix_length
        prefix_length += len(self._batcher.get_hut_colors())

        hut_rotations: torch.Tensor = batch.hut_rotations + prefix_length
        prefix_length += len(self._batcher.get_hut_rotations())

        tree_types: torch.Tensor = batch.tree_types + prefix_length
        prefix_length += len(self._batcher.get_tree_types())

        plant_types: torch.Tensor = batch.plant_types + prefix_length
        prefix_length += len(self._batcher.get_plant_types())

        windmill_rotations: torch.Tensor = batch.windmill_rotations + prefix_length
        prefix_length += len(self._batcher.get_windmill_rotations())

        tower_rotations: torch.Tensor = batch.tower_rotations + prefix_length
        prefix_length += len(self._batcher.get_tower_rotations())

        tent_rotations: torch.Tensor = batch.tent_rotations + prefix_length
        prefix_length += len(self._batcher.get_tent_rotations())

        terrains: torch.Tensor = batch.terrain + prefix_length

        # Stack these into a single tensor.
        # Size should be N x B x 25 x 25, where N is the number of properties.
        stacked_properties: torch.Tensor = torch.stack(
            (prop_types, hut_colors, hut_rotations, tree_types, plant_types,
             windmill_rotations, tower_rotations, tent_rotations, terrains))

        # Embed. Embedder embeds everything in this, so no need to reshape/permute yet.
        # Size should be N x B x 25 x 25 x H, where H is the embedding size.
        all_embeddings: torch.Tensor = self._embeddings(stacked_properties)

        # Permute so it is B x N x H x 25 x 25
        all_embeddings = all_embeddings.permute(1, 0, 4, 2, 3)

        if self._zero_out_nonexistent_property_embeddings:
            mask: torch.Tensor = batch.nonempty_property_mask.unsqueeze(
                2).repeat(1, 1, self._embedding_size, 1, 1)

            all_embeddings[:, :
                           -1, :, :, :] = all_embeddings[:, :-1, :, :, :] * mask

        if self._hierarchical:
            # Sum each type of thing individually, then concatenate.
            # Each should be of size B x H x 25 x 25.
            prop_emb: torch.Tensor = torch.sum(all_embeddings[:, :-1, :, :, :],
                                               dim=1)
            terrain_emb: torch.Tensor = all_embeddings[:, -1, :, :, :]

            return torch.cat((prop_emb, terrain_emb), dim=1)

        else:
            # Sum across the property dimension.
            return torch.sum(all_embeddings, dim=1)
