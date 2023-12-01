"""Keeps track of indices of wordtypes in a consistent list, and embeds each wordtype into a vector."""
from __future__ import annotations

import torch
from torch import nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.bpe_tokenizer import BPETokenizer


class VocabularyEmbedder(nn.Module):
    def __init__(self, word_embedding_size: int, tokenizer: BPETokenizer):
        super(VocabularyEmbedder, self).__init__()

        self._tokenizer: BPETokenizer = tokenizer

        self._wordtype_embedding_size: int = word_embedding_size

        self._num_wordtypes = self._tokenizer.get_vocabulary_size()
        self._wordtype_embeddings: nn.Embedding = nn.Embedding(
            self._num_wordtypes, self._wordtype_embedding_size)
        nn.init.xavier_normal_(self._wordtype_embeddings.weight)

    def get_tokenizer(self) -> BPETokenizer:
        return self._tokenizer

    def forward(self, wordtypes: torch.Tensor) -> torch.Tensor:
        return self._wordtype_embeddings(wordtypes)
