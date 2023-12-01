"""Encodes a sequence of wordtypes into a sequence of hidden states with an RNN."""
from __future__ import annotations

import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.modules.vocabulary_embedder import VocabularyEmbedder

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.model_util_configs import RNNConfig
    from data.bpe_tokenizer import BPETokenizer
    from learning.batching.instruction_batch import InstructionBatch


def _fast_run_rnn(seq_lens_tensor: torch.Tensor, rnn_input: torch.Tensor,
                  rnn: torch.nn.RNN) -> torch.Tensor:
    max_length: int = rnn_input.size(1)

    # Sort the lengths and get the old indices
    sorted_lengths, permuted_indices = seq_lens_tensor.sort(0, descending=True)

    # Resort the input
    sorted_input: torch.Tensor = rnn_input[permuted_indices]

    # Pack the input
    packed_input = pack_padded_sequence(sorted_input,
                                        sorted_lengths.cpu().numpy(),
                                        batch_first=True)

    # Run the RNN
    rnn.flatten_parameters()
    packed_output, cell_memories = rnn(packed_input)

    output: torch.Tensor = pad_packed_sequence(packed_output,
                                               batch_first=True,
                                               total_length=max_length)[0]

    _, unpermuted_indices = permuted_indices.sort(0)

    # Finally, sort back to original state
    hidden_states: torch.Tensor = output[unpermuted_indices]

    return hidden_states


def _initialize_rnn(rnn: nn.RNN) -> None:
    for name, param in rnn.named_parameters():
        # Set biases to zero
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            # Otherwise do Xavier initialization
            nn.init.xavier_normal_(param)


class SentenceEncoder(nn.Module):
    def __init__(self, config: RNNConfig, tokenizer: BPETokenizer):
        super(SentenceEncoder, self).__init__()

        self._config: RNNConfig = config

        # TODO: Maybe add in dropout.
        self._vocabulary_embedder: VocabularyEmbedder = VocabularyEmbedder(
            self._config.word_embedding_size, tokenizer)

        self._rnn: nn.RNN = nn.LSTM(self._config.word_embedding_size,
                                    self._config.hidden_size // 2,
                                    self._config.num_layers,
                                    batch_first=True,
                                    bidirectional=True)
        _initialize_rnn(self._rnn)

    def get_tokenizer(self) -> BPETokenizer:
        return self._vocabulary_embedder.get_tokenizer()

    def forward(self, instruction_batch: InstructionBatch) -> torch.Tensor:
        # B x M x N; M is max instruction length; N is embedding size.
        word_embeddings: torch.Tensor = self._vocabulary_embedder(
            instruction_batch.sequences)

        return _fast_run_rnn(instruction_batch.sequence_lengths,
                             word_embeddings, self._rnn)
