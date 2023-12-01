"""Batches an instruction sequence."""
from __future__ import annotations

import torch

from dataclasses import dataclass

from util import torch_util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.bpe_tokenizer import BPETokenizer
    from typing import List


@dataclass
class InstructionBatch:
    """Sequences of instructions.
    
    Attributes:
        self.sequences: The tokens representing the sequences, including padding.
        self.sequence_lengths: The lengths of each instruction in the batch.
    """

    # B x L (long)
    sequences: torch.Tensor

    # B (long)
    sequence_lengths: torch.Tensor

    def to_device(self):
        self.sequences = self.sequences.to(torch_util.DEVICE)
        self.sequence_lengths = self.sequence_lengths.to(torch_util.DEVICE)


def batch_instructions(instructions: List[str],
                       tokenizer: BPETokenizer) -> InstructionBatch:
    tokenized_instructions: List[List[str]] = [
        tokenizer.tokenize(instruction) for instruction in instructions
    ]

    instruction_batch_indices: List[List[int]] = [[
        tokenizer.get_wordtype_index(tok) for tok in seq
    ] for seq in tokenized_instructions]

    instruction_lengths: List[int] = [
        len(instruction) for instruction in tokenized_instructions
    ]

    instruction_index_tensor: torch.Tensor = torch.zeros(
        (len(instructions), max(instruction_lengths)),
        dtype=torch.long,
        device=torch_util.DEVICE)

    for idx, (sequence, sequence_length) in enumerate(
            zip(instruction_batch_indices, instruction_lengths)):
        instruction_index_tensor[idx, :sequence_length] = torch.tensor(
            sequence, dtype=torch.long, device=torch_util.DEVICE)
    instruction_lengths_tensor: torch.Tensor = torch.tensor(
        instruction_lengths, dtype=torch.long, device=torch_util.DEVICE)

    return InstructionBatch(instruction_index_tensor,
                            instruction_lengths_tensor)
