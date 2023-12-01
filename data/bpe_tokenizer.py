"""Constructs a BPE scheme from CerealBar instructions, which is used to tokenize instructions and index wordtypes."""
from __future__ import annotations

import logging
import os
import tokenizers

from tokenizers import models, normalizers, pre_tokenizers, trainers

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.data_config import TokenizerConfig
    from typing import List, Optional

UNK_STR: str = '[UNK]'
BPE_TOK_FILENAME: str = 'bpe_tok_data.json'


class BPETokenizer:
    def __init__(self, hf_tokenizer: tokenizers.Tokenizer, directory: str):
        self._hf_tokenizer: tokenizers.Tokenizer = hf_tokenizer
        self._directory = directory

    def save(self, directory: Optional[str] = None):
        if not directory:
            directory = self._directory

        self._hf_tokenizer.save(os.path.join(directory, BPE_TOK_FILENAME))

    def log_info(self):
        logging.info('BPE tokenizer with vocab size of %s' %
                     self._hf_tokenizer.get_vocab_size())

    def get_vocabulary_size(self):
        return self._hf_tokenizer.get_vocab_size()

    def get_wordtype_index(self, token: str):
        return self._hf_tokenizer.token_to_id(token)

    def tokenize(self, text: str) -> List[str]:
        return self._hf_tokenizer.encode(text).tokens


def train_bpe_tokenizer(sentences: List[str],
                        config: TokenizerConfig) -> tokenizers.Tokenizer:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = tokenizers.Tokenizer(models.BPE(unk_token=UNK_STR))

    if not config.case_sensitive:
        tokenizer.normalizer = normalizers.Lowercase()

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Special tokens only include the UNK token.
    trainer = trainers.BpeTrainer(
        special_tokens=[UNK_STR],
        vocab_size=config.maximum_vocabulary_size,
        min_frequency=config.minimum_wordtype_occurrence)

    tokenizer.train_from_iterator(sentences, trainer)

    return tokenizer


def load_bpe_tokenizer(directory: str) -> BPETokenizer:
    filepath: str = os.path.join(directory, BPE_TOK_FILENAME)
    if not os.path.exists(filepath):
        raise ValueError(f'Tokenizer filepath does not exist: {directory}')
    return BPETokenizer(tokenizers.Tokenizer.from_file(filepath), directory)
