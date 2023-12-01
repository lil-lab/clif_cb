"""Configurations for various model utilities."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RNNConfig:
    """Configures an LSTM RNN.
    
    Attributes:
        self.word_embedding_size: the word embedding size.
        self.hidden_size: the (final) hidden size for each timestep.
        self.num_layers: the number of layers in the RNN.
    """
    word_embedding_size: int

    hidden_size: int

    num_layers: int

    def validate(self):
        if self.word_embedding_size <= 0:
            raise ValueError('Word embedding size must be at least one.')
        if self.hidden_size <= 0:
            raise ValueError('Hidden size must be at least one.')
        if self.num_layers <= 0:
            raise ValueError('Number of layers must be at least one.')


@dataclass
class EnvironmentEmbedderConfig:
    """Configures an EnvironmentEmbedder.
    
    Attributes:
        self.embedding_size: the embedding size for each property in the environment.
    """
    embedding_size: int

    previous_target_dropout_rate: float = 0.

    use_previous_cards_in_input: bool = False

    hierarchical: bool = False

    zero_out_nonexistent_property_embeddings: bool = False

    share_card_embeddings_with_previous: bool = False

    def validate(self):
        if self.embedding_size <= 0:
            raise ValueError('Embedding size must be at least one.')

        if self.share_card_embeddings_with_previous and not self.use_previous_cards_in_input:
            raise ValueError(
                'If sharing card embeddings with previous-card channel, previous-card channel must be '
                'used.')


@dataclass
class LingunetConfig:
    # Strides for convolutions in the network (e.g., in the LingUNet).
    convolution_strides: int = 2

    # Sizes of kernels in the LingUNet
    kernel_size: int = 3

    # Padding of convolution operations in the LingUNet
    convolution_padding: int = 1

    # The number of LingUNet layers
    num_layers: int = 4

    # The number of channels after the first (lefthand) series of convolutions
    after_first_convolution_channels: int = 48

    # The number channels after the text kernel convolutions
    after_text_convolution_channels: int = 24

    # The number of convolution layers per convolution block
    num_convolutions_per_block: int = 2

    # Dropout in the LingUNet
    dropout: float = 0

    # Internal size of additional scalar predictions
    additional_head_internal_size: int = 32

    # Use instance norm?
    use_instance_norm: bool = True

    # Use Tanh surrounding instance norm?
    use_surrounding_tanh: bool = False

    crop_for_additional_heads: bool = False

    max_pool_for_additional_heads: bool = False

    def validate(self):
        if self.convolution_strides <= 0:
            raise ValueError('LingUNet must use strides of at least 1: %s' %
                             self.convolution_strides)
        if self.kernel_size <= 0:
            raise ValueError('LingUNet kernel size must be at least 1: %s' %
                             self.kernel_size)
        if self.convolution_padding < 0:
            raise ValueError(
                'LingUNet convolution padding may not be negative: %s' %
                self.convolution_padding)
        if self.num_layers < 1:
            raise ValueError('LingUNet must have at least one layer: %s' %
                             self.num_layers)
        if self.after_first_convolution_channels <= 0:
            raise ValueError(
                'LingUNet must have a positive number of channels after the first convolutions: %s'
                % self.after_first_convolution_channels)
        if self.after_text_convolution_channels <= 0:
            raise ValueError(
                'LingUNet must have a positive number of channels after the text convolutions: %s'
                % self.after_text_convolution_channels)
        if self.num_convolutions_per_block <= 0:
            raise ValueError(
                'LingUNet must have a positive number of convolutions per block: %s'
                % self.num_convolutions_per_block)

        if self.crop_for_additional_heads and self.max_pool_for_additional_heads:
            raise ValueError(
                'Cannot both crop and max pool for additional LingUNet heads.')
