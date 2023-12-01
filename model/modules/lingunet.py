"""A LingUNet.

Attribution:
    LingUNet: Blukis et al. 2018 (CoRL), Misra et al. 2018 (EMNLP)
    Code: Chen et al. 2019 (CVPR), https://github.com/clic-lab/street-view-navigation
          Blukis et al. 2018 (CoRL); and ongoing work by Valts Blukis.

    Official drone sim code:
        https://github.com/clic-lab/drone-sim/blob/release/learning/modules/unet/unet_5_contextual_bneck3.py
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from model.modules import convolution_layer, hex_conv, inverse_convolution_layer
from model.modules.hex_conv import HexCrop

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.model_util_configs import LingunetConfig
    from typing import List, Optional, Tuple


@dataclass
class LingUNetOutput:
    images: Optional[torch.Tensor]
    additional_pixels: Optional[torch.Tensor]
    per_layer_pixels: List[torch.Tensor]


CROP_SIZE: int = 5
ENV_MIDPOINT: int = 38


class LingUNet(nn.Module):
    def __init__(self,
                 lingunet_config: LingunetConfig,
                 input_channels: int,
                 text_hidden_size: int,
                 output_channels: int,
                 additional_scalar_outputs: int = 0,
                 scalar_layer_predictions: bool = False):
        super(LingUNet, self).__init__()
        self._config = lingunet_config
        self._input_channels: int = input_channels

        depth: int = lingunet_config.num_layers
        if text_hidden_size % depth != 0:
            raise ValueError(
                'Text hidden size should be evenly divisible by depth: ' +
                str(text_hidden_size) + ' vs. ' + str(depth))

        sliced_text_vector_size: int = text_hidden_size // depth

        self._convolution_layers = nn.ModuleList([])
        self._text_kernel_fully_connected = nn.ModuleList([])
        self._text_convolution_instance_norms = nn.ModuleList([])
        self._inverse_convolution_layers = nn.ModuleList([])
        self._text_kernel_outsizes: List[Tuple[int, int]] = list()

        self._scalar_layer_predictions: bool = scalar_layer_predictions
        self._scalar_layer_predictors = nn.ModuleList([])
        self._top_text_scalar_predictor = None

        for i in range(depth):
            # INPUT CONV LAYERS
            conv_in_channels: int = lingunet_config.after_first_convolution_channels if i > 0 else input_channels
            conv_out_channels: int = lingunet_config.after_first_convolution_channels

            conv_module = nn.ModuleList([])
            conv_module.append(
                convolution_layer.ConvolutionLayer(
                    lingunet_config.num_convolutions_per_block,
                    conv_in_channels,
                    conv_out_channels,
                    kernel_size=lingunet_config.kernel_size,
                    stride=lingunet_config.convolution_strides,
                    padding=lingunet_config.convolution_padding,
                    use_hex_conv=True))

            # Add a ReLU after each layer

            if self._config.use_surrounding_tanh:
                conv_module.append(nn.Tanh())
            else:
                conv_module.append(nn.LeakyReLU())

            # Add instance norm on the output of each layer except the last
            if lingunet_config.use_instance_norm:
                conv_module.append(nn.InstanceNorm2d(conv_out_channels))

            if self._config.use_surrounding_tanh:
                conv_module.append(nn.Tanh())

            # Have to set this as an attribute of the class, otherwise it won't be included in the parameters.
            self._convolution_layers.append(conv_module)

            # TEXT KERNEL LAYERS
            text_out_channels: int = lingunet_config.after_text_convolution_channels if i < depth - 1 \
                else conv_out_channels
            self._text_kernel_fully_connected.append(
                nn.Linear(sliced_text_vector_size,
                          conv_out_channels * text_out_channels))
            self._text_kernel_outsizes.append(
                (text_out_channels, conv_out_channels))

            if lingunet_config.use_instance_norm:
                self._text_convolution_instance_norms.append(
                    nn.InstanceNorm2d(text_out_channels))

            # This goes on the output of the text kernel at the topmost layer
            if self._scalar_layer_predictions and i == 0:
                # It's a 1x1 kernel, so doesn't need to be a hex kernel.
                self._top_text_scalar_predictor = nn.Conv2d(text_out_channels,
                                                            1,
                                                            1,
                                                            bias=False)
                nn.init.kaiming_uniform(self._top_text_scalar_predictor.weight)

            # INVERSE CONVOLUTION LAYERS
            if i > 0:  # The deconv layer for i = 0 is separate.
                # At the very bottom, this takes as input only the output from the text kernel.
                deconv_in_channels = text_out_channels if i == depth - 1 else text_out_channels + conv_out_channels

                deconv_out_channels = conv_out_channels if i > 1 else input_channels

                deconv_module = nn.ModuleList([])
                deconv_module.append(
                    inverse_convolution_layer.InverseConvolutionLayer(
                        lingunet_config.num_convolutions_per_block,
                        deconv_in_channels,
                        deconv_out_channels,
                        kernel_size=lingunet_config.kernel_size,
                        stride=lingunet_config.convolution_strides,
                        padding=lingunet_config.convolution_padding,
                        upsampling_deconv=False,
                        use_hex_conv=True))

                if self._config.use_surrounding_tanh:
                    deconv_module.append(nn.Tanh())
                else:
                    deconv_module.append(nn.LeakyReLU())

                if lingunet_config.use_instance_norm:
                    deconv_module.append(
                        nn.InstanceNorm2d(deconv_out_channels))

                if self._config.use_surrounding_tanh:
                    deconv_module.append(nn.Tanh())

                self._inverse_convolution_layers.append(deconv_module)

                if self._scalar_layer_predictions:
                    map_conv: nn.Module = nn.Conv2d(deconv_out_channels,
                                                    1,
                                                    1,
                                                    bias=False)
                    nn.init.kaiming_uniform(map_conv.weight)

                    self._scalar_layer_predictors.append(map_conv)

        # Then the final deconv layer
        # The input size will be number of input channels (from previous deconv)
        # + number of channels from top-level text conv
        # Output size is 1 (predicting a distribution)
        input_to_deconv_size = input_channels + self._text_kernel_outsizes[0][0]
        out_size = output_channels

        if lingunet_config.crop_for_additional_heads:
            assert out_size == 0
            out_size = lingunet_config.additional_head_internal_size

        self._final_deconv: Optional[hex_conv.HexConvTranspose2d] = None
        if out_size > 0:
            self._final_deconv = hex_conv.HexConvTranspose2d(
                input_to_deconv_size,
                out_size,
                kernel_size=lingunet_config.kernel_size,
                stride=lingunet_config.convolution_strides,
                padding=lingunet_config.convolution_padding)

        self._num_second_head: int = additional_scalar_outputs
        self._second_head_conv: nn.Module = None
        self._second_head_linear: nn.Module = None
        self._second_head_crop: nn.Module = None
        self._second_head_maxpool: nn.Module = None
        if additional_scalar_outputs:
            if lingunet_config.crop_for_additional_heads:
                internal_size = lingunet_config.additional_head_internal_size * CROP_SIZE * CROP_SIZE
                self._second_head_crop = HexCrop(CROP_SIZE)
            elif lingunet_config.max_pool_for_additional_heads:
                internal_size = input_to_deconv_size
                self._second_head_maxpool = nn.MaxPool2d(38)
            else:
                # The input size and internal sizes are hype
                internal_size = lingunet_config.additional_head_internal_size
                self._second_head_conv = convolution_layer.ResNetBlock(
                    lingunet_config.additional_head_internal_size,
                    lingunet_config.additional_head_internal_size, depth - 1)
            self._second_head_linear = nn.Linear(internal_size,
                                                 additional_scalar_outputs)

    def forward(self, images: torch.Tensor,
                texts: torch.Tensor) -> LingUNetOutput:
        # [1] Apply the input convolutions.
        conv_outputs: List[torch.Tensor] = list()
        for i, layer in enumerate(self._convolution_layers):
            # Apply the layer to the previous output
            conv_in = images if i == 0 else conv_outputs[-1]
            x = conv_in
            for l in layer:
                x = l(x)

            conv_outputs.append(x)

        # [2] Apply the text convolutions.
        batch_size, text_hidden_size = texts.size()
        depth: int = self._config.num_layers

        texts: torch.Tensor = nn.functional.normalize(texts, p=2, dim=1)
        sliced_size: int = text_hidden_size // depth

        text_conv_outputs: List[torch.Tensor] = list()
        top_text: torch.Tensor = None

        for i in range(depth):
            text_slices: torch.Tensor = texts[:,
                                              (i * sliced_size):((i + 1) *
                                                                 sliced_size)]

            layer_fc = self._text_kernel_fully_connected[i]

            input_size, output_size = self._text_kernel_outsizes[i]

            kernel_shape = (batch_size, input_size, output_size, 1, 1)

            # Compute the text kernel; normalize
            text_kernel: torch.Tensor = layer_fc(text_slices).view(
                kernel_shape)
            text_kernel: torch.Tensor = nn.functional.normalize(text_kernel)

            # Apply the text kernel. Have to do this by iterating over the batch.
            conv_output = conv_outputs[i]

            batch_size, out_channels, in_channels, _, _ = text_kernel.size()

            # Kernel has size B x O (output channels) x I (input channels)
            # Image has size B x I x H x W (H = W)

            # Einsum (faster than looping):
            #   boi,bihw->bohw
            text_outputs = torch.einsum(
                'boi,bihw->bohw',
                text_kernel.view(batch_size, out_channels, in_channels),
                conv_output)

            # Apply the instance norm
            if i < depth - 1:
                if self._config.use_instance_norm:
                    text_outputs = self._text_convolution_instance_norms[i](
                        text_outputs)

            if i == 0:
                top_text = text_outputs

            if self._config.use_surrounding_tanh:
                text_outputs = nn.Tanh()(text_outputs)

            text_outputs = nn.Dropout(self._config.dropout)(text_outputs)

            text_conv_outputs.append(text_outputs)

        # [3] Apply the deconvolutions from the bottom up.
        last_text_out: torch.Tensor = text_conv_outputs[-1]

        text_out_single_pred = None
        if self._scalar_layer_predictions:
            text_out_single_pred = \
                nn.AvgPool2d(top_text.size(2))(nn.LeakyReLU()(self._top_text_scalar_predictor(top_text))).view(-1, 1)

        deconv_outputs: List[torch.Tensor] = list()
        single_layer_preds: List[torch.Tensor] = list()
        for i in range(depth - 1, 0, -1):
            if i == depth - 1:
                deconv_input = last_text_out
            else:
                last_text_out = text_conv_outputs[i]
                # Concatenate the previous deconv output with this layer's text conv output
                deconv_input = torch.cat((last_text_out, deconv_outputs[-1]),
                                         1)

            x = deconv_input
            for l in self._inverse_convolution_layers[i - 1]:
                if isinstance(
                        l, inverse_convolution_layer.InverseConvolutionLayer):
                    x = l(x, output_size=text_conv_outputs[i - 1].size())

                else:
                    x = l(x)

            deconv_outputs.append(x)
            if self._scalar_layer_predictions:
                unpooled_pred: torch.Tensor = nn.LeakyReLU()(
                    self._scalar_layer_predictors[i - 1](x))

                single_layer_preds.append(
                    nn.AvgPool2d(unpooled_pred.size(2))(unpooled_pred).view(
                        -1, 1))
        if self._scalar_layer_predictions:
            single_layer_preds: torch.Tensor = torch.cat(
                tuple(single_layer_preds + [text_out_single_pred]), dim=1)

        # Apply the final deconvolution operation
        # The size of this is B x 60 x 36 x 36 (with current settings)
        final_input = torch.cat((text_conv_outputs[0], deconv_outputs[-1]), 1)

        out: Optional[torch.Tensor] = None
        if self._final_deconv is not None:
            out = self._final_deconv(final_input, output_size=images.size())

        extra_preds: torch.Tensor = None
        if self._num_second_head:
            # Applied on top of the final input, limited number of channels to the extra head internal size
            if self._second_head_crop is not None:
                output = self._second_head_crop(
                    out,
                    torch.tensor([[ENV_MIDPOINT, ENV_MIDPOINT]]).repeat(
                        (batch_size, 1)),
                    mask=True)[0].contiguous().view(batch_size, -1)
                out = None
            elif self._config.max_pool_for_additional_heads:
                output = self._second_head_maxpool(final_input).view(
                    batch_size, -1)
            else:
                input_to_last_layer = final_input[:, :self._config.
                                                  additional_head_internal_size, :, :]

                output = self._second_head_conv(input_to_last_layer).view(
                    final_input.size(0), -1)

            extra_preds = self._second_head_linear(output)

        return LingUNetOutput(out, extra_preds, single_layer_preds)
