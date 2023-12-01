# Misc. modules for defining models

## Related to environments
- `environment_embedder.py` used for embedding dynamic and static components of an environment observation into a 
tensor.
- `dynamic_embedder.py` used for embedding dynamic properties in an environment.
- `static_embedder.py` used for embedding static properties in an environment.

## Related to instructions
- `vocabulary_embedder.py` used for embedding vocabulary wordtypes.
- `sentence_encoder.py` used for encoding sentences with an RNN.

## Basic convolution modules
- `inverse_convolution_layer.py` includes inverse convolution layers.
- `convolution_layer.py` includes a basic convolution layer and residual blocks.
- `hex_conv.py` includes hex convolution layers (HexaConv)

## Other
- `lingunet.py` transforms an environemnt and instruction representation into another tensor using the LingUNet 
architecture (Blukis et al. 2018).