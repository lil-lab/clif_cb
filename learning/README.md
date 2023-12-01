# Learning

Utilities for training a model.

- `training.py`: entry point for training any model; loads experiment configuration and starts training.
- `util.py`: various utilities for training (e.g., checking for NaN).

Various kinds of trainers, each inheriting from `Trainer` (`abstract_trainer.py`):

- `position_supervision_training.py` trains a model to map instructions and observations to a distribution over 
positions in hex-voxel space (plus a STOP action) using supervised learning.

Various utilities for training:
- `patience_tracker.py` defines a Ray object which will keep track of model improvements across epochs and decide 
when to stop.
- `parameter_server.py` defines a Ray object which keeps track of model parameters, e.g., to send to the evaluation 
server.

Subdirectories:
- `batching/` includes utilities for batching examples into datastructures which can be put through a pytorch module 
network.
- `optimizers/` includes optimizers for different kinds of learning (e.g., position prediction)
