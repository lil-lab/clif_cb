# Config

Configuration files for different components of the project.

- `experiment.py`: configuration for an experiment (training a model).
- `experiment_metadata.py`: common metadata for all experiments (e.g., save directory).
- `evaluation.py`: configurations for evalution, including for the main function
- `model_configs.py`: configurations for model architectures.
- `model_util_configs.py` includes configurations for modules of pytorch models (e.g., RNNs).
- `rollout.py`: configurations for doing rollouts with a model.
- `training_configs.py` includes training configurations for different kinds of experiments.
- `training_util_configs.py` includes configurations for training utilities, like patience or an optimizer, that are 
used across different kinds of training.
- `data_config.py` includes configuration for the dataset (e.g., maximum observation age).

