# Evaluation

Defines evaluation metrics, functions, and evaluation loops.

- `evaluate.py` is the endpoint for evaluation from the main file. It loads a config for evaluation and evaluates a 
model.
- `metric.py` defines evaluation metrics.
- `position_prediction_evaluation.py` defines evaluation loop for position prediction evaluation.
- `position_prediction_metrics.py` includes metric computation for position prediction evaluation.
- `rollout_metrics.py` evaluates using a position predictor in rollouts in real games.
- `rollout.py` includes data structures for keeping track of an example rollout.