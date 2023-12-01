# Model

Defines parameterized pytorch models (and modules) for following instructions in CerealBar.

- `position_prediction.py` contains a `PositionPredictionModel`: maps instructions and observations to a distribution
 over voxels and the STOP action.
 
Subdirectories:
- `hex_space/` includes utilities for dealing with tensors in hex space (e.g., transforming from axial to offset 
coordinates).
- `modules/` includes various modules which may be useful in different parts of a model (e.g., word embedder).