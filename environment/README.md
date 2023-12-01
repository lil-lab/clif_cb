# Environment

Files for defining the CerealBar environment.

- `action.py`: player actions, e.g., moving, rotating, and stopping.
- `state.py`: keeps track of a single state in the game, including information which may change (i.e., cards and 
player positions).
- `static_environment.py`: static information about a CerealBar environment, including props and terrain.
- `position.py`: positions in the CerealBar environment (offset coordinates).
- `rotation.py`: rotations in the CerealBar (hex) environment.
- `terrain.py`: terrains in the CerealBar environment.
- `player.py`: Classes for the two players (leader and follower).
- `card.py`: A card in the game, with color/shape/count and selection properties.
- `observation.py`: A follower's observation, including the believed set of cards and leader, and time since last 
observing each location.
- `sets.py` defines utilities associated with making sets.

In addition, the `static_props` directory contains static props in the environment (e.g., houses, plants).
