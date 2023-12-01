# Simulation

Code for simulating the CerealBar game, either through the Unity standalone or a python copy.

- `server_socket.py`: Necessary for launching and communicating with the Unity standalone.
- `game.py`: An abstract game for keeping track of player positions and card states.
- `python_game.py`: For simulating the game in python.
- `unity_game.py` communicates with the Unity standalone.
- `planner.py`: Utilities for planning, e.g., to find the next configuration of a player given an action.