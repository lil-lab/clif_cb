# Installation

1. Create the conda environment: 
```conda create -n cb_vin python=3.7```
1. Make sure that git-LFS is installed before cloning the repo.
1. Clone the repo.
1. Install the requirements: ```pip install -r requirements.txt```
1. Unzip the preprocessed data: `unzip preprocessed.zip`. This will create a new directory `preprocessed/` including 
two subdirectories: (1) `examples/` includes a pickle file for each example in the dataset; the filenames are prefixed
 with which split the example is from. (2) `games/` includes a pickle file for each original CerealBar game, which 
 includes the game ID and all the static environment information for that game.
 1. Make a directory to store experiments: ```mkdir experiments/```

You will need to install pytorch separately according to your machine's requirements. We tested using torch 1.6.0 and
 1.2.0. See [this page](https://pytorch.org/get-started/previous-versions/) for details.
 
 # Subdirectories
 - `config/` defines configuration files, including for running the program, model parameters, etc. Configurations 
 are dataclasses and stored as JSON files on disk in the `json_configs/` file.
 - `data/` defines classes and functions for loading and storing data, including instructions and game information.
 - `environment/` defines information about the CerealBar environment, including props, positions, cards, etc.
 - `evaluation/` is used to evaluate models; defines metrics, evaluation functions, and evaluation loops.
 - `experiments/` contains subdirectories for each experiment ran. 
 - `inference/` is used to run inference on a model.
 - `json_configs/` includes configurations for running the program.
 - `learning/` is used to train a model.
 - `protobuf/` is used to store the protobuf format for communication with the standalone or the web server.
 - `model/` defines the model architectures.
 - `preprocessed/` contains preprocessed data (instruction examples and games).
 - `simulation/` contains code for simulating the CerealBar game, either through the Unity standalone or a python copy.
 - `util/` contains various utilities.
 - `web_agent/` contains code necessary for running the model in live games in browser.
