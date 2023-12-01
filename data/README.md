# Data

Utilities and classes defining data for CerealBar.

- `bpe_tokenizer.py` tokenizes instructions with a pre-computed byte-pair-embedding scheme.
- `example.py` includes data structures for examples in the dataset.
- `step_example.py` is a per-step example, mapping an instruction, observation, and previous actions to a target set 
of plausible positions and a target action.
- `game.py` includes a game object, which maps each game to static environment information.
- `loading.py` contains functions for loading data.
- `dataset_split.py` defines different kinds of splits of the dataset for tagging individual examples.
- `dataset.py` defines collections of examples. Each object of type `Dataset` contains a collection of examples in a 
slice of the dataset, for example, the training or development dataset. A `DatasetCollection` collects various splits
 together, including the games (static environment data) for each.