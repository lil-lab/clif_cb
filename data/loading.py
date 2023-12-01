"""Loads data from disk."""
from __future__ import annotations

import logging
import os
import pickle
import random

from tqdm import tqdm

from config.data_config import DataConfig
from data.dataset import Dataset, DatasetCollection, GamesCollection
from data.dataset_split import DatasetSplit

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.example import Example
    from data.game import Game

    from typing import Dict, List, Set

INSTRUCTION_PATH: str = 'preprocessed/examples/'
GAME_PATH: str = 'preprocessed/games/'
PICKLE_SUFFIX: str = '.pkl'

EVAL_NUM_DEBUG_EXAMPLES: int = 5


def _extract_game_id(filename: str) -> str:
    return filename.split('_')[1].split('-')[0]


def _get_debug_filenames(filenames: List[str], num_examples: int,
                         splits: List[DatasetSplit]) -> List[str]:
    debug_split: DatasetSplit = DatasetSplit.TRAIN
    if len(splits) == 1 and (DatasetSplit.DEV in splits
                             or DatasetSplit.VALIDATION in splits):
        debug_split = splits[0]
        num_examples = EVAL_NUM_DEBUG_EXAMPLES

    return [
        filename for filename in filenames
        if filename.startswith(debug_split.shorthand())
    ][:num_examples]


def _select_training_subset(filenames: List[str], prop_training_data: float,
                            additional_num_steps: int) -> List[str]:
    # Need to limit the amount of training data.
    # First, grab unique game IDs
    unique_train_game_ids: Set[str] = set()
    for filename in filenames:
        if filename.startswith(DatasetSplit.TRAIN.shorthand()):
            unique_train_game_ids.add(_extract_game_id(filename))
    for game_id in unique_train_game_ids:
        if not os.path.exists(
                os.path.join(GAME_PATH, f'{game_id}{PICKLE_SUFFIX}')):
            raise ValueError(
                f'Extracted game ID {game_id} from instructions, but game was not in the games '
                f'folder.')

    # Then, randomly sample games (from a fixed seed). Grab only the proportion of games to keep.
    r: random.Random = random.Random(72)
    unique_train_game_ids: List[str] = sorted(list(unique_train_game_ids))
    r.shuffle(unique_train_game_ids)
    num_to_keep: int = int(prop_training_data * len(unique_train_game_ids))
    kept_train_game_ids = unique_train_game_ids[:num_to_keep]
    logging.info(
        f'Keeping {(prop_training_data * 100.):.1f}% of training examples; this encompasses '
        f'{len(kept_train_game_ids)} games.')

    # Keep only filenames which include this game ID, or filenames which don't start with 'train' (i.e.,
    # validation).
    base_filenames = [
        filename for filename in filenames
        if _extract_game_id(filename) in kept_train_game_ids
        or not filename.startswith(DatasetSplit.TRAIN.shorthand())
    ]

    if additional_num_steps:
        # Add more instructions
        logging.info(
            f'Also selecting {additional_num_steps} additional steps (as close as possible to this number).'
        )
        additional_step_example_filenames: List[str] = list()

        num_added_steps: int = 0
        for game_id in unique_train_game_ids[num_to_keep:]:
            example_filenames = [
                filename for filename in filenames
                if _extract_game_id(filename) == game_id
            ]
            for example_filename in example_filenames:
                with open(os.path.join(INSTRUCTION_PATH, example_filename),
                          'rb') as infile:
                    example: Example = pickle.load(infile)

                num_steps: int = len(example.target_action_sequence)
                additional_step_example_filenames.append(example_filename)

                num_added_steps += num_steps

                if num_added_steps > additional_num_steps:
                    break
            if num_added_steps > additional_num_steps:
                break

        logging.info(
            f'Added {len(additional_step_example_filenames)} additional instruction examples, which covers '
            f'{num_added_steps} training steps (as close as possible to {additional_num_steps} requested '
            f'steps.)')
        base_filenames.extend(additional_step_example_filenames)

    logging.info(
        f'Keeping {len(base_filenames)} examples in total (over all specified splits).'
    )

    return base_filenames


def _load_training_subset(filenames: List[str],
                          game_id_filename: str) -> List[str]:
    with open(game_id_filename) as infile:
        keep_games: Set[str] = {
            line.strip()
            for line in infile.readlines() if line.strip()
        }

    filenames = [
        filename for filename in filenames
        if _extract_game_id(filename) in keep_games
        or not filename.startswith(DatasetSplit.TRAIN.shorthand())
    ]

    logging.info(
        f'Specified {len(keep_games)} games to keep via file {game_id_filename}; this results in keeping '
        f'{len(filenames)} instructions (over all specified splits).')
    return filenames


def load_instructions(data_config: DataConfig,
                      splits: List[DatasetSplit],
                      debug: bool = False) -> Dict[DatasetSplit, Dataset]:
    """Loads instructions from disk and returns a dataset for each split."""
    if not os.path.exists(INSTRUCTION_PATH):
        raise NotADirectoryError(
            f'Directory containing instruction examples does not exist. Did you forget to unzip '
            f'preprocessed.zip?')

    def _keep_example_from_split(filename: str) -> bool:
        for s in splits:
            if s == DatasetSplit.HELD_OUT_TRAIN:
                if filename.startswith(DatasetSplit.TRAIN.shorthand()):
                    return True
            elif filename.startswith(s.shorthand()):
                return True

        return False

    datasets: Dict[DatasetSplit,
                   List[Example]] = {split: list()
                                     for split in splits}

    filenames: List[str] = sorted([
        filename for filename in os.listdir(INSTRUCTION_PATH) if
        filename.endswith(PICKLE_SUFFIX) and _keep_example_from_split(filename)
    ])

    update_train_filenames: List[str] = list()

    if debug:
        filenames = _get_debug_filenames(filenames,
                                         data_config.debug_num_examples,
                                         splits)
    elif DatasetSplit.TRAIN in splits or DatasetSplit.HELD_OUT_TRAIN in splits:
        if data_config.prop_training_data < 1:
            update_train_filenames = _select_training_subset(
                filenames, data_config.prop_training_data,
                data_config.additional_num_steps)
            if DatasetSplit.HELD_OUT_TRAIN not in splits:
                filenames = update_train_filenames

        elif data_config.game_id_filename:
            filenames = _load_training_subset(filenames,
                                              data_config.game_id_filename)

    splits_id: str = ', '.join([str(s) for s in splits])
    print(f'loading {len(filenames)} instructions (splits: {splits_id})')
    for filename in tqdm(filenames):
        if (filename.startswith(DatasetSplit.TRAIN.shorthand())
                and filename in update_train_filenames
                and DatasetSplit.TRAIN not in splits):
            # The original data isn't being loaded here, but will still appear in this list.
            continue

        with open(os.path.join(INSTRUCTION_PATH, filename), 'rb') as infile:
            data: Example = pickle.load(infile)

        if data_config.full_observability:
            data.set_fully_observable()
        else:
            data.set_maximum_memory_age(data_config.maximum_memory_age)

        split: DatasetSplit = data.dataset_split

        if DatasetSplit.HELD_OUT_TRAIN in splits and filename not in update_train_filenames:
            # This example is loaded anyway, but we need to change its split and put it in the other split list
            data.dataset_split = DatasetSplit.HELD_OUT_TRAIN
            datasets[DatasetSplit.HELD_OUT_TRAIN].append(data)
        else:
            datasets[split].append(data)

    return {
        split: Dataset(split, examples)
        for split, examples in datasets.items()
    }


def load_games(game_ids: Set[str],
               directory: str = GAME_PATH) -> Dict[str, Game]:
    """Loads the game objects for each specified game ID."""
    if not os.path.exists(directory):
        raise NotADirectoryError(
            f'Directory containing game examples does not exist. Did you forget to unzip '
            f'preprocessed.zip?')

    games: Dict[str, Game] = dict()

    print(f'loading {len(game_ids)} games')
    for game_id in tqdm(game_ids):
        filename: str = f'{game_id}{PICKLE_SUFFIX}'
        path: str = os.path.join(directory, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Game file {filename} was not found in game directory {directory}.'
            )

        with open(path, 'rb') as infile:
            game: Game = pickle.load(infile)
        games[game_id] = game

    return games


def load_recorded_data(
        data_config: DataConfig,
        dataset_ids: List[str],
        dataset_directory: str,
        use_ips: Dict[str, bool],
        val_only: bool = False,
        limit_to_examples: Set[str] = None) -> DatasetCollection:
    """Loads game and rollout data from human-agent games."""
    games_dict: Dict[str, Game] = dict()

    online_datasets: Dict[DatasetSplit, Dict[str, Dataset]] = {
        DatasetSplit.TRAIN: dict(),
        DatasetSplit.VALIDATION: dict()
    }

    for did in dataset_ids:
        # Load the instructions
        examples_directory: str = os.path.join(dataset_directory, did,
                                               'examples')
        filenames: List[str] = sorted([
            filename for filename in os.listdir(examples_directory)
            if filename.endswith(PICKLE_SUFFIX)
        ])

        print(
            f'loading {len(filenames)} instructions from recorded dataset {did}'
        )

        if limit_to_examples:
            keep_filenames: List[str] = list()
            for filename in filenames:
                exid = filename.split('.')[0].split('_')[1]
                if exid in limit_to_examples:
                    keep_filenames.append(filename)
            filenames = keep_filenames

        train_examples: List[Example] = list()
        val_examples: List[Example] = list()

        game_ids: Set[str] = set()
        for filename in tqdm(filenames):
            if val_only and 'val' not in filename:
                continue
            with open(os.path.join(examples_directory, filename),
                      'rb') as infile:
                data: Example = pickle.load(infile)

            if data_config.full_observability:
                data.set_fully_observable()
            else:
                data.set_maximum_memory_age(data_config.maximum_memory_age)

            if data.dataset_split == DatasetSplit.TRAIN:
                if not filename.startswith(DatasetSplit.TRAIN.shorthand()):
                    raise ValueError(
                        f'Example {filename} has inconsistent train/val setting: it is marked as a training example, '
                        f'but the filename is incorrect.')
                if not val_only:
                    train_examples.append(data)
            else:
                # Val example.
                if not filename.startswith(
                        DatasetSplit.VALIDATION.shorthand()):
                    raise ValueError(
                        f'Example {filename} has inconsistent train/val setting: it is marked as a validation example, '
                        f'but the filename is incorrect.')
                val_examples.append(data)

            game_ids.add(data.get_game_id())

        # Load the games
        dataset_games: Dict[str, Game] = load_games(
            game_ids, os.path.join(dataset_directory, did, 'games'))

        for gid, game in dataset_games.items():
            if gid in games_dict:
                raise ValueError(f'Already loaded game ID {gid}.')
            games_dict[gid] = game

        train_dataset: Dataset = Dataset(DatasetSplit.TRAIN, train_examples)
        val_dataset: Dataset = Dataset(DatasetSplit.VALIDATION, val_examples)

        train_dataset.construct_feedback_step_examples(did, use_ips[did])
        val_dataset.construct_feedback_step_examples(did, use_ips[did])

        online_datasets[DatasetSplit.TRAIN][did] = train_dataset
        online_datasets[DatasetSplit.VALIDATION][did] = val_dataset

    # Training data: online, all recorded data.
    return DatasetCollection(dict(),
                             GamesCollection(games_dict, dict()),
                             online_datasets=online_datasets)
