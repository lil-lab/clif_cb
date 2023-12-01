"""Scripts to find and analyze potential errors in live games."""
from __future__ import annotations

from absl import app, flags
from random import Random

from config.data_config import DataConfig, TokenizerConfig
from data import loading
from data.dataset_split import DatasetSplit
from environment.card import CardSelection

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.dataset import Dataset, DatasetCollection
    from data.example import Example
    from environment.card import Card
    from environment.position import Position
    from environment.state import State
    from typing import Dict, List, Set

FLAGS = flags.FLAGS

DATASETS: List[str] = [
    '11_15', '11_16', '11_17', '11_18', '11_22', '12_14', '12_22'
]

RESTRICT_TO_CARD_EVENTS: bool = False


def _contains_deselection(example: Example):
    for i, step in enumerate(example.step_examples[:-1]):
        next_step = example.step_examples[i + 1]

        prev_state: State = step.state
        next_state: State = next_step.state

        pos_to_card_prev: Dict[Position, Card] = {
            card.position: card
            for card in prev_state.cards
        }
        pos_to_card_next: Dict[Position, Card] = {
            card.position: card
            for card in next_state.cards
        }

        same_pos: Set[Position] = set(pos_to_card_prev.keys()) & set(
            pos_to_card_next.keys())

        same_pos = {
            pos
            for pos in same_pos
            if pos_to_card_next[pos] == pos_to_card_prev[pos]
        }

        for pos in same_pos:
            if pos_to_card_prev[
                    pos].selection == CardSelection.SELECTED and pos_to_card_next[
                        pos].selection == CardSelection.UNSELECTED:
                return True

    return False


def _find_erroneous_examples(did: str, dataset: Dataset, rng: Random):
    num_with_deselection: int = 0

    deselection_examples: Set[str] = set()
    for example in dataset.instruction_examples:
        # Use heuristics to find examples.

        # First heuristic: does it include a deselection? If so, choose the previous instruction in the game.
        contains_deselection = _contains_deselection(example)

        if contains_deselection:
            num_with_deselection += 1

            # Find the previous instruction
            instruction_idx: int = example.get_instruction_idx()

            if instruction_idx > 0:
                game_id: str = example.get_game_id()

                prev_instr_id: str = f'{game_id}-{instruction_idx - 1}'
                deselection_examples.add(prev_instr_id)

        # Second heuristic: was instruction rebooted?
        pass

        # Third heuristic: is there negative feedback?
        pass

        # Fourth heuristic: does selected set become invalid?

    num_exs: int = len(dataset.instruction_examples)
    print(
        f'{(100. * num_with_deselection / num_exs):.1f}% contain a deselection.'
    )

    with open(f'analysis/errors_deselect_{did}.txt', 'w') as ofile:
        desel: List[str] = list(deselection_examples)
        rng.shuffle(desel)

        for exid in desel:
            ofile.write(f'{exid}\n')


def main(argv):
    data_config: DataConfig = DataConfig(TokenizerConfig())
    data: DatasetCollection = loading.load_recorded_data(
        data_config,
        DATASETS,
        'game_recordings/', {dataset: True
                             for dataset in DATASETS},
        val_only=False)

    rng: Random = Random(72)

    for did, dataset in data.online_datasets[DatasetSplit.TRAIN].items():
        _find_erroneous_examples(did, dataset, rng)


if __name__ == '__main__':
    app.run(main)
