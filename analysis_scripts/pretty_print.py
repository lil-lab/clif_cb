from __future__ import annotations

import copy
import random
import tqdm

from config.data_config import DataConfig, TokenizerConfig, FeedbackHeuristicsConfig
from data.dataset_split import DatasetSplit
from data.loading import load_recorded_data
from environment.card import CardSelection

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.dataset import DatasetCollection
    from data.example import Example
    from environment.card import Card
    from environment.position import Position
    from typing import Dict, List, Set

# Load the examples to analyze
dids: List[str] = [
    '1_29', '1_30', '1_31', '2_1', '2_2', '2_3', '2_5', '2_7', '2_9', '2_12'
]
examples: Set[str] = set()

NUM_EXS: int = 25

with open(f'single_card_exs.txt') as infile:
    examples |= {line.strip() for line in infile.readlines() if line.strip()}

print(f'Loaded {len(examples)} examples and {len(dids)} datasets')

# Load the dataset
data: DatasetCollection = load_recorded_data(DataConfig(TokenizerConfig()),
                                             dids,
                                             'game_recordings/',
                                             {did: True
                                              for did in dids},
                                             limit_to_examples=examples)

num_with_empty_suffix = 0
num_empty_suffix_with_reboot = 0
num_neg = 0
total_num = 0
with open(f'analysis/single_card_examples.txt', 'w') as ofile, \
        open(f'analysis/single_card_examples_list.txt', 'w') as lfile:
    all_examples = list()
    ex_to_did = dict()
    for did in dids:
        examples: List[Example] = data.online_datasets[
            DatasetSplit.TRAIN][did].instruction_examples
        random.Random(72).shuffle(examples)
        for ex in examples[:NUM_EXS]:
            ex_to_did[ex.example_id] = did

        all_examples.extend(examples[:NUM_EXS])

    random.Random(72).shuffle(all_examples)

    num_printed = 0
    for example in tqdm.tqdm(all_examples):
        total_num += 1

        step_actions: List[str] = list()
        raw_fbs: List[str] = list()
        select_actions: List[str] = list()
        heuristic_fbs: List[str] = list()

        example_copy = copy.deepcopy(example)
        example_copy.reannotate_feedback_with_heuristics(
            FeedbackHeuristicsConfig(new_fitb=True))

        has_card_selection: bool = False
        used_heuristics: List[bool] = list()
        reboot = list()
        has_reboot: bool = False

        has_pos: bool = False
        has_neg: bool = False

        empty_suffix_pos = False

        for i, step in enumerate(example.step_examples):
            # Get the raw feedback
            fb: int = step.action_annotation.feedback.polarity()

            if fb > 0:
                has_pos = True
                raw_fb = '+'
            elif fb < 0:
                has_neg = True
                raw_fb = '-'
            else:
                raw_fb = '0'
            raw_fbs.append(raw_fb)

            reboot.append(
                'rb' if step.action_annotation.feedback.reboot else '  ')
            if step.action_annotation.feedback.reboot:
                has_reboot = True

            # TODO: Get the heuristic feedback
            heuristic_fb: int = example_copy.step_examples[
                i].action_annotation.feedback.polarity()

            if heuristic_fb > 0:
                heur_fb = '+'
            elif heuristic_fb < 0:
                heur_fb = '-'
            else:
                heur_fb = '0'
            heuristic_fbs.append(heur_fb)

            used_heuristics.append(fb == 0 and heuristic_fb != 0)

            if i < len(example.step_examples) - 1 and fb == 0:
                in_empty_suffix = True
                for subsequent_step in example.step_examples[i + 1:]:
                    if subsequent_step.action_annotation.feedback.polarity(
                    ) != 0:
                        in_empty_suffix = False
                if in_empty_suffix and heuristic_fb > 0:
                    empty_suffix_pos = True

            # Get the card annotation (if any)
            next_pos: Position = step.sampled_resulting_configuration.position
            current_pos: Position = step.state.follower.position

            current_cards: Dict[Position, Card] = {
                card.position: card
                for card in step.state.cards
            }

            if next_pos != current_pos and next_pos in current_cards:
                card_touched: Card = current_cards[next_pos]

                if card_touched.selection == CardSelection.SELECTED:
                    sel_str = 'd'
                else:
                    sel_str = 's'

                card_sel_str = f'{sel_str}' \
                               f'{card_touched.count}' \
                               f'{card_touched.color.shorthand()}' \
                               f'{card_touched.shape.shorthand()}'

                has_card_selection = True
            else:
                card_sel_str = '     '

            step_actions.append(step.sampled_action.shorthand())
            select_actions.append(card_sel_str)

        has_neg_card_selection: bool = False
        final_neg_card_selection: int = -1
        if has_card_selection:
            for i, step in enumerate(select_actions):
                if step.strip() and heuristic_fbs[i] == '-':
                    has_neg_card_selection = True
                    final_neg_card_selection = i

        has_neg_heuristic_suffix: bool = False
        if has_neg_card_selection:
            has_neg_heuristic_suffix: bool = True
            for heu in used_heuristics[final_neg_card_selection:]:
                if not heu:
                    has_neg_heuristic_suffix = False

        if has_neg_heuristic_suffix:
            if has_reboot:
                num_empty_suffix_with_reboot += 1
            else:
                num_with_empty_suffix += 1

        if has_neg_card_selection:
            num_neg += 1

        lfile.write(example.example_id + '\n')
        ofile.write(f'----- {did} {example.example_id} -----\n')
        ofile.write(example.instruction + '\n')

        ofile.write('      '.join([str(i)
                                   for i in range(len(step_actions))]) + '\n')
        ofile.write('      '.join(step_actions) + '\n')
        ofile.write('      '.join(raw_fbs) + '\n')
        ofile.write('      '.join(heuristic_fbs) + '\n')
        ofile.write('  '.join(select_actions) + '\n')

        if has_reboot:
            ofile.write('     '.join(reboot))

        ofile.write('\n')

        if has_card_selection:
            ofile.write('Card selections:\n')
            for i, action in enumerate(select_actions):
                if action.strip():
                    if used_heuristics[i]:
                        heur = f'(heu, {heuristic_fbs[i]})'
                    else:
                        heur = f'(raw, {heuristic_fbs[i]})'
                    ofile.write(f'{i} {action} {heur}: \n')
        ofile.write('\n')

        num_printed += 1

        # if num_printed >= NUM_EXS:
        #    break

print(num_with_empty_suffix)
print(num_empty_suffix_with_reboot)
print(num_neg)
print(total_num)
