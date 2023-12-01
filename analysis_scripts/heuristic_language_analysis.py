from __future__ import annotations

import nltk
import numpy as np
from tqdm import tqdm

from config.data_config import DataConfig, TokenizerConfig
from data.dataset_split import DatasetSplit
from data.loading import load_recorded_data
from environment.action import Action

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.dataset import DatasetCollection
    from data.example import Example
    from typing import List

# Load the examples to analyze
dids: List[str] = [
    #'1_29', '1_30',
    '1_31',
    '2_1',
    '2_2',
    '2_3',
    '2_5',
    '2_7',
    '2_9',
    '2_12',
    '2_16'
]
with open("annotation/sampled_examples.txt") as infile:
    sampled_examples = [
        line.strip() for line in infile.readlines() if line.strip()
    ]

COUNTS = {
    '3': 3,
    '2': 2,
    'two': 2,
    'three': 3,
    '1': 1,
    'single': 1,
    'one': 1,
    'double': 2,
    'triple': 3,
    'thre': 3
}
COLORS = {
    'green': 'green',
    'blue': 'blue',
    'black': 'black',
    'red': 'red',
    'yellow': 'yellow',
    'yel': 'yellow',
    'orange': 'orange',
    'pink': 'pink',
    'purple': 'pink',
    'blk': 'black',
    'yello': 'yellow',
    'grn': 'green',
    'blu': 'blue',
    'gree': 'green',
    'orangle': 'orange',
    'oranges': 'orange',
    'ornage': 'orange',
    'yelow': 'yellow',
    'reds': 'red',
    'purp': 'pink',
    'blacks': 'black',
    'yellows': 'yellow',
    'orang': 'orange',
    'balck': 'black',
    'pinks': 'pink',
    'greens': 'green',
    'blues': 'blue',
    'gren': 'green',
    'yelloe': 'yellow',
    'blck': 'black'
}
SHAPES = {
    'hearts': 'heart',
    'circles': 'torus',
    'squares': 'cube',
    'stars': 'star',
    'triangles': 'triangle',
    'star': 'star',
    'square': 'cube',
    'circle': 'torus',
    'heart': 'heart',
    'triangle': 'triangle',
    'plus': 'plus',
    'diamonds': 'diamond',
    'diamond': 'diamond',
    'crosses': 'plus',
    'cross': 'plus',
    'lines': 'diamond',
    'line': 'diamond',
    'plusses': 'plus',
    'pluses': 'plus',
    'dash': 'diamond',
    'stripe': 'diamond',
    'sqaures': 'cube',
    'sqaure': 'cube',
    'cricles': 'torus',
    'stripes': 'diamond',
    'dashes': 'diamond',
    'sq': 'cube',
    'tri': 'triangle',
    'cir': 'torus',
    '+': 'plus',
    'traingle': 'triangle',
    'traingles': 'triangle',
    'trinagle': 'triangle',
    'cirlce': 'torus',
    'slash': 'diamond',
    'circ': 'torus',
    'trianges': 'triangle',
    'tirangles': 'triangle',
    'slits': 'diamond',
    'tria': 'triangle',
    'triangels': 'triangle',
    'cirlces': 'torus',
    'quare': 'cube',
    'squared': 'cube',
    'sqares': 'cube',
    'sqare': 'cube',
    'trangles': 'triangle',
    'squres': 'cube',
    'stary': 'star',
    'corsses': 'plus',
    'trianglesg': 'triangle',
    'swuare': 'cub:w'
    'e',
    'diamon': 'diamond',
    'hearst': 'heart'
}

CARD_REFS = [
    'card', 'cards', 'deselect', 'take', 'grab', 'pick', 'collect', 'step',
    'select', 'unselect', 'hit', 'undo', 'reselect', 'deselct', 'deslect',
    'cad', 'ecard'
]

PROPS = [
    'path', 'house', 'lake', 'tree', 'trees', 'water', 'road', 'windmill',
    'mountain', 'tower', 'roof', 'glacier', 'tent', 'pond', 'houses',
    'building', 'grass', 'flower', 'flowers', 'rock', 'hill', 'lamp', 'field',
    'post', 'bush', 'hut', 'pathway', 'ice', 'palm', 'pine', 'wooden',
    'bushes', 'lakes', 'lamppost', 'plants', 'plant', 'grassy', 'towers',
    'iceberg', 'snowy', 'tree'
]

num_card_ref_dict = {did: dict() for did in dids}
ref_type_dict = {did: dict() for did in dids}

ref_patterns = set()

prop_refs = {did: list() for did in dids}

num_printed = 0

num_single_card = {did: 0 for did in dids}
num_correct_single_card = {did: 0 for did in dids}
num_total = {did: 0 for did in dids}

instr_types = {did: {'all_pos': 0, 'all_neg': 0, 'mixed': 0} for did in dids}


def get_card_refs(instruction: str):
    num_count_refs = 0
    num_color_refs = 0
    num_shape_refs = 0

    color_refs = list()
    count_refs = list()
    shape_refs = list()

    has_card_ref = False
    prev_color = False
    toks: List[str] = nltk.wordpunct_tokenize(instruction.lower())

    for tok in toks:
        if prev_color:
            prev_color = False
            if tok in PROPS:
                num_color_refs -= 1
        if tok in COUNTS:
            num_count_refs += 1
            count_refs.append(COUNTS[tok])
        if tok in COLORS:
            num_color_refs += 1
            prev_color = True
            color_refs.append(COLORS[tok])
        if tok in SHAPES:
            num_shape_refs += 1
            shape_refs.append(SHAPES[tok])
        if tok in CARD_REFS:
            has_card_ref += 1

    num_card_refs = max(num_count_refs, num_color_refs, num_shape_refs)
    if num_card_refs == 0 and has_card_ref:
        num_card_refs = 1

    return num_card_refs, (color_refs, count_refs, shape_refs)


def get_has_prop(instruction: str) -> bool:
    toks: List[str] = nltk.wordpunct_tokenize(instruction.lower())
    for tok in toks:
        if tok in PROPS:
            return True
    return False


def analyze_examples(exs: List[Example], dataset_id):
    max_refs = 0
    num_rb_with_ref = dict()
    for example in tqdm(exs):
        instruction: str = example.instruction

        has_prop = get_has_prop(instruction)
        num_card_refs, (color_refs, count_refs,
                        shape_refs) = get_card_refs(instruction)

        if example.target_action_sequence[-1].target_action != Action.STOP:
            has_rb = False
            for step in example.target_action_sequence:
                if step.feedback_annotation.feedback.reboot:
                    has_rb = True
            if has_rb:
                if num_card_refs not in num_rb_with_ref:
                    num_rb_with_ref[num_card_refs] = 0
                num_rb_with_ref[num_card_refs] += 1

            continue

        has_stop_action = False
        has_pos = False
        has_neg = False
        for step in example.target_action_sequence:
            if step.target_action == Action.STOP or step.feedback_annotation.feedback.reboot:
                has_stop_action = True
            if step.feedback_annotation.feedback.reboot or step.feedback_annotation.feedback.polarity(
            ) < 0:
                has_neg = True
            elif step.feedback_annotation.feedback.polarity() > 0:
                has_pos = True
        if has_stop_action:
            num_total[dataset_id] += 1

        if has_pos and has_neg:
            instr_types[dataset_id]['mixed'] += 1
        elif has_pos:
            instr_types[dataset_id]['all_pos'] += 1
        elif has_neg:
            instr_types[dataset_id]['all_neg'] += 1

        if num_card_refs == 1 and has_stop_action:
            # Find the set of cards picked up.
            num_single_card[dataset_id] += 1
            sel_cards = list()
            prev_cards = {
                c.position: c
                for c in example.target_action_sequence[0].previous_state.cards
            }
            for i, step in enumerate(example.target_action_sequence[:-1]):
                if step.target_action in {Action.MF, Action.MB}:
                    sub_pos = example.target_action_sequence[
                        i + 1].previous_state.follower.position
                    if sub_pos in prev_cards:
                        sel_cards.append(prev_cards[sub_pos])

                prev_cards = {
                    c.position: c
                    for c in example.target_action_sequence[
                        i + 1].previous_state.cards
                }

            if len(sel_cards) == 1:
                target_color = ''
                target_shape = ''
                target_count = ''

                if color_refs:
                    target_color = color_refs[0]

                if shape_refs:
                    target_shape = shape_refs[0]

                if count_refs:
                    target_count = count_refs[0]

                sel_card = sel_cards[0]

                correct_color = True
                correct_shape = True
                correct_count = True

                if target_color and target_color.lower() != str(
                        sel_card.color).lower():
                    correct_color = False
                if target_shape and target_shape.lower() != str(
                        sel_card.shape).lower():
                    correct_shape = False
                if target_count and str(target_count) != str(
                        sel_card.count).lower():
                    correct_count = False

                if correct_color and correct_shape and correct_count:
                    num_correct_single_card[dataset_id] += 1

        if num_card_refs not in num_card_ref_dict[dataset_id]:
            num_card_ref_dict[dataset_id][num_card_refs] = 0
        num_card_ref_dict[dataset_id][num_card_refs] += 1

        if num_card_refs > 0:
            ref_type_pattern = (len(shape_refs) + len(count_refs) +
                                len(color_refs)) / num_card_refs
            if ref_type_pattern not in ref_type_dict[dataset_id]:
                ref_type_dict[dataset_id][ref_type_pattern] = 0
            ref_type_dict[dataset_id][ref_type_pattern] += 1

            ref_patterns.add(ref_type_pattern)

        if num_card_refs > max_refs:
            max_refs = num_card_refs

        prop_refs[dataset_id].append(int(has_prop))

    for num_card_refs in range(max_refs):
        if num_card_refs in num_card_ref_dict[dataset_id]:
            print(num_card_refs, num_card_ref_dict[dataset_id][num_card_refs])


def eval_dataset(dataset_id):
    # Load the dataset
    data: DatasetCollection = load_recorded_data(DataConfig(TokenizerConfig()),
                                                 [dataset_id],
                                                 'game_recordings/',
                                                 {dataset_id: True})

    examples: List[Example] = data.online_datasets[
        DatasetSplit.TRAIN][dataset_id].instruction_examples

    analyze_examples(examples, dataset_id)
    print(
        f'{dataset_id}\t{num_single_card[dataset_id]}\t{num_correct_single_card[dataset_id]}'
    )


def main():
    for did in dids:
        eval_dataset(did)

    exit()
    print('props')
    for did, proplist in prop_refs.items():
        print(f'{did}\t{np.mean(np.array(proplist))}')

    print('number of card references')
    print('\t' + '\t'.join([str(i) for i in range(max_refs + 1)]))
    for did, refs in num_card_ref_dict.items():
        ref_counts = list()
        for i in range(max_refs + 1):
            if i in refs:
                ref_counts.append(refs[i])
            else:
                ref_counts.append(0)
        print(did + '\t' + '\t'.join([str(count) for count in ref_counts]))

    print('')
    print('type of card references')
    for did, refs in ref_type_dict.items():
        print(did)
        for pattern in sorted(list(ref_patterns)):
            count = 0
            if pattern in refs:
                count = refs[pattern]
            print(f'\t{count}\t{pattern}')

    for did, count in num_total.items():
        print(
            f'{did}\t{count}\t{num_single_card[did]}\t{num_correct_single_card[did]}'
        )

    for did, types in instr_types.items():
        print(
            f'{did}\t{types["all_pos"]}\t{types["all_neg"]}\t{types["mixed"]}')


if __name__ == '__main__':
    main()
