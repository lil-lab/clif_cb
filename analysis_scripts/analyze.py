"""Analyzes data in a dataset."""
from __future__ import annotations

import logging
import os
import numpy as np
import random
import sqlite3

from absl import app, flags
import nltk
from tqdm import tqdm

from config.data_config import DataConfig, TokenizerConfig, FeedbackHeuristicsConfig
from config.training_configs import SupervisedTargetConfig
from data import loading
from data.dataset_split import DatasetSplit
from environment.action import Action
from environment.card import CardSelection
from simulation.planner import find_path_between_positions
from learning.training import load_training_data

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.dataset import Dataset, DatasetCollection
    from data.game import Game
    from environment.player import Player
    from environment.position import Position
    from typing import Dict, List, Optional, Tuple

RESTRICT_TO_CARD_EVENTS: bool = False
USE_HEURISTICS: bool = True
DATASETS = []


def _analyze_action_feedback(dataset: Dataset, gid_to_wid):
    if USE_HEURISTICS:
        dataset.reannotate_feedback_with_heuristics(
            FeedbackHeuristicsConfig(new_fitb=True))

    action_feedbacks: Dict[Action, Dict[str, int]] = dict()
    event = ''

    instruction_pos_neg = dict()

    unique_instructions = dict()
    vocabs = dict()

    sentence_lengths = dict()

    random.shuffle(dataset.step_examples)

    worker_data = dict()

    num_nonneutral = 0
    for instruction in dataset.instruction_examples:
        has_nonneutral = False
        for step in instruction.step_examples:
            if step.action_annotation.feedback.polarity() != 0:
                has_nonneutral = True
                break

        if has_nonneutral:
            num_nonneutral += 1

    print(
        f'{num_nonneutral} / {len(dataset.instruction_examples)} = non-neutral / all instructions'
    )

    for i, step in enumerate(dataset.step_examples):

        if step.action_annotation is not None:
            action: Action = step.sampled_action
            pos: int = step.action_annotation.feedback.num_positive
            neg: int = step.action_annotation.feedback.num_negative
            reboot: bool = step.action_annotation.feedback.reboot

            annotation: str = 'neutral'
            if pos > neg:
                annotation = 'positive'
            elif neg > pos:
                annotation = 'negative'

            if reboot:
                annotation = 'negative'
            resulting_pos = step.sampled_resulting_configuration.position
        else:
            action = step.target_action
            annotation = 'positive'
            resulting_pos = step.next_target_configuration.position

        if step.example_id not in instruction_pos_neg:
            instruction_pos_neg[step.example_id] = dict()
        if annotation not in instruction_pos_neg[step.example_id]:
            instruction_pos_neg[step.example_id][annotation] = 0
        instruction_pos_neg[step.example_id][annotation] += 1

        if annotation == 'positive':
            instr = step.instruction.lower()
            toks = nltk.word_tokenize(instr)

            for tok in toks:
                if tok not in vocabs:
                    vocabs[tok] = 0
                vocabs[tok] += 1

            sen_len = len(toks)
            if sen_len not in sentence_lengths:
                sentence_lengths[sen_len] = 0

            if instr not in unique_instructions:
                sentence_lengths[sen_len] += 1

            if instr not in unique_instructions:
                unique_instructions[instr] = 0
            unique_instructions[instr] += 1

            if step.game_id not in gid_to_wid:
                print(f'could not find game {step.game_id} in database!')
            else:
                wid = gid_to_wid[step.game_id]
                if wid not in worker_data:
                    worker_data[wid] = 0
                worker_data[wid] += 1

        if RESTRICT_TO_CARD_EVENTS:
            if not (resulting_pos
                    in {card.position
                        for card in step.state.cards}
                    and resulting_pos != step.state.follower.position):
                continue

            for card in step.state.cards:
                if card.position == resulting_pos:
                    if card.selection == CardSelection.SELECTED:
                        event = f'deselected a card'
                    else:
                        event = f'selected a card'

        action = f'{action} {event}'
        if action not in action_feedbacks:
            action_feedbacks[action] = {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        action_feedbacks[action][annotation] += 1

    print(f'\tpos\tneg\tneut\tall')
    total_num_pos = 0
    for ac, feedbacks in action_feedbacks.items():
        print(
            f'{ac}\t{feedbacks["positive"]}\t{feedbacks["negative"]}\t{feedbacks["neutral"]}\t{sum(feedbacks.values())}'
        )
        total_num_pos += feedbacks['positive']

    return

    print('\n----- Instruction stats -----')
    print(f'\t{len(unique_instructions)} unique instructions')
    for instr, freq in sorted(unique_instructions.items(),
                              key=lambda x: x[1])[::-1][:10]:
        print(f'\t\t{freq}\t{instr}')

    print(
        f'\n\tVocabulary size: {len(vocabs)} (density: {len(vocabs) / len(unique_instructions)})'
    )

    print(f'\n----- Worker data -----')
    for wid, freq in worker_data.items():
        print(f'\t{wid}\t{freq}')

    num_pos_actions = 0
    num_pos_instr = 0
    for instr, data in instruction_pos_neg.items():
        if 'negative' in data:
            continue
        if 'positive' in data:
            num_pos_instr += 1
            num_pos_actions += data['positive']


def _analyze_instruction_feedback(dataset: Dataset):
    num_mixed: int = 0
    num_all_pos: int = 0
    num_all_neg: int = 0
    num_all_neutral: int = 0
    for example in dataset.instruction_examples:
        has_pos: bool = False
        has_neg: bool = False
        neutral_stop: bool = False

        # Check if it has a neutral stop
        for step in example.step_examples:

            if step.sampled_action == Action.STOP and step.action_annotation.feedback.is_neutral(
            ):
                neutral_stop = True

            polarity: int = step.action_annotation.feedback.polarity()
            if polarity > 0:
                has_pos = True
            elif polarity < 0:
                has_neg = True

        if neutral_stop:
            if has_pos and has_neg:
                num_mixed += 1
            elif has_pos:
                num_all_pos += 1
            elif has_neg:
                num_all_neg += 1
            else:
                num_all_neutral += 1


def _analyze_feedback(dataset: Dataset):
    _analyze_action_feedback(dataset)


def _analyze_shortest_paths(dataset: Dataset, games: Dict[str, Game]):
    path_lengths: List[int] = list()

    for instr in tqdm(dataset.instruction_examples):
        for step in instr.step_examples:
            current_follower: Player = step.state.follower
            target_follower: Player = step.action_annotation.sampled_goal_voxel

            if target_follower:
                avoid_positions: List[Position] = list(games[
                    instr.get_game_id()].environment.get_obstacle_positions())

                card_positions: List[Position] = [
                    card.position for card in step.state.cards
                ]
                for card_pos in card_positions:
                    if card_pos not in {
                            current_follower.position, target_follower.position
                    }:
                        # Only consider cards as obstacles if they are (1) not agent's starting position and (2) not the next
                        # card target along the demonstration (if it exists)
                        avoid_positions.append(card_pos)

                shortest_path: Optional[Tuple] = find_path_between_positions(
                    avoid_positions, current_follower, target_follower)

                if not shortest_path:
                    print('could not find shortest path!')

                else:
                    path_lengths.append(len(shortest_path[0]))

                    if len(shortest_path[0]) == 31:
                        print(instr.example_id)

    print(f'mean path length: {np.mean(np.array(path_lengths))}')
    print(f'median path length: {np.median(np.array(path_lengths))}')
    print(f'min path length: {np.min(np.array(path_lengths))}')
    print(f'max path length: {np.max(np.array(path_lengths))}')


def main(argv):
    for dataset in DATASETS:
        prefix = '../cb_ft_mturk/cb_ft_mturk/databases/games/' + dataset
        gid_to_wid = dict()
        for filename in os.listdir(prefix):
            if filename.endswith('.db'):
                print(filename)
                db = sqlite3.connect(os.path.join(prefix, filename))
                c = db.cursor()
                q = 'select gameId, workerId from games join clients on clients.clientSid = games.leaderSid'
                c.execute(q)
                res = c.fetchall()
                db.close()

                for gid, wid in res:
                    gid_to_wid[gid] = wid
        print(f'Loaded {len(gid_to_wid)} games')

        data: DatasetCollection = loading.load_recorded_data(
            DataConfig(TokenizerConfig()), [dataset],
            'game_recordings/',
            use_ips={dataset: False})

        logging.info(
            f'Loaded {data.online_datasets[DatasetSplit.TRAIN][dataset].get_num_instructions()} '
            f'training examples ({len(data.games.games)} games) from recorded human-agent games.'
        )

        _analyze_action_feedback(
            data.online_datasets[DatasetSplit.TRAIN][dataset], gid_to_wid)
        print(f'Finished analyzing {dataset}')


if __name__ == '__main__':
    app.run(main)
