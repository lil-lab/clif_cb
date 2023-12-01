"""Converts recorded rollouts and data from online games into Games and Examples."""
from __future__ import annotations

import os
import pickle
import random
import sqlite3

from absl import app, flags
from dataclasses import dataclass
from datetime import datetime
from google.protobuf import text_format
from tqdm import tqdm

from data.dataset_split import DatasetSplit
from data.feedback import ActionFeedback, load_feedback_from_file, SampledActionAnnotation
from data.example import ActionStep, Example
from data.game import Game
from environment.action import Action
from environment.card import Card, load_cards_from_proto
from protobuf import CerealBarProto_pb2
from simulation.game import FOLLOWER_MOVES_PER_TURN
from web_agent.recording import RecordedGame, RecordedRollout

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple, Union

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'db_directory',
    '../cb_ft_mturk/cb_ft_mturk/databases'
    '/games/', 'The directory where the databases are stored.')
flags.DEFINE_string(
    'feedback_directory',
    '../cb_ft_mturk/cb_ft_mturk/games/feedback/',
    'Directory where feedback is stored.')
flags.DEFINE_string(
    'agent_rollout_directory',
    '../cb_ft_mturk/cb_ft_mturk/agent_rollouts/',
    'Directory where recorded rollouts and games are stored.')
flags.DEFINE_string(
    'save_directory',
    '../cb_vin_feedback/game_recordings/',
    'The directory where data is saved.')
flags.DEFINE_float(
    'delay_amount', 0.2,
    'A constant (expected) delay (in seconds) before a worker will process a move '
    'and adjust their feedback for it.')

flags.DEFINE_string(
    'unique_id', None,
    'The unique ID of the batch to process. Usually in the format M_DD.')
flags.DEFINE_string(
    'model_suffix', None,
    'A suffix for a model to take data from, if multiple models are in each '
    'database / feedback file.')

flags.mark_flag_as_required('unique_id')

GAME_STR: str = 'games'
EXAMPLES_STR: str = 'examples'

FOLLOWER: int = 'Follower'
LEADER: int = 'Leader'

DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S.%f'

VAL_PROP: float = 0.05


@dataclass
class FollowerAction:
    action_id: str
    executed_time: float
    server_time: datetime

    def is_finish_instruction(self) -> bool:
        return 'instruction' in self.action_id


@dataclass
class Instruction:
    action_id: str
    server_time: datetime


@dataclass
class SingleFeedbackSignal:
    action_id: str
    executed_time: float
    original_time: float
    polarity: str
    server_time: datetime


def get_game_ids_in_db(cursor) -> List[str]:
    q = 'select gameId from games'
    cursor.execute(q)

    ids = cursor.fetchall()

    game_ids = [
        game_id[0] for game_id in ids if not game_id[0].startswith('tut')
    ]

    if FLAGS.model_suffix:
        game_ids_for_model: List[str] = list()
        for game_id in game_ids:
            q = 'select workerId from clients where clientSid in (select followerSid from games where gameId=?)'
            t = (game_id, )
            cursor.execute(q, t)
            wid: str = cursor.fetchone()[0]
            if wid.endswith(FLAGS.model_suffix):
                game_ids_for_model.append(game_id)
        return game_ids_for_model
    else:
        return game_ids


def get_executed_instruction_filepaths(directory: str) -> List[str]:
    file_paths: List[str] = list()
    for filename in os.listdir(directory):
        if 'instruction' in filename:
            file_paths.append(os.path.join(directory, filename))
    return file_paths


def _convert_time(time_str: str) -> datetime:
    return datetime.strptime(time_str, DATE_FORMAT)


def _get_idx_from_action_id(action_id: str) -> int:
    return int(action_id.split('_')[-1])


def _get_num_steps_remaining(cursor, game_id: str, action: FollowerAction,
                             turn_id: int) -> int:
    q = 'select moveId, serverTime from movements where gameId=? and turnId=? order by moveId asc'
    t = (game_id, turn_id)
    cursor.execute(q, t)

    moves_in_turn: List[int] = cursor.fetchall()

    if action.is_finish_instruction():
        num_executed_before: int = 0

        for _, server_time_str in moves_in_turn:
            server_time: datetime = _convert_time(server_time_str)
            if server_time > action.server_time:
                break
            num_executed_before += 1

    else:
        move_idx: int = _get_idx_from_action_id(action.action_id)

        num_executed_before: int = 0
        for move_id, _ in moves_in_turn:
            if move_id == move_idx:
                break
            num_executed_before += 1

    return FOLLOWER_MOVES_PER_TURN - num_executed_before


def _find_last_movement_id_before_finish_instr(
        cursor, game_id: str, finish_command_action: FollowerAction,
        turn_start_times: List[datetime]) -> int:
    turn_of_last_instr: int = _get_turn_id_for_action(
        cursor, game_id, finish_command_action.action_id, turn_start_times)

    q = 'select moveId, serverTime from movements where gameId=? and turnId <= ?'
    t = (game_id, turn_of_last_instr)
    cursor.execute(q, t)
    movement_data: List[Tuple[int, str]] = cursor.fetchall()

    moves_before_finish_instr: List[Tuple[int, datetime]] = sorted(
        [(move_id, _convert_time(server_time))
         for move_id, server_time in movement_data
         if _convert_time(server_time) < finish_command_action.server_time],
        key=lambda x: x[1])
    return moves_before_finish_instr[-1][0]


def _get_turn_id_for_action(cursor, game_id: str, action_id: str,
                            turn_start_times: List[datetime]) -> int:
    fixed_action_id: int = _get_idx_from_action_id(action_id)
    if 'move' in action_id:

        q = 'select serverTime from movements where moveId=? and gameId=?'
        t = (fixed_action_id, game_id)
        cursor.execute(q, t)
        result: str = cursor.fetchone()
        if not result:
            raise ValueError(
                f'Could not find server time for {action_id} in movements table for game {game_id}.'
            )
        action_time: datetime = _convert_time(result[0])
    elif 'instruction' in action_id:

        q = 'select serverTime from commandFinishingActions where instructionId=? and gameId=?'
        t = (fixed_action_id, game_id)
        cursor.execute(q, t)
        result: str = cursor.fetchone()

        if not result:
            raise ValueError(
                f'Could not find server time for {action_id} in command finishing table for game {game_id}.'
            )
        action_time: datetime = _convert_time(result[0])
    else:
        raise ValueError(
            f'Action is not a movement or instruction-finishing action: {action_id}'
        )

    found_turn_id: int = -1
    for turn_id, start_time in enumerate(turn_start_times):
        end_time: Optional[datetime] = None
        if turn_id < len(turn_start_times) - 1:
            end_time = turn_start_times[turn_id + 1]
        if action_time >= start_time and (not end_time
                                          or action_time <= end_time):
            found_turn_id = turn_id
            break
    return found_turn_id


def _get_expected_sets(
        cursor, game_id: str, first_move_idx: int,
        last_move_idx: int) -> List[Tuple[List[Card], List[Card]]]:
    q = 'select moveId from movements where gameId=? and moveId between ? and ? and moveId in (select ' \
        'moveId from cardSets where gameId=?) order by moveId asc'
    t = (game_id, first_move_idx, last_move_idx, game_id)
    cursor.execute(q, t)
    result = cursor.fetchall()

    moves_with_set: List[int] = [val[0] for val in result]

    sets: List[Tuple[List[Card], List[Card]]] = list()
    for move_id in moves_with_set:
        t = (move_id, game_id)

        q = 'select cards from cardSets where moveId=? and gameId=?'
        cursor.execute(q, t)
        valid_set_data: str = cursor.fetchone()[0]

        card_list = CerealBarProto_pb2.CardList()
        text_format.Parse(valid_set_data, card_list)

        valid_cards: List[Card] = load_cards_from_proto(card_list.cardlist)

        q = 'select cards from newCards where moveId=? and gameId=?'
        cursor.execute(q, t)
        new_card_sets: str = cursor.fetchone()[0]
        new_cards: List[str] = new_card_sets.replace('[',
                                                     '').replace(']',
                                                                 '').split(',')
        card_str: str = ''
        for c in new_cards:
            card_str += f'newcards {{ {c} }}\n'
        score_set_card = CerealBarProto_pb2.ScoreSetCard()
        text_format.Parse(card_str, score_set_card)
        new_cards: List[Card] = load_cards_from_proto(score_set_card.newcards)

        sets.append((valid_cards, new_cards))

    return sets


def _create_instruction_obj(game_id: str, instruction_data: RecordedRollout,
                            leader_actions: List[List[Action]],
                            initial_num_steps: int,
                            expected_sets: List[Tuple[List[Card], List[Card]]],
                            instruction_feedback: Dict[str, ActionFeedback],
                            is_val: bool) -> Optional[Example]:

    # Get the target action sequence, including the previous state (leader/follower position, cards), observation (
    # what was visible at the time), and action taken
    target_action_sequence: List[ActionStep] = list()

    for action in instruction_data.get_actions():
        if action.global_game_action_id not in instruction_feedback:
            print(
                f'Could not find action id {action.global_game_action_id} in feedback: {instruction_feedback.keys()}'
            )
            return None
        target_action_sequence.append(
            ActionStep(
                action.preceding_observation, action.preceding_state,
                action.executed_action, action.resulting_configuration,
                SampledActionAnnotation(
                    instruction_feedback[action.global_game_action_id],
                    action.voxel_predictions, action.argmax_voxel)))

    # Get the new game ID
    new_id: str = f'{game_id}-{instruction_data.get_instruction_index()}'

    return Example(instruction=instruction_data.get_instruction(),
                   example_id=new_id,
                   target_action_sequence=target_action_sequence,
                   leader_actions=leader_actions,
                   num_first_turn_steps=initial_num_steps,
                   dataset_split=DatasetSplit.VALIDATION
                   if is_val else DatasetSplit.TRAIN,
                   expected_sets=expected_sets,
                   all_remaining_leader_actions=None)


def _parse_instruction_data(cursor, game_id: str,
                            instruction_data: RecordedRollout,
                            aligned_actions: List[FollowerAction],
                            turn_start_times: List[datetime],
                            leader_actions_in_turns: List[List[Action]],
                            feedback: Dict[str, ActionFeedback],
                            is_val: bool) -> Optional[Example]:
    sorted_aligned_actions: List[FollowerAction] = sorted(
        aligned_actions, key=lambda x: x.server_time)
    first_action: FollowerAction = sorted_aligned_actions[0]
    last_action: FollowerAction = sorted_aligned_actions[-1]

    start_turn_idx: int = _get_turn_id_for_action(cursor, game_id,
                                                  first_action.action_id,
                                                  turn_start_times)
    if not start_turn_idx % 2:
        # This actually may happen because the reboot turn switches are stored with the time the reboot is
        # *received*, not the time that it is actually executed on the server side. Which makes things a bit
        # messy. Instead, just subtract one to the turn index.
        start_turn_idx -= 1

    end_turn_idx: int = _get_turn_id_for_action(cursor, game_id,
                                                last_action.action_id,
                                                turn_start_times)
    if not end_turn_idx % 2:
        # Same as above.
        end_turn_idx -= 1

    relevant_turns: List[List[Action]] = list()
    for i in range(start_turn_idx, end_turn_idx):
        if not i % 2:
            if i >= len(leader_actions_in_turns):
                raise ValueError(
                    f'Leader actions for turn {i} not found; '
                    f'finding leader actions between turns {start_turn_idx} and {end_turn_idx} (game ID {game_id})'
                )
            relevant_turns.append(leader_actions_in_turns[i])

    num_steps_in_first_turn: int = _get_num_steps_remaining(
        cursor, game_id, first_action, start_turn_idx)

    if first_action.is_finish_instruction():
        expected_sets: List[Tuple[List[Card], List[Card]]] = list()
    else:
        if last_action.is_finish_instruction():
            # Get the one before it
            last_action_id: int = _find_last_movement_id_before_finish_instr(
                cursor, game_id, last_action, turn_start_times)
        else:
            last_action_id: int = _get_idx_from_action_id(
                last_action.action_id)

        expected_sets: List[Tuple[List[Card],
                                  List[Card]]] = _get_expected_sets(
                                      cursor, game_id,
                                      _get_idx_from_action_id(
                                          first_action.action_id),
                                      last_action_id)

    return _create_instruction_obj(game_id, instruction_data, relevant_turns,
                                   num_steps_in_first_turn, expected_sets,
                                   feedback, is_val)


def _get_aligned_actions_and_feedback(
    cursor, game_id: str
) -> List[Union[SingleFeedbackSignal, FollowerAction, Instruction]]:
    # Get all the follower actions executed on the leader client's side
    q = 'select actionId, leaderExecutedTime, serverTime from followerActions where gameId=?'
    t = (game_id, )
    cursor.execute(q, t)
    follower_actions = cursor.fetchall()

    if not follower_actions:
        return list()

    executed_actions: List[FollowerAction] = list()
    for action in follower_actions:
        executed_actions.append(
            FollowerAction(action[0], float(action[1]),
                           _convert_time(action[2])))

    # Get the instructions
    q = 'select instructionId, serverTime from instructions where gameId=?'
    t = (game_id, )
    cursor.execute(q, t)
    result = cursor.fetchall()
    instructions: List[Instruction] = list()
    for instruction in result:
        instructions.append(
            Instruction('instruction_%s' % instruction[0],
                        _convert_time(instruction[1])))

    # Get all the feedback signals and their timing information
    q = 'select actionId, clientFeedbackTime, type, serverTime from feedback where gameId=?'
    t = (game_id, )
    cursor.execute(q, t)
    feedbacks = cursor.fetchall()

    feedback_signals: List[SingleFeedbackSignal] = list()
    for feedback in feedbacks:
        # The times from the database are recorded in milliseconds (1/1000 th of a second). The delay is then
        # multiplied by 1000.
        original_time: float = float(feedback[1])
        corrected_time = original_time - FLAGS.delay_amount * 1000
        feedback_signals.append(
            SingleFeedbackSignal(feedback[0], corrected_time, original_time,
                                 feedback[2], _convert_time(feedback[3])))

    # Map actions to instructions
    return sorted(feedback_signals + executed_actions + instructions,
                  key=lambda x: x.server_time)


def _get_instruction_move_alignment(
        cursor, game_id: str) -> Dict[str, List[FollowerAction]]:
    server_sorted_actions = _get_aligned_actions_and_feedback(cursor, game_id)

    if not server_sorted_actions:
        return dict()

    instruction_to_action_map: Dict[str, List[FollowerAction]] = dict()
    instruction_queue = list()
    last_instr_id = ''
    for action in server_sorted_actions:
        if instruction_queue:
            last_instr_id = instruction_queue[0]
        if isinstance(action, Instruction):
            instruction_queue.append(action.action_id)
        elif isinstance(action, FollowerAction):
            if last_instr_id not in instruction_to_action_map:
                instruction_to_action_map[last_instr_id] = list()
            instruction_to_action_map[last_instr_id].append(action)

            if action.is_finish_instruction():
                instruction_queue = instruction_queue[1:]

                if not instruction_queue:
                    last_instr_idx = _get_idx_from_action_id(
                        action.action_id) + 1
                    last_instr_id = f'instruction_{last_instr_idx}'
        elif isinstance(action,
                        SingleFeedbackSignal) and action.polarity == 'Cancel':
            instruction_queue = list()
    return instruction_to_action_map


def _get_leader_actions_in_turns(cursor, game_id: str) -> List[List[Action]]:
    # Only need the MovementActions because full games won't be replayed (hence, don't need the instruction actions).
    q = 'select max(turnId) from turns where gameId=?'
    t = (game_id, )
    cursor.execute(q, t)
    max_turn_id: int = cursor.fetchone()[0]

    if not max_turn_id % 2:
        # Make the last turn odd so we can get all of the leader turns.
        max_turn_id += 1

    all_actions: List[List[Action]] = [list() for _ in range(max_turn_id)]

    for turn_id in range(0, max_turn_id):
        q = 'select character, action, serverTime from movements where gameId=? and turnId=?'
        t = (game_id, turn_id)
        cursor.execute(q, t)

        movements = cursor.fetchall()

        actions: List[Tuple[Action, str]] = list()
        if turn_id % 2:
            for character, action, server_time in movements:
                if character == LEADER:
                    raise ValueError(
                        f'Leader should not be moving during even turn ({turn_id}); game ID {game_id}'
                    )
        else:
            for character, action, server_time in movements:
                if character == FOLLOWER:
                    raise ValueError(
                        f'Follower should not be moving during even turn ({turn_id}); game ID {game_id}'
                    )
                actions.append((action, server_time))

        sorted_movements: List[Tuple[Action, str]] = sorted(
            actions, key=lambda x: _convert_time(x[1]))
        all_actions[turn_id] = [
            Action(action) for action, time in sorted_movements
        ]

    return all_actions


def _get_turn_start_times(cursor, game_id: str) -> List[datetime]:
    q = 'select turnId, serverTime from turns where gameId=? and type=? order by turnId asc'
    t = (game_id, 'begin')
    cursor.execute(q, t)

    results: List[Tuple[int, str]] = cursor.fetchall()

    times: List[datetime] = list()
    expected_turn_id: int = 0
    for list_idx, (turn_id, time_str) in enumerate(results):
        if turn_id != expected_turn_id:
            raise ValueError(
                f'Did not expect turn with index {list_idx} to have turn ID {turn_id}; expected {expected_turn_id}'
            )

        times.append(_convert_time(time_str))

        expected_turn_id += 1
    return times


def _convert_recorded_game_to_game(recorded_game: RecordedGame) -> Game:
    return Game(recorded_game.get_game_id(), recorded_game.get_environment())


def _parse_db_game_info(cursor, game_id: str,
                        feedback: Dict[str,
                                       Dict[str,
                                            ActionFeedback]], machine_id: str,
                        is_val: bool) -> Optional[Tuple[Game, List[Example]]]:
    # Static game info, including hexes and static props
    game_directory: str = os.path.join(FLAGS.agent_rollout_directory,
                                       FLAGS.unique_id, machine_id, game_id)
    game_filepath: str = os.path.join(game_directory, f'game_{game_id}.pkl')
    if not os.path.exists(game_filepath):
        print(
            f'Pickle file for game {game_id} did not exist (should be at {game_filepath})'
        )
        return None

    # Load the leader actions in each leader turn
    leader_actions_in_turns: List[List[Action]] = _get_leader_actions_in_turns(
        cursor, game_id)

    # The game data object contains the static game info, including props and terrains.
    with open(game_filepath, 'rb') as infile:
        game_data: RecordedGame = pickle.load(infile)

    # Get alignments between instruction IDs and aligned moves.
    instruction_move_alignment: Dict[
        str, List[FollowerAction]] = _get_instruction_move_alignment(
            cursor, game_id)

    # Get each turn's start time.
    turn_start_times: List[datetime] = _get_turn_start_times(cursor, game_id)

    # Create each instruction example
    instructions: List[Example] = list()

    instruction_filepaths: List[str] = get_executed_instruction_filepaths(
        game_directory)
    for filepath in instruction_filepaths:
        try:
            with open(filepath, 'rb') as infile:
                instruction_data: RecordedRollout = pickle.load(infile)
        except Exception as e:
            print(e)
            print(f'Could not open file {filepath}')
            continue

        instruction_id: str = f'instruction_{instruction_data.get_instruction_index()}'

        if instruction_id in instruction_move_alignment:
            instr: Optional[Example] = _parse_instruction_data(
                cursor, game_id, instruction_data,
                instruction_move_alignment[instruction_id], turn_start_times,
                leader_actions_in_turns, feedback[instruction_id], is_val)
            if instr is not None:
                instructions.append(instr)
        else:
            print('could not find alignment for instruction %s' %
                  instruction_id)

    game: Game = _convert_recorded_game_to_game(game_data)

    return game, instructions


def _extract_from_db(db_filepath: str,
                     feedback_data: Dict[str, Dict[str, Dict[str,
                                                             ActionFeedback]]],
                     machine_id: str) -> Tuple[List[Game], List[Example]]:
    if not os.path.exists(db_filepath):
        raise FileNotFoundError(
            f'Database must already exist; not found at {db_filepath}')

    games: List[Game] = list()
    all_examples: List[Example] = list()

    db = sqlite3.connect(db_filepath)
    c = db.cursor()

    game_ids: List[str] = sorted(get_game_ids_in_db(c))

    print(f'extracting {len(game_ids)} games from DB {db_filepath}')
    if FLAGS.model_suffix:
        print(f'(only considering games played by model {FLAGS.model_suffix})')

    r = random.Random(72)
    r.shuffle(game_ids)

    split_idx: int = round(VAL_PROP * len(game_ids))
    train_ids: List[str] = game_ids[split_idx:]
    val_ids: List[str] = game_ids[:split_idx]
    print(
        f'Splitting data into {len(val_ids)} validation games and {len(train_ids)} training games.'
    )
    assert len(val_ids) + len(train_ids) == len(game_ids)

    for game_id in tqdm(game_ids):
        if game_id in feedback_data:
            result = _parse_db_game_info(c, game_id, feedback_data[game_id],
                                         machine_id, game_id in val_ids)
            if result is not None:
                game, examples = result
                games.append(game)
                all_examples.extend(examples)

    return games, all_examples


def main(argv):
    feedback_data: Dict[str,
                        Dict[str,
                             Dict[str,
                                  ActionFeedback]]] = load_feedback_from_file(
                                      os.path.join(FLAGS.feedback_directory,
                                                   f'{FLAGS.unique_id}.tsv'))

    games: List[Game] = list()
    examples: List[Example] = list()

    rootdir: str = os.path.join(FLAGS.db_directory, FLAGS.unique_id)

    for db_name in os.listdir(rootdir):
        if not db_name.endswith('.db'):
            continue

        new_games, new_examples = _extract_from_db(
            os.path.join(rootdir, db_name), feedback_data,
            db_name.split('.')[0].replace('database', 'agent_rollouts'))
        games.extend(new_games)
        examples.extend(new_examples)

    subdirectory: str = os.path.join(FLAGS.save_directory, FLAGS.unique_id)

    if FLAGS.model_suffix:
        subdirectory = f'{subdirectory}_{FLAGS.model_suffix}'

    if not os.path.exists(subdirectory):
        os.mkdir(subdirectory)

    print(f'saving {len(games)} games...')
    game_path: str = os.path.join(subdirectory, GAME_STR)
    if not os.path.exists(game_path):
        os.mkdir(game_path)

    for game in tqdm(games):
        with open(os.path.join(game_path, f'{game.game_id}.pkl'),
                  'wb') as ofile:
            pickle.dump(game, ofile)

    print(f'saving {len(examples)} examples...')
    examples_path: str = os.path.join(subdirectory, EXAMPLES_STR)
    if not os.path.exists(examples_path):
        os.mkdir(examples_path)

    for example in tqdm(examples):
        with open(
                os.path.join(
                    examples_path,
                    f'{example.dataset_split.shorthand()}_{example.example_id}.pkl'
                ), 'wb') as ofile:
            pickle.dump(example, ofile)

    print('done saving!')


if __name__ == '__main__':
    app.run(main)
