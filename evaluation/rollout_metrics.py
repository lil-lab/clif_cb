"""Evaluates a position predictor in rollouts with real games."""
from __future__ import annotations

import copy
import logging
import numpy as np
import torch

from datetime import datetime
from tqdm import tqdm

from config.rollout import RolloutConfig
from data.dataset import GamesCollection
from evaluation.metric import Metric, InstructionFollowingErrorType
from environment.position import compute_distance
from environment.sets import card_states_equal
from environment.state import get_cards_changed_along_states
from inference.rollout_tracker import RolloutTracker
from inference import rollout
from simulation.python_game import PythonGame
from util import util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.example import Example
    from environment.card import Card
    from environment.observation import Observation
    from environment.player import Player
    from environment.position import Position
    from environment.state import State
    from model.position_prediction import PositionPredictionModel
    from typing import Dict, List, Optional, Tuple


def _np_mean_list(l: List) -> float:
    return float(np.mean(np.array(l)))


def _compute_position_redundancy(states: List[State]) -> int:
    """Returns the maximum number of times the agent was in a particular position."""
    visits: Dict[Position, int] = dict()
    for state in states:
        position: Position = state.follower.position
        if position not in visits:
            visits[position] = 0
        visits[position] += 1

    return max(visits.values())


def _has_shifting_targets(targets: List[Player], states: List[State]) -> bool:
    for i, (target, state) in enumerate(zip(targets, states)):
        if i >= 2:
            if (target == targets[i - 2]
                    and state.follower == states[i - 2].follower
                    and target != targets[i - 1]
                    and state.follower != states[i - 1].follower):
                # Prev target and config is different, but 2 steps before is the same.
                return True
    return False


def _evaluate_card_accuracy(state: State, target_state: State) -> bool:
    return card_states_equal(state.cards, target_state.cards)


def cascaded_evaluation(model: PositionPredictionModel,
                        game_examples: List[Example], game_id: str,
                        games: GamesCollection,
                        rollout_config: RolloutConfig) -> Optional[float]:
    """Does cascaded evaluation for a game, returning the proportion of points scored.

    Arguments:
        model: the model to run inference with.
        game_examples: the examples in the game, in order of their original execution.
        game_id: the ID of the game
        games: collection of games
        rollout_config: configuration for the rollout, including game config and rollout parameters.

    """
    model.eval()

    # The proportion of points scored by the pair of players (according to how many points should have been scored.)
    score_props: List[float] = list()

    # Game config needs to check for valid state.
    rollout_config.game_config.check_valid_state = True

    #print(f'Evaluating for {game_id}')
    for i, start_example in enumerate(game_examples):
        num_expected_points: int = len(start_example.expected_sets)
        #print(f'starting with example {i}, expecting {num_expected_points} points')

        if num_expected_points > 0:
            game: PythonGame = PythonGame(
                games.games[game_id].environment,
                start_example.get_initial_state(), None,
                start_example.num_first_turn_steps, rollout_config.game_config,
                start_example.all_remaining_leader_actions,
                start_example.expected_sets)

            current_observation: Observation = start_example.get_initial_observation(
            )

            for j, current_example in enumerate(game_examples[i:]):
                # Create a new tracker
                #print(f'\ttrying to execute instruction {j}: {current_example.instruction}')
                game.add_instruction(current_example.instruction)
                tracker: RolloutTracker = RolloutTracker(
                    current_example.instruction, current_example.get_game_id(),
                    current_example.example_id, current_observation,
                    game.get_current_state(), game, model.uses_copy())

                # Run a rollout with this tracker
                rollout.single_rollout(model, tracker, games, rollout_config)

                expected_changed_cards: List[Card] = [
                    card for card in get_cards_changed_along_states([
                        step.previous_state
                        for step in current_example.target_action_sequence
                    ]) if card in tracker.get_visited_states()[0].cards
                ]

                actually_changed_cards: List[
                    Card] = get_cards_changed_along_states(
                        tracker.get_visited_states())

                #print(f'\t\tExpected {len(expected_changed_cards)} cards to change; actually changed {len(actually_changed_cards)} cards')
                changed_expected_cards: bool = len(
                    expected_changed_cards) == len(actually_changed_cards)
                if changed_expected_cards:
                    for card1, card2 in zip(sorted(expected_changed_cards),
                                            sorted(actually_changed_cards)):
                        if card1 != card2:
                            changed_expected_cards = False
                            break

                if not game.in_valid_state() or not changed_expected_cards:
                    #print(f'\t\tBreaking. Game in valid state? {game.in_valid_state()}; Changed expected cards? {changed_expected_cards}')
                    break

            score_props.append(game.get_score() / num_expected_points)

    return _np_mean_list(score_props) if score_props else None


def _card_sets_share_properties(test_cards: List[Card],
                                target_cards: List[Card]) -> bool:
    for additional_card in test_cards:
        shares_properties: bool = False
        for target_card in target_cards:
            if (target_card.count == additional_card.count
                    or target_card.shape == additional_card.shape
                    or target_card.color == additional_card.color):
                shares_properties = True
                break
        if not shares_properties:
            return False
    return True


def _get_error_type(completed_rollout: RolloutTracker, example: Example,
                    accurate_cards: bool,
                    distance: int) -> InstructionFollowingErrorType:
    if len(completed_rollout.get_executed_actions()) == 1 and len(
            example.target_action_sequence) > 1:
        # Stopped immediately but should not have.
        return InstructionFollowingErrorType.STOPS_IMMEDIATELY

    if accurate_cards:
        if distance == 0:
            # Correct cards and end position.
            return InstructionFollowingErrorType.CORRECT_CARDS_AND_POS

        # Correct cards but wrong end position.
        return InstructionFollowingErrorType.CORRECT_CARDS_WRONG_POS

    # Wrong cards.
    target_cards: List[Card] = get_cards_changed_along_states(
        [step.previous_state for step in example.target_action_sequence])
    visited_cards: List[Card] = get_cards_changed_along_states(
        completed_rollout.get_visited_states())

    additional_cards: List[Card] = [
        card for card in visited_cards if card not in target_cards
    ]
    missed_cards: List[Card] = [
        card for card in target_cards if card not in visited_cards
    ]

    all_additional_share_properties_with_target: bool = _card_sets_share_properties(
        additional_cards, target_cards)
    if not missed_cards:
        # Agent selected some additional cards -- check if they share properties with targets.
        if all_additional_share_properties_with_target:
            return InstructionFollowingErrorType.ADDITIONAL_CARDS_SHARE_PROPERTIES

        return InstructionFollowingErrorType.ADDITIONAL_CARDS_DO_NOT_SHARE_PROPERTIES

    if additional_cards:
        # Agent missed some cards, but selected other ones. Perhaps the agent just selected the wrong cards.
        if all_additional_share_properties_with_target:
            if _card_sets_share_properties(additional_cards, missed_cards):
                return InstructionFollowingErrorType.WRONG_CARDS_SHARE_PROPERTIES
            else:
                return InstructionFollowingErrorType.MISSES_TARGETS
        return InstructionFollowingErrorType.WRONG_CARDS_DO_NOT_SHARE_PROPERTIES

    return InstructionFollowingErrorType.MISSES_TARGETS


def evaluate_position_predictor(
    models: List[PositionPredictionModel],
    examples: List[Example],
    games: GamesCollection,
    config: RolloutConfig,
    batch_size: int,
    sample_actions: bool = False,
    show_progress: bool = False,
    compute_accuracy_metrics: bool = True,
    logfile_path: str = ''
) -> Tuple[Dict[Metric, float], Dict[InstructionFollowingErrorType, int]]:
    for model in models:
        model.eval()

    sequence_lengths: List[int] = list()
    path_redundancies: List[int] = list()
    shifting_targets: List[int] = list()

    card_accuracies: List[int] = list()
    stop_distances: List[int] = list()

    exact_env_accuracies: List[int] = list()
    relaxed_env_accuracies: List[int] = list()
    sequence_accuracies: List[int] = list()
    swsds: List[float] = list()

    # Whether the follower did not stop before the horizon expired.
    no_stops: List[int] = list()

    error_classes: Dict[InstructionFollowingErrorType, int] = {
        name: 0
        for name in InstructionFollowingErrorType
    }

    logfile = None
    if logfile_path:
        logfile = open(logfile_path, 'w')

    with torch.no_grad():
        chunks: List[List[Example]] = list(util.chunks(examples, batch_size))
        if show_progress:
            chunks = tqdm(chunks)

        for batch_idx, batch in enumerate(chunks):
            if not show_progress:
                logging.info(
                    f'{datetime.now()}\t{batch_idx + 1} / {len(chunks)}')

            rollouts: List[RolloutTracker] = rollout.batch_rollout(
                models, batch, games, config, sample_actions)

            for i, r in enumerate(rollouts):
                visited_states: List[State] = r.get_visited_states()
                sequence_lengths.append(len(visited_states))
                path_redundancies.append(
                    _compute_position_redundancy(visited_states))
                shifting_targets.append(
                    int(_has_shifting_targets(r.get_targets(),
                                              visited_states)))
                no_stops.append(int(r.stop_was_forced()))

                if compute_accuracy_metrics:
                    final_state: State = r.get_final_state()
                    target_state: State = batch[i].get_target_state()

                    accurate_cards: bool = _evaluate_card_accuracy(
                        final_state, target_state)

                    card_accuracies.append(int(accurate_cards))

                    distance: int = compute_distance(
                        final_state.follower.position,
                        target_state.follower.position)

                    stop_distances.append(distance)

                    exact_env_accuracies.append(
                        int(accurate_cards
                            and final_state.follower == target_state.follower))
                    relaxed_env_accuracies.append(
                        int(accurate_cards and final_state.follower.position
                            == target_state.follower.position))
                    swsds.append(int(accurate_cards) / (1 + distance))

                    sequence_accuracies.append(
                        int([
                            action.target_action
                            for action in batch[i].target_action_sequence
                        ] == r.get_executed_actions()))

                    error_type: InstructionFollowingErrorType = _get_error_type(
                        rollouts[i], batch[i], accurate_cards, distance)

                    if logfile:
                        logfile.write(
                            f'{batch[i].example_id}\t{accurate_cards}\t{distance}'
                            f'\t{int(accurate_cards) / (distance + 1)}\t{str(error_type).upper()}\n'
                        )

                    if error_type not in error_classes:
                        error_classes[error_type] = 0
                    error_classes[error_type] += 1

    results: Dict[Metric, float] = {
        Metric.SEQUENCE_LENGTH: _np_mean_list(sequence_lengths),
        Metric.PATH_REDUNDANCY: _np_mean_list(path_redundancies),
        Metric.PROP_SHIFTING_TARGETS: _np_mean_list(shifting_targets),
        Metric.PROP_NO_STOP: _np_mean_list(no_stops)
    }

    logging.info(f'Error classes:')
    for error_type, count in sorted(error_classes.items(),
                                    key=lambda x: str(x[0])):
        logging.info(f'\t{error_type}\t{(100. * count / len(examples)):.1f}%')

    if logfile:
        logfile.close()

    if compute_accuracy_metrics:
        results.update({
            Metric.CARD_ACCURACY:
            _np_mean_list(card_accuracies),
            Metric.SUCCESS_STOP_DISTANCE:
            _np_mean_list(swsds),
            Metric.EXACT_ENVIRONMENT_ACCURACY:
            _np_mean_list(exact_env_accuracies),
            Metric.STOP_DISTANCE:
            _np_mean_list(stop_distances),
            Metric.RELAXED_ENVIRONMENT_ACCURACY:
            _np_mean_list(relaxed_env_accuracies),
            Metric.EXACT_SEQUENCE_ACCURACY:
            _np_mean_list(sequence_accuracies)
        })

    return results, error_classes
