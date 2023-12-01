"""Contains the superclass for a game."""
from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod

from config.rollout import GameConfig
from environment.action import Action
from environment import sets
from environment.card import CardSelection
from environment.player import Player
from simulation import planner

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.card import Card
    from environment.position import Position
    from environment.state import State
    from environment.static_environment import StaticEnvironment
    from typing import List, Optional, Set, Tuple

FOLLOWER_MOVES_PER_TURN: int = 10
LEADER_MOVES_PER_TURN: int = 5


class Game(ABC):
    def __init__(self,
                 static_game_info: StaticEnvironment,
                 initial_state: State,
                 initial_instruction: Optional[str],
                 initial_num_moves: int,
                 config: GameConfig,
                 leader_actions: Optional[List[List[Action]]] = None,
                 expected_sets: Optional[List[Tuple[List[Card],
                                                    List[Card]]]] = None):
        """Abstract class for games."""
        self._config: GameConfig = config

        self._static_game_info: StaticEnvironment = static_game_info
        self._current_state: State = copy.deepcopy(initial_state)

        self._instruction_buffer: List[str] = list()
        if initial_instruction:
            self._instruction_buffer = [initial_instruction]

        self._num_moves_left: int = initial_num_moves

        self._is_leader_turn: bool = config.start_with_leader
        self._leader_actions: Optional[List[List[Action]]] = leader_actions
        self._current_leader_turn_idx: int = 0

        self._current_set_index: int = 0
        self._expected_sets: List[Tuple[
            List[Card], List[Card]]] = copy.deepcopy(expected_sets)
        self._valid_state: bool = True

        self._score: int = 0
        self._turn_id: int = 0
        self._num_completed_instructions: int = 0

        self._selected_cards: List[Card] = []
        for current_card in self._current_state.cards:
            if current_card.selection == CardSelection.SELECTED:
                self._selected_cards.append(current_card)

        self._random_generator: random.Random = random.Random(72)

    @abstractmethod
    def _card_change_handler(self, changed_card: Card):
        pass

    @abstractmethod
    def _new_set_handler(self, made_set: List[Card], new_cards: List[Card]):
        pass

    @abstractmethod
    def _execute_follower(self, action: Action) -> None:
        pass

    @abstractmethod
    def _execute_leader(self, action: Action) -> None:
        pass

    @abstractmethod
    def send_command(self, command: str) -> None:
        pass

    def _log(self, text: str):
        if self._config.log_fn:
            self._config.log_fn(text)

    def _execute_leader_actions(self) -> None:
        self._log(f'Executing leader turn {self._current_leader_turn_idx}')
        if self._current_leader_turn_idx < len(self._leader_actions):
            for action in self._leader_actions[self._current_leader_turn_idx]:
                # Stop executing leader actions.
                if not self._valid_state:
                    self._log('Resulted in an invalid state.')

                self.execute_leader_action(action)
            self._current_leader_turn_idx += 1
        else:
            self._log('No more leader actions to execute')

        self._log('Done with executing leader actions, so ending the turn')
        self.end_turn()

    def _check_for_new_set(self, prev_state: State):
        """ Checks whether a set was made after the provided state with the action taken."""
        if self._expected_sets is None:
            return

        new_set: List[Card] = sets.valid_set_was_made(
            prev_state.cards, self._current_state.cards)
        if not new_set:
            return

        self._log('Made a set!')

        # Check that this was the expected set
        new_cards: List[Card] = []
        if self._current_set_index >= len(self._expected_sets):
            self._log(
                f'Set was not expected; made set #{self._current_set_index}; expected only '
                f'{len(self._expected_sets)} sets.')
            was_expected_set = False
        else:
            expected_set: List[Card] = self._expected_sets[
                self._current_set_index][0]

            was_expected_set = sets.card_states_equal(
                expected_set, new_set, require_same_selection=False)

            if not was_expected_set:
                self._log('Set was not expected.')

                expected_str: str = '\n\t'.join([str(c) for c in expected_set])
                got_str: str = '\n\t'.join([str(c) for c in new_set])
                self._log(f'Expected set with cards:\n{expected_str}')
                self._log(f'Got set with cards:\n{got_str}')

            new_cards: List[Card] = self._expected_sets[
                self._current_set_index][1]

        if was_expected_set:
            self._current_state.cards.extend(new_cards)
            self._selected_cards = list()
            for card in self._current_state.cards:
                if card.selection == CardSelection.SELECTED:
                    self._selected_cards.append(card)

            self._current_set_index += 1
        else:
            if self._config.check_valid_state:
                self._valid_state = False

            if self._config.generate_new_cards:
                # Now, generate random new cards...
                new_cards: List[Card] = sets.generate_new_cards(
                    self._current_state.cards,
                    self._static_game_info.get_obstacle_positions() | {
                        self._current_state.leader.position,
                        self._current_state.follower.position
                    }, 3, self._random_generator)

                for new_card in new_cards:
                    self._current_state.cards.append(new_card)

        self._new_set_handler(new_set, new_cards)

    def _update_leader_with_action(self, action: Action):
        leader: Player = self._current_state.leader
        obstacles: Set[
            Position] = self._static_game_info.get_obstacle_positions()
        if not self._config.allow_player_intersections:
            obstacles |= {self._current_state.follower.position}

        new_position, new_rotation = planner.get_new_player_orientation(
            leader, action, obstacles)

        # Set the leader's state
        self._current_state.leader = Player(is_follower=False,
                                            position=new_position,
                                            rotation=new_rotation)

        if action in {Action.MB, Action.MF}:
            self._update_card_with_move(new_position)

    def _update_follower_with_action(self, action: Action):
        follower: Player = self._current_state.follower
        obstacles: Set[
            Position] = self._static_game_info.get_obstacle_positions()
        if not self._config.allow_player_intersections:
            obstacles |= {self._current_state.leader.position}

        new_position, new_rotation = planner.get_new_player_orientation(
            follower, action, obstacles)

        # Set the follower's state
        self._current_state.follower = Player(is_follower=True,
                                              position=new_position,
                                              rotation=new_rotation)

        if action in {Action.MB, Action.MF}:
            self._update_card_with_move(new_position)

    def _update_card_with_move(self, new_position: Position):
        # First, if it stepped on any cards, then select or unselect the card
        changed_card: bool = False
        for current_card in self._current_state.cards:
            if current_card.position == new_position:
                self._log(f'Stepped on card {current_card}')
                if current_card.selection == CardSelection.UNSELECTED:
                    current_card.selection = CardSelection.SELECTED
                    self._selected_cards.append(current_card)
                else:
                    if current_card not in self._selected_cards:
                        raise ValueError(
                            f'Card {current_card} is selected but not in selected cards set!'
                        )
                    current_card.selection = CardSelection.UNSELECTED
                    self._selected_cards.remove(current_card)

                self._card_change_handler(current_card)

                changed_card = True

        if changed_card:
            # Check whether a set was made, or update the selection of the selected cards.
            made_set: bool = sets.is_current_selection_valid(
                self._selected_cards) and len(self._selected_cards) == 3

            # If made a set, then remove the cards. Don't add in cards -- the superclass will handle it.
            if made_set:
                self._score += 1
                self._log('Made a new set: removing the relevant cards!')

                prev_len: int = len(self._current_state.cards)
                for selected_card in self._selected_cards:
                    self._current_state.cards.remove(selected_card)
                if len(self._current_state.cards) != prev_len - 3:
                    raise ValueError(
                        'Should have removed three cards from the current state delta!'
                    )

                self._selected_cards = []

    def get_current_state(self) -> State:
        return copy.deepcopy(self._current_state)

    def get_turn_index(self) -> int:
        return self._turn_id

    def get_score(self) -> int:
        return self._score

    def in_valid_state(self) -> bool:
        return self._valid_state

    def get_instruction_index(self) -> int:
        """Gets the current instruction index."""
        return self._num_completed_instructions

    def instruction_buffer_size(self) -> int:
        return len(self._instruction_buffer)

    def get_current_instruction(self) -> str:
        if self._instruction_buffer:
            return self._instruction_buffer[0]

    def set_turn(self, is_leader: bool):
        self._is_leader_turn = is_leader

    def is_leader_turn(self) -> bool:
        return self._is_leader_turn

    def reboot(self):
        num_incomplete_instructions = len(self._instruction_buffer)
        self._num_completed_instructions += num_incomplete_instructions
        self._instruction_buffer = list()

    def finish_all_leader_actions(self) -> None:
        """Finishes all specified leader actions at once."""
        self._is_leader_turn = True
        while self._current_leader_turn_idx < len(self._leader_actions):
            for action in self._leader_actions[self._current_leader_turn_idx]:
                self.execute_leader_action(action)
                if not self._valid_state:
                    return
            self._current_leader_turn_idx += 1

    def end_turn(self) -> None:
        """Ends the current player's turn."""
        self._turn_id += 1

        if self._is_leader_turn:
            self._log('Ending leader turn')
            # If it was the leader's turn, change to the follower's turn.
            self._is_leader_turn = False
            self._num_moves_left = FOLLOWER_MOVES_PER_TURN

            # But if there aren't any instructions left now, return to the leader's turn.
            if not self._instruction_buffer and self._leader_actions is not None and self._current_leader_turn_idx < \
                    len(self._leader_actions) and self._config.auto_end_turn:
                self._log(
                    'Automatically ending follower turn because there are no instructions left!'
                )
                self.end_turn()
        else:
            self._log('Ending follower turn')

            # Only decrease number of turns when the follower has finished.
            self._is_leader_turn = True
            self._num_moves_left = LEADER_MOVES_PER_TURN

            self._execute_leader_actions()

    def execute_follower_action(self, action: Action) -> State:
        assert isinstance(action, Action)
        if action != Action.STOP and len(
                self._current_state.cards) != sets.NUM_CARDS_ON_BOARD:
            raise ValueError(
                f'There are not {sets.NUM_CARDS_ON_BOARD} cards in the current state!'
            )

        if self._is_leader_turn and self._config.keep_track_of_turns:
            raise ValueError(
                'Can\'t execute agent action when it\'s the leader\'s turn.')

        if not self._instruction_buffer and self._leader_actions is not None:
            raise ValueError(
                f'Should not be able to execute actions when there are {len(self._instruction_buffer)} instructions '
                f'available')

        prev_state_delta: State = copy.deepcopy(self._current_state)

        # Internal execute MUST set the current state delta.
        self._log(f'Executing follower action {action}.')
        self._execute_follower(action)

        resulting_state: State = prev_state_delta

        if action == Action.STOP:
            self._instruction_buffer = self._instruction_buffer[1:]
            self._num_completed_instructions += 1
            self._log(
                f'{len(self._instruction_buffer)} instructions in buffer')
            if not self._instruction_buffer and self._config.auto_end_turn:
                self._log(
                    'Ending follower turn because there are no instructions left.'
                )
                self.end_turn()
        else:
            self._check_for_new_set(prev_state_delta)
            resulting_state = copy.deepcopy(self._current_state)

            self._num_moves_left -= 1

            if self._num_moves_left == 0 and self._valid_state and self._config.auto_end_turn:
                self._log('Ending turn because there are no steps left.')
                self.end_turn()

        del prev_state_delta
        return resulting_state

    def execute_leader_action(self, action: Action):
        self._log('Executing leader action ' + str(action))

        assert isinstance(action, Action)

        if len(self._current_state.cards) != sets.NUM_CARDS_ON_BOARD:
            raise ValueError(
                f'There are not {sets.NUM_CARDS_ON_BOARD} cards in the current state!'
            )

        if not self._is_leader_turn and self._config.keep_track_of_turns:
            raise ValueError(
                'Can\'t execute leader action when it\'s the agent\'s turn.')

        prev_state_delta: State = copy.deepcopy(self._current_state)

        self._execute_leader(action)

        self._check_for_new_set(prev_state_delta)

        self._num_moves_left -= 1

        if self._num_moves_left == 0 and self._leader_actions is None and self._config.auto_end_turn:
            self.end_turn()

        del prev_state_delta

    def add_instruction(self, instruction: str):
        """Adds an instruction to the queue."""
        self._instruction_buffer.append(instruction)
        self._log('Added instruction: ' + str(instruction))
        self._log('There are now ' + str(len(self._instruction_buffer)) +
                  ' instructions.')
