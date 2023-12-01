"""Interface for a python game."""
from __future__ import annotations

from typing import TYPE_CHECKING

from environment.action import Action
from environment.card import CardSelection
from simulation.game import Game
from config.rollout import GameConfig

if TYPE_CHECKING:
    from environment.card import Card
    from environment.state import State
    from environment.static_environment import StaticEnvironment
    from typing import List, Tuple, Optional


class PythonGame(Game):
    def __init__(self, static_game_info: StaticEnvironment,
                 initial_state: State, initial_instruction: Optional[str],
                 initial_num_moves: int, game_config: GameConfig,
                 leader_actions: Optional[List[List[Action]]],
                 expected_sets: Optional[List[Tuple[List[Card], List[Card]]]]):
        super(PythonGame,
              self).__init__(static_game_info, initial_state,
                             initial_instruction, initial_num_moves,
                             game_config, leader_actions, expected_sets)

    def _card_change_handler(self, changed_card: Card):
        pass

    def _new_set_handler(self, made_set: List[Card], new_cards: List[Card]):
        pass

    def _execute_follower(self, action: Action) -> None:
        self._update_follower_with_action(action)

    def _execute_leader(self, action: Action) -> None:
        self._update_leader_with_action(action)

    def send_command(self, command: str):
        pass

    def add_cards(self, new_cards: List[Card]):
        assert len(self._selected_cards) == 0
        assert len(self._current_state.cards) == 18
        assert len(new_cards) == 3

        self._log(f'Adding {len(new_cards)} more cards')
        self._current_state.cards.extend(new_cards)

        for new_card in new_cards:
            if new_card.selection == CardSelection.SELECTED:
                self._log('New card was selected; adding to selected set')
                self._selected_cards.append(new_card)
