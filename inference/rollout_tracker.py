"""Structure for keeping track of a rollout for a specific instruction."""
from __future__ import annotations

import copy

from data.step_example import StepExample
from environment.action import Action
from environment.observation import update_observation

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from environment.card import Card
    from environment.observation import Observation
    from environment.player import Player
    from environment.position import Position
    from environment.state import State
    from simulation.game import Game
    from typing import Dict, List, Optional


class RolloutTracker:
    def __init__(self, instruction: str, game_id: str, example_id: str,
                 initial_observation: Observation, initial_state: State,
                 game: Game, copy_allowed: bool):
        self._instruction: str = instruction
        self._game_id: str = game_id
        self._example_id: str = f'{example_id}_rollout'

        self._current_observation: Observation = copy.deepcopy(
            initial_observation)
        self._current_state: State = copy.deepcopy(initial_state)
        self._game: Game = game

        self._visited_states: List[State] = [self._current_state]
        self._previously_visited_cards: List[Card] = list()
        self._executed_actions: List[Action] = list()
        self._targets: List[Player] = list()

        self._stopped: bool = False

        self._executed_all_leader_actions: bool = False

        self._stop_was_forced: bool = False

        self._copy_allowed: bool = copy_allowed

    def set_stop_forced(self, forced: bool):
        self._stop_was_forced = forced

    def stop_was_forced(self) -> bool:
        return self._stop_was_forced

    def get_game_id(self) -> str:
        return self._game_id

    def get_current_observation(self) -> Observation:
        return self._current_observation

    def get_executed_actions(self) -> List[Action]:
        return self._executed_actions

    def set_current_observation(self, observation: Observation):
        self._current_observation = observation

    def has_stopped(self) -> bool:
        return self._stopped

    def get_current_state(self) -> State:
        return self._current_state

    def can_copy(self) -> bool:
        # Whether the previous target can be copied.
        return self._copy_allowed and len(
            self._targets
        ) > 0 and self._targets[-1] != self._current_state.follower

    def get_current_step_example(self) -> StepExample:
        return StepExample(self._instruction,
                           self._game_id,
                           self._example_id,
                           len(self._executed_actions),
                           self._current_observation,
                           self._current_state,
                           [state.follower for state in self._visited_states],
                           self._targets,
                           self._previously_visited_cards,
                           is_supervised_example=False,
                           can_copy=self.can_copy())

    def execute_action(self,
                       action: Action,
                       target_config: Optional[Player],
                       allow_no_config: bool = False):
        if (action == Action.STOP) != (target_config is
                                       None) and not allow_no_config:
            raise ValueError(
                f'If the provided action is STOP, a target should not be provided; if it is not STOP, '
                f'a target config must be provided. Got action {action} and config {target_config}.'
            )

        previous_state: State = self._visited_states[-1]

        self._game.execute_follower_action(action)

        self._current_state = self._game.get_current_state()
        self._visited_states.append(self._current_state)

        if target_config is not None:
            self._targets.append(target_config)

        self._current_observation = update_observation(
            copy.deepcopy(self._current_observation),
            self._current_state,
            update_ages=action != Action.STOP)

        if action == Action.STOP:
            self._stopped = True

        self._executed_actions.append(action)

        previous_cards: Dict[Position, Card] = {
            card.position: card
            for card in previous_state.cards
        }
        current_position: Position = self._current_state.follower.position
        if action in {Action.MF, Action.MB
                      } and current_position in previous_cards:
            self._previously_visited_cards.append(
                previous_cards[current_position])

#            if current_position not in {card.position for card in self._current_state.cards}:
#                print(self._example_id)

    def finish_all_leader_actions(self):
        self._game.finish_all_leader_actions()
        self._current_state = self._game.get_current_state()
        self._executed_all_leader_actions = True

    def get_visited_states(self) -> List[State]:
        if not self._stopped:
            raise ValueError(
                'Should not get visited states unless rollout is stopped.')
        return self._visited_states

    def get_targets(self) -> List[Player]:
        return self._targets

    def get_final_state(self) -> State:
        if not self._executed_all_leader_actions:
            raise ValueError(
                'Must have executed all leader actions before getting a final state.'
            )
        return self._current_state
