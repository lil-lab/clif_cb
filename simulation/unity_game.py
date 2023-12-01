"""Interface for the Unity game."""
from __future__ import annotations

import json
import numpy as np
from typing import TYPE_CHECKING

from environment.action import Action
from environment.card import CardSelection
from environment.position import EDGE_WIDTH
from environment import sets
from protobuf import CerealBarProto_pb2
from simulation.game import Game
from config.rollout import GameConfig
from simulation.server_socket import ServerSocket

if TYPE_CHECKING:
    from environment.card import Card
    from environment.state import State
    from environment.static_environment import StaticEnvironment
    from typing import List, Tuple, Optional

MAX_MESSAGE_LENGTH: int = 65536


def _np_float_to_trunc_float(float_val: np.float32):
    """Converts an NP float to Python float and reduces the precision."""
    rounded = round(float_val.item(), 2)
    if str(rounded) == '0.0':
        return 0
    return rounded


def _create_card_status_change(
        current_card: Card,
        selected_cards: List[Card]) -> CerealBarProto_pb2.CardStatusChange:

    card_selection_info = CerealBarProto_pb2.CardStatusChange()

    current_card.fill_buf(card_selection_info.card)

    card_selection_info.result = str(current_card.selection)
    card_selection_info.valid = sets.is_current_selection_valid(selected_cards)
    for selected_card in selected_cards:
        c = card_selection_info.selectedcards.add()
        selected_card.fill_buf(c)
    return card_selection_info


class UnityGame(Game):
    def __init__(self, static_game_info: StaticEnvironment,
                 initial_state: State, initial_instruction: Optional[str],
                 initial_num_moves: int, game_config: GameConfig,
                 leader_actions: Optional[List[List[Action]]],
                 expected_sets: Optional[List[Tuple[List[Card], List[Card]]]],
                 connection: ServerSocket):
        super(UnityGame,
              self).__init__(static_game_info, initial_state,
                             initial_instruction, initial_num_moves,
                             game_config, leader_actions, expected_sets)
        self._connection: ServerSocket = connection

        # Start a new game and set the environment state.
        self._connection.start_new_game(self._environment_protobuf())

        for c in self._current_state.cards:
            if c.selection == CardSelection.SELECTED:
                self._card_change_handler(c)

    def _environment_protobuf(self) -> CerealBarProto_pb2.MapInfo:
        map_info: CerealBarProto_pb2.MapInfo = CerealBarProto_pb2.MapInfo()

        self._static_game_info.add_to_buf(map_info)
        self._current_state.add_to_buf(map_info)

        return map_info

    def _card_change_handler(self, changed_card: Card):
        card_selection: CerealBarProto_pb2.CardStatusChange = _create_card_status_change(
            changed_card, self._selected_cards)
        self._connection.send_card_selection(card_selection)

    def _new_set_handler(self, made_set: List[Card], new_cards: List[Card]):
        score_set_card = CerealBarProto_pb2.ScoreSetCard()
        for c in new_cards:
            buf: CerealBarProto_pb2.Card = CerealBarProto_pb2.Card()
            c.fill_buf(buf)
            score_set_card.newcards.append(buf)
        for c in made_set:
            buf: CerealBarProto_pb2.Card = CerealBarProto_pb2.Card()
            c.fill_buf(buf)
            score_set_card.setcard.append(buf)

        score_set_card.newscore = self._score

        # set card removal animation setting
        score_set_card.animationsetting.ymovevalue = 10
        score_set_card.animationsetting.ymoveinterval = 0.5
        score_set_card.animationsetting.rotatevalue = -90
        score_set_card.animationsetting.rotateinterval = 0.4
        score_set_card.animationsetting.rotatetospininterval = 0.2
        score_set_card.animationsetting.spininterval = 0.2

        self._connection.send_card_replacement(score_set_card)

    def _execute_follower(self, action: Action) -> None:
        """ Executes the specified action for the follower in the environment.

        Inputs:
            action (str): The action that the follower should take.

        Returns:
            Whether it was still the Follower's turn when the action finished.

        Raises:
            ValueError if it's not the follower's turn.
        """
        if action != Action.STOP:
            self._connection.send_data('follower, ' + str(action))

        if action == Action.STOP:
            # The STOP action maps to finishing a command for the follower.
            self._connection.send_data('finishcommand')

        _ = self._connection.receive_data()

        self._update_follower_with_action(action)

    def _execute_leader(self, action: Action) -> None:
        """ Executes an action taken by the leader.

        Input:
            action (agent_actions.AgentAction): The action taken by the leader.

        Returns:
            Whether the action increased the score (collected a card).

        Raises:
            ValueError if it's not the leader's turn.
        """
        self._connection.send_data('leader, ' + str(action))
        self._connection.receive_data()

        self._update_leader_with_action(action)

    def send_command(self, command: str) -> None:
        """ Sends a command to the game to display it in the interface. """
        self._connection.send_data('grammer, ' + command)

        _ = self._connection.receive_data()

    def send_map_probability_dist(self,
                                  probabilities: np.ndarray,
                                  color: int = 0) -> bool:
        """Sends distributions over hexes to the Unity game for display."""
        probs = list()
        for x in range(EDGE_WIDTH):
            for y in range(EDGE_WIDTH):
                val = _np_float_to_trunc_float(probabilities[x, y])
                if val:
                    probs.append({'p': [x, y], 'v': val})

        str_val = json.dumps({'hexInfo': probs})

        # Don't send the message if it's too long.
        if len(str_val) > MAX_MESSAGE_LENGTH:
            return False

        distname: str = 'trajdist'
        if color == 1:
            distname: str = 'goaldist'
        elif color == 2:
            distname: str = 'obsdist'
        elif color == 3:
            distname: str = 'avoiddist'

        self._connection.send_data(f'{distname},' + str_val)
        _ = self._connection.receive_data()

        return True

    def end_turn(self) -> None:
        """Ends the current player's turn."""
        self._connection.send_data('turn')
        _ = self._connection.receive_data()

        return super(UnityGame, self).end_turn()
