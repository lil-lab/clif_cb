"""
Contains a state of the game board, including all information which could change from user actions (e.g., 
cards and user configurations).
"""
from __future__ import annotations

from dataclasses import dataclass

from environment.card import Card, load_cards_from_proto
from environment.player import Player, player_from_proto

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.position import Position
    from protobuf import CerealBarProto_pb2
    from typing import Dict, List, Set


@dataclass
class State:
    leader: Player
    follower: Player

    cards: List[Card]

    def __str__(self):
        card_rep: str = '\n\t'.join([str(card) for card in self.cards])
        if card_rep:
            card_rep = '\n\t' + card_rep
        else:
            card_rep = '(none)'
        return f'State with leader=[{self.leader}]; follower=[{self.follower}]; cards: {card_rep}'

    def get_card_locations(self) -> Set[Position]:
        return {card.position for card in self.cards}

    def add_to_buf(self, map_info: CerealBarProto_pb2):
        # Fill in Leader and Follower Separately
        self.leader.add_to_buf(map_info)
        self.follower.add_to_buf(map_info)

        # Fill in cards
        for c in self.cards:
            c.add_to_buf(map_info)


def state_from_proto(map_info: CerealBarProto_pb2) -> State:
    leader: Player = player_from_proto(map_info.leaderinfo, is_follower=False)
    follower: Player = player_from_proto(map_info.followerinfo,
                                         is_follower=True)

    return State(leader, follower, load_cards_from_proto(map_info.cards))


def get_cards_changed_along_states(visited_states: List[State]) -> List[Card]:
    reached_cards: List[Card] = list()

    original_cards: List[Card] = visited_states[0].cards

    unique_card_visitations: List[Position] = list()

    original_position: Position = visited_states[0].follower.position
    has_moved: bool = False

    previous_position: Position = original_position

    for i, visited_state in enumerate(visited_states[1:]):
        position: Position = visited_state.follower.position
        if position == original_position and not has_moved:
            continue
        has_moved = True

        if position != previous_position:
            if position in [
                    card.position for card in visited_states[i].cards
            ] and position in [card.position for card in original_cards]:
                unique_card_visitations.append(position)

            previous_position = position

    card_pos_visitations: Dict[Position, int] = {
        pos1: len([pos2 for pos2 in unique_card_visitations if pos1 == pos2])
        for pos1 in set(unique_card_visitations)
    }

    reached_card_positions: Set[Position] = {
        pos
        for pos, count in card_pos_visitations.items() if count % 2 == 1
    }

    for card in original_cards:
        if card.position in reached_card_positions:
            reached_cards.append(card)

    return reached_cards
