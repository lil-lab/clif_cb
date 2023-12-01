"""Represents the follower's beliefs about the environment, given what it has observed in the past."""
from __future__ import annotations

import pickle

from dataclasses import dataclass

from environment.card import Card, CardSelection
from environment.player import Player
from environment.position import Position
from environment.rotation import Rotation

from typing import Dict, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from environment.state import State
    from typing import Set, Tuple

with open('../cb_vin_feedback/util/visibility_map.pkl', 'rb') as infile:
    VISIBILITY_MAP: Dict[Tuple[Position, Rotation],
                         Set[Position]] = pickle.load(infile)


def get_player_observed_positions(player: Player) -> Set[Position]:
    return VISIBILITY_MAP[(player.position, player.rotation)]


@dataclass
class Observation:
    """An observation available to the follower at a point in the game.
    
    Attributes:
        self.believed_leader
            The believed configuration of the leader agent (where it was last seen, if not currently 
            visible). May be None if leader has never been observed so far in a game.
        self.believed_cards
            The believed set of cards (may be out of date from true current set of cards).
        self.observation_ages
            The number of moves the agent has made since last observing each position on the board.
    """
    believed_leader: Optional[Player]
    believed_cards: List[Card]

    observation_ages: Dict[Position, int]

    maximum_memory_age: Optional[int] = None

    fully_observable: bool = False

    def __str__(self):
        card_rep: str = '\n\t'.join(
            [str(card) for card in self.believed_cards])
        if card_rep:
            card_rep = '\n\t' + card_rep
        else:
            card_rep = '(none)'
        return f'Observation with believed leader=[{self.believed_leader}]; believed cards: {card_rep}'

    def get_positions_in_memory(self) -> Set[Position]:
        if self.fully_observable:
            raise ValueError('No memory for positions if fully observed!')

        if self.maximum_memory_age is None:
            raise ValueError(
                'Maximum memory age was not set for this observation.')

        positions: Set[Position] = set()

        if self.maximum_memory_age < 0:
            for pos, age in self.observation_ages.items():
                if age >= 0:
                    positions.add(pos)
        else:
            for pos, age in self.observation_ages.items():
                if 0 <= age <= self.maximum_memory_age:
                    positions.add(pos)

        return positions


def update_observation(current_observation: Observation, current_state: State,
                       update_ages: bool) -> Observation:
    new_follower: Player = current_state.follower

    now_visible_positions: List[Position] = VISIBILITY_MAP[(
        new_follower.position, new_follower.rotation)]

    # Update the previous observation ages.
    previous_observation_ages = current_observation.observation_ages
    new_observation_ages: Dict[Position, int] = dict()

    for previous_observation, previous_age in previous_observation_ages.items(
    ):
        new_observation_ages[previous_observation] = previous_age
        if update_ages:
            new_observation_ages[previous_observation] += 1

    # Make sure all the currently-visible positions get an age of zero.
    for pos in now_visible_positions:
        new_observation_ages[pos] = 0

    # Leader: if it's now in view, update it, otherwise it will stay where it was.
    current_leader: Player = current_state.leader
    if current_leader.position not in now_visible_positions:
        current_leader = current_observation.believed_leader

    # Cards:

    # First, make sure everything in view is correct.
    actual_cards = current_state.cards
    actual_card_dict: Dict[Position, Card] = dict()
    for actual_card in actual_cards:
        actual_card_dict[actual_card.position] = actual_card

    new_card_beliefs = list()
    for observed_position in now_visible_positions:
        # If the position contains a now-visible card, add it.
        if observed_position in actual_card_dict:
            actual_card = actual_card_dict[observed_position]
            selection: CardSelection = actual_card.selection

            new_card = Card(actual_card.color, actual_card.count,
                            actual_card.shape, actual_card.selection,
                            actual_card.position)

            new_card_beliefs.append(new_card)

    # Then add all the cards that were previously observed but no longer observed.
    previously_observed_cards = current_observation.believed_cards

    for card_belief in previously_observed_cards:
        if card_belief.position not in now_visible_positions:
            new_card_beliefs.append(card_belief)

    return Observation(current_leader, new_card_beliefs, new_observation_ages,
                       current_observation.maximum_memory_age,
                       current_observation.fully_observable)


def create_first_observation(state: State, maximum_memory_age: int,
                             fully_observable: bool) -> Observation:
    initial_follower = state.follower

    # Figure out which hexes can be seen from this configuration
    visible_positions: List[Position] = VISIBILITY_MAP[(
        initial_follower.position, initial_follower.rotation)]

    # Construct a state delta with the dynamic information
    leader = state.leader
    if leader.position not in visible_positions:
        leader = None

    card_beliefs: List[Card] = list()
    for state_card in state.cards:
        if state_card.position in visible_positions:
            card_beliefs.append(
                Card(state_card.color, state_card.count, state_card.shape,
                     state_card.selection, state_card.position))

    # The initial observation history is an age of 0 (currently observed) for all visible positions.
    return Observation(leader, card_beliefs,
                       {pos: 0
                        for pos in visible_positions}, maximum_memory_age,
                       fully_observable)
