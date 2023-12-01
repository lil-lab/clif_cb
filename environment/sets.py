"""Utilities associated with card sets in CerealBar."""
from __future__ import annotations

import random

from environment import card
from environment.card import CardCount, CardSelection
from environment.position import EDGE_WIDTH, Position

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from environment.card import Card
    from environment.position import Position
    from typing import List, Set

NUM_CARDS_ON_BOARD: int = 21


def card_list_difference(cards1: List[Card], cards2: List[Card]) -> List[Card]:
    """Returns the cards that were in one set but not in another."""
    in_cards2: List[bool] = []
    for card1 in cards1:
        remained: bool = False
        for card2 in cards2:
            if card1 == card2:
                remained = True
                break
        in_cards2.append(not remained)
    return [test_card for i, test_card in enumerate(cards1) if in_cards2[i]]


def valid_set_was_made(prev_cards: List[Card],
                       current_cards: List[Card]) -> List[Card]:
    """Returns the set of cards that were cleared from prev_cards (i.e., don't exist in cards)."""
    # Returns the cards missing from prev_state
    if len(current_cards) == len(prev_cards) - 3:
        # A set was made
        # Find the cards that are missing
        detected_set: List[Card] = card_list_difference(
            prev_cards, current_cards)
        if not len(detected_set) == 3:
            raise ValueError(
                'Detected three fewer cards, but got the following cards different: '
                + '\n'.join(
                    [str(detected_card) for detected_card in detected_set]))
        return detected_set

    elif len(current_cards) == len(prev_cards):
        for card1, card2 in zip(sorted(current_cards), sorted(prev_cards)):
            if card1 != card2 and card1.selection != card2.selection:
                raise ValueError(
                    f'Got the same number of cards, but cards did not match: \n {card1} vs. {card2}'
                )
    else:
        raise ValueError(
            f'Not a 3-card difference between card sets: {len(current_cards)} vs. {len(prev_cards)}'
        )

    return list()


def card_states_equal(card_list_1: List[Card],
                      card_list_2: List[Card],
                      require_same_selection: bool = True) -> bool:
    """Compares lists of cards and returns whether they are the same."""
    if len(card_list_1) != len(card_list_2):
        return False
    for card1, card2 in zip(sorted(list(card_list_1)),
                            sorted(list(card_list_2))):
        if card1 != card2 or require_same_selection and card1.selection != card2.selection:
            return False
    return True


def valid_set_exists(cards: List[Card]) -> bool:
    """Returns whether there is a possible set in the list of cards."""
    if len(cards) != NUM_CARDS_ON_BOARD:
        return False
    onecards: List[Card] = [
        test_card for test_card in cards if test_card.count == CardCount.ONE
    ]
    twocards: List[Card] = [
        test_card for test_card in cards if test_card.count == CardCount.TWO
    ]
    threecards: List[Card] = [
        test_card for test_card in cards if test_card.count == CardCount.THREE
    ]

    for card1 in onecards:
        for card2 in twocards:
            for card3 in threecards:
                if (len({card1.shape, card2.shape, card3.shape}) == 3
                        and len({card1.color, card2.color, card3.color}) == 3):
                    return True
    return False


def generate_new_cards(current_cards: List[Card],
                       obstacle_positions: Set[Position], num_to_generate: int,
                       r: random.Random) -> List[Card]:
    """Generates new random cards given a set of current cards and obstacle positions."""
    new_generated_cards: List[Card] = list()

    while not valid_set_exists(current_cards + new_generated_cards):
        # Generate new cards.
        new_generated_cards = list()
        for _ in range(num_to_generate):
            new_generated_cards.append(card.generate_random_card_properties(r))

    # Place the cards
    updated_generated_cards: List[card.Card] = list()
    non_obstacle_positions: Set[Position] = set()
    for x in range(EDGE_WIDTH):
        for y in range(EDGE_WIDTH):
            if Position(x, y) not in obstacle_positions:
                non_obstacle_positions.add(Position(x, y))

    for new_card in new_generated_cards:
        # Possible locations are all places in the map except those with obstacles, and where other
        # cards are now.
        possible_locations: Set[Position] = (
            non_obstacle_positions -
            set([current_card.position for current_card in current_cards]))

        chosen_location: Position = r.choice(list(possible_locations))
        new_card.position = chosen_location

        # Add the card here now so that it won't place later cards on top of it (unlikely,
        # but possible)
        updated_generated_cards.append(new_card)
    return updated_generated_cards


def is_current_selection_valid(cards: List[card.Card]) -> bool:
    """Returns whether the current selection of cards is valid"""
    selected_cards = [
        c for c in cards if c.selection == CardSelection.SELECTED
    ]
    if len(selected_cards) > 3:
        return False

    sel_len: int = len(selected_cards)
    num_counts: int = len(set([c.count for c in selected_cards]))
    num_shapes: int = len(set([c.shape for c in selected_cards]))
    num_colors: int = len(set([c.color for c in selected_cards]))

    return num_counts == sel_len and num_shapes == sel_len and num_colors == sel_len
