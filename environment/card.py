"""Represents a card in the game."""
from __future__ import annotations

import random

from dataclasses import dataclass
from enum import Enum

from environment.position import Position
from protobuf import CerealBarProto_pb2

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Tuple


class CardColor(str, Enum):
    """ Unique card colors. """
    ORANGE: str = 'ORANGE'
    BLACK: str = 'BLACK'
    BLUE: str = 'BLUE'
    YELLOW: str = 'YELLOW'
    PINK: str = 'PINK'
    RED: str = 'RED'
    GREEN: str = 'GREEN'

    def __str__(self):
        return self.value

    def shorthand(self):
        if self == CardColor.ORANGE:
            return '🟧'
        if self == CardColor.BLACK:
            return '⬛'
        if self == CardColor.BLUE:
            return '🟦'
        if self == CardColor.YELLOW:
            return '🟨'
        if self == CardColor.PINK:
            return '🟪'
        if self == CardColor.RED:
            return '🟥'
        if self == CardColor.GREEN:
            return '🟩'

    def friendly(self):
        return self.value.lower()


class CardCount(str, Enum):
    ONE: str = '1'
    TWO: str = '2'
    THREE: str = '3'

    def __str__(self):
        return self.value


class CardShape(str, Enum):
    TORUS: str = 'TORUS'
    PLUS: str = 'PLUS'
    STAR: str = 'STAR'
    CUBE: str = 'CUBE'
    HEART: str = 'HEART'
    DIAMOND: str = 'DIAMOND'
    TRIANGLE: str = 'TRIANGLE'

    def __str__(self):
        return self.value

    def shorthand(self):
        if self == CardShape.TORUS:
            return '○'
        if self == CardShape.PLUS:
            return '+'
        if self == CardShape.STAR:
            return '★'
        if self == CardShape.CUBE:
            return '■'
        if self == CardShape.HEART:
            return '♥'
        if self == CardShape.DIAMOND:
            return '◆'
        if self == CardShape.TRIANGLE:
            return '▲'

    def friendly(self):
        if self == CardShape.TORUS:
            return 'circle'
        if self == CardShape.CUBE:
            return 'square'
        if self == CardShape.DIAMOND:
            return 'line'
        else:
            return self.value.lower()


class CardSelection(str, Enum):
    UNSELECTED: str = 'UNSELECTED'
    SELECTED: str = 'SELECTED'

    def __str__(self):
        return self.value


@dataclass
class Card:
    """A CerealBar card.
    
    Attributes:
        self.color: The color of the card.
        self.count: The count of items on the card.
        self.shape: The types of shapes on the card.
        self.selection: Whether the card is currently selected.
        self.position: The position of the card on the board. May be optional if card is not placed on the board yet.
    """
    color: CardColor
    count: CardCount
    shape: CardShape
    selection: CardSelection
    position: Optional[Position]

    def __str__(self):
        return f'{self.count} {self.color} {self.shape} at {self.position} ({self.selection})'

    def __eq__(self, other) -> bool:
        """ For equality of cards, don't care whether selection is the same, but other properties should not change. """
        if not isinstance(other, Card):
            return False
        return (self.position == other.position and self.color == other.color
                and self.count == other.count and self.shape == other.shape)

    def __lt__(self, other) -> bool:
        if not isinstance(other, Card):
            raise ValueError(
                f'When ordering cards, expected element to have type Card, but was {type(other)}'
            )
        return self.position < other.position

    def fill_buf(self, c: CerealBarProto_pb2.Card):
        c.color = str(self.color)
        c.shape = str(self.shape)
        c.count = str(self.count.name)
        c.selected = self.selection == CardSelection.SELECTED
        self.position.set_buf(c.coordinate)

    def add_to_buf(self, buf: CerealBarProto_pb2):
        c_buf = buf.cards.add()
        self.fill_buf(c_buf)


def generate_random_card_properties(r: random.Random) -> Card:
    """Generates a card with random properties and unspecified position."""
    card_count, card_color, card_shape = r.choice(POSSIBLE_CARDS)

    # Position/rotation are dummy values -- don't place the card yet!
    return Card(card_color, card_count, card_shape, CardSelection.UNSELECTED,
                None)


def load_cards_from_proto(card_info) -> List[Card]:
    cards: List[Card] = list()
    for card_rep in card_info:
        if card_rep.count in {str(CardCount.ONE), 'ONE'}:
            card_count = CardCount.ONE
        elif card_rep.count in {str(CardCount.TWO), 'TWO'}:
            card_count = CardCount.TWO
        elif card_rep.count in {str(CardCount.THREE), 'THREE'}:
            card_count = CardCount.THREE
        else:
            raise ValueError(
                f'Card count from proto format was not recognized: {card_rep.count}'
            )

        cards.append(
            Card(
                CardColor(
                    card_rep.color.replace('(UnityEngine.Material)',
                                           '').replace('card_',
                                                       '').strip().upper()),
                card_count,
                CardShape(
                    card_rep.shape.replace('(UnityEngine.GameObject)',
                                           '').strip().upper()),
                CardSelection.SELECTED
                if card_rep.selected else CardSelection.UNSELECTED,
                Position(card_rep.coordinate.hexX, card_rep.coordinate.hexZ)))

    return cards


POSSIBLE_CARDS: List[Tuple[CardCount, CardColor, CardShape]] = []
for count in sorted([count for count in CardCount]):
    for color in sorted([color for color in CardColor]):
        for shape in sorted([shape for shape in CardShape]):
            POSSIBLE_CARDS.append((count, color, shape))
