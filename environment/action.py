"""All actions that can be taken by a player."""
from __future__ import annotations

from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List


class Action(str, Enum):
    """All actions either the leader or follower can take during the game.

        MF = move forward
        MB = move backward
        RR = rotate right
        RL = rotate left

        STOP = finish the current instruction

    """
    MF: str = 'MF'
    MB: str = 'MB'
    RR: str = 'RR'
    RL: str = 'RL'
    STOP: str = 'STOP'
    COPY: str = 'COPY'

    def __str__(self):
        return self.value

    def shorthand(self):
        if self == Action.MF:
            return '↑'
        if self == Action.MB:
            return '↓'
        if self == Action.RR:
            return '→'
        if self == Action.RL:
            return '←'
        if self == Action.STOP:
            return '🛑'


MOVEMENT_ACTIONS: List[Action] = [Action.MF, Action.MB, Action.RR, Action.RL]
