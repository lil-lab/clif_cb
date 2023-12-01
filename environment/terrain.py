"""Terrains in the CerealBar environments."""
from __future__ import annotations

from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List


class Terrain(Enum):
    """ Terrains in the environment.

    Grass: green hexes.
    Path: brown/tan hexes.
    Water: blue hexes; lower than grass and path.
    Deep water: deeper water.
    Hill: white hexes; taller than grass and path.
    Short hill: a short mountain.
    """
    GRASS: str = 'GRASS'
    PATH: str = 'PATH'
    WATER: str = 'WATER'
    DEEP_WATER: str = 'DEEP_WATER'
    HILL: str = 'HILL'
    SHORT_HILL: str = 'SHORT_HILL'

    def __str__(self):
        return self.value


OBSTACLE_TERRAINS: List[Terrain] = [
    Terrain.WATER, Terrain.DEEP_WATER, Terrain.HILL, Terrain.SHORT_HILL
]
