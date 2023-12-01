"""Defines the various types of static props in the environment."""
from __future__ import annotations

from enum import Enum


class PropType(Enum):
    TREE: str = 'TREE'
    HUT: str = 'HUT'
    WINDMILL: str = 'WINDMILL'
    TOWER: str = 'TOWER'
    PLANT: str = 'PLANT'
    LAMPPOST: str = 'LAMPPOST'
    TENT: str = 'TENT'
