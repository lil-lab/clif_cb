"""Huts / houses in the environment."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from environment.position import Position
from environment.rotation import Rotation

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from protobuf import CerealBarProto_pb2

HUT_PROTO_STR: str = 'HUT'


class HutColor(str, Enum):
    """ Various possible hut colors. """
    YELLOW: str = 'YELLOW'
    RED: str = 'RED'
    GREEN: str = 'GREEN'
    BLUE: str = 'BLUE'
    BLACK: str = 'BLACK'

    def __str__(self):
        return self.value


@dataclass
class Hut:
    """A hut (house).
    
    Attributes:
        self.color: The color of the hut.
        self.position: Object position.
        self.rotation: Object rotation.
    """
    color: HutColor
    position: Position
    rotation: Rotation

    def add_to_buf(self, buf: CerealBarProto_pb2.MapInfo):
        prop_buf = buf.propinfo.add()
        prop_buf.pName = f'{HUT_PROTO_STR}_{self.color}\t{self.rotation}'
        self.position.set_buf(prop_buf.coordinate)
        prop_buf.rotV3 = self.rotation.to_v3()
