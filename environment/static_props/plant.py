"""Huts / houses in the environment."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from environment.position import Position

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from protobuf import CerealBarProto_pb2


class PlantType(Enum):
    """ Types of foliage. """
    BUSH: str = 'BUSH'  # Short green bush.
    YELLOW_BUSH: str = 'BUSH_YW'  # Short yellowish-gold bush.

    GRASS: str = 'GRASS'  # Tall green grass.

    RED_FLOWER: str = 'FLOW_RD'  # Bright pink/red flower with distinct petals.
    PURPLE_FLOWER: str = 'FLOW_PRP'  # Short flower with big green leaves and light purple petals inside.
    BLUE_FLOWER: str = 'FLOW_BL'  # Small, short flower with petals larger than the leaves. Petals are purple.

    def __str__(self):
        return self.value


@dataclass
class Plant:
    """A plant / foliage.
    
    Attributes:
        self.plant_type: The type of plant.
        self.position: Object position.
    """
    plant_type: PlantType
    position: Position

    def add_to_buf(self, buf: CerealBarProto_pb2.MapInfo):
        prop_buf = buf.propinfo.add()
        prop_buf.pName = str(self.plant_type)
        self.position.set_buf(prop_buf.coordinate)
        prop_buf.rotV3 = '0,0,0'
