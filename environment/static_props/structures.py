"""Structures in the environment (windmill, tower) which have no interesting properties, and may have rotations."""
from __future__ import annotations

from dataclasses import dataclass

from environment.position import Position
from environment.rotation import Rotation

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from protobuf import CerealBarProto_pb2

WINDMILL_PROTO_STR: str = 'WINDMILL'
TOWER_PROTO_STR: str = 'TOWER'
TENT_PROTO_STR: str = 'HOUSE_LVL1'
LAMPPOST_PROTO_STR: str = 'STREET_LAMP'


@dataclass
class Windmill:
    position: Position
    rotation: Rotation

    def add_to_buf(self, buf: CerealBarProto_pb2.MapInfo):
        prop_buf = buf.propinfo.add()
        prop_buf.pName = WINDMILL_PROTO_STR
        self.position.set_buf(prop_buf.coordinate)
        prop_buf.rotV3 = self.rotation.to_v3()


@dataclass
class Tower:
    position: Position
    rotation: Rotation

    def add_to_buf(self, buf: CerealBarProto_pb2.MapInfo):
        prop_buf = buf.propinfo.add()
        prop_buf.pName = TOWER_PROTO_STR
        self.position.set_buf(prop_buf.coordinate)
        prop_buf.rotV3 = self.rotation.to_v3()


@dataclass
class Tent:
    position: Position
    rotation: Rotation

    def add_to_buf(self, buf: CerealBarProto_pb2.MapInfo):
        prop_buf = buf.propinfo.add()
        prop_buf.pName = TENT_PROTO_STR
        self.position.set_buf(prop_buf.coordinate)
        prop_buf.rotV3 = self.rotation.to_v3()


@dataclass
class Lamppost:
    position: Position

    def add_to_buf(self, buf: CerealBarProto_pb2.MapInfo):
        prop_buf = buf.propinfo.add()
        prop_buf.pName = LAMPPOST_PROTO_STR
        self.position.set_buf(prop_buf.coordinate)
        prop_buf.rotV3 = '0,0,0'
