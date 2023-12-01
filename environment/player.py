"""Represents a player in CerealBar."""
from __future__ import annotations

from dataclasses import dataclass

from environment.position import Position
from environment.rotation import Rotation, rotation_from_v3

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from protobuf import CerealBarProto_pb2

_FOLLOWER_NAME: str = 'FOLLOWER'
_LEADER_NAME: str = 'LEADER'


@dataclass
class Player:
    """A CerealBar player.
    
    Attributes:
        self.is_follower
            Whether the player is the follower. If this is false, the player is the leader.
        self.position
            The player's current position.
        self.rotation
            The player's current rotation.
    """
    is_follower: bool
    position: Position
    rotation: Rotation

    def __eq__(self, other):
        if not isinstance(other, Player):
            return False

        return (self.is_follower == other.is_follower
                and self.position == other.position
                and self.rotation == other.rotation)

    def __str__(self):
        return f'follower={self.is_follower}; position={self.position}; rotation={self.rotation}'

    def __hash__(self):
        return (self.position.__hash__(), self.rotation.__hash__(),
                self.is_follower).__hash__()

    def add_to_buf(self, buf: CerealBarProto_pb2):
        if self.is_follower:
            save_buf = buf.followerinfo
            save_buf.pName = _FOLLOWER_NAME
        else:
            save_buf = buf.leaderinfo
            save_buf.pName = _LEADER_NAME

        self.position.set_buf(save_buf.coordinate)
        save_buf.rotV3 = self.rotation.to_v3()


def player_from_proto(player_info, is_follower: bool) -> Player:
    position: Position = Position(player_info.coordinate.hexX,
                                  player_info.coordinate.hexZ)
    rotation: Rotation = rotation_from_v3(player_info.rotV3)

    pname: str = player_info.pName.replace('(Clone)',
                                           '').upper().split('\t')[0]
    if pname in {'AGENT_HUMAN', 'LEADER'} and is_follower:
        raise ValueError(
            f'Caller specified player prop as follower, but had leader pName (prop: {player_info})'
        )
    elif (pname.startswith('AGENT_A')
          or pname == 'FOLLOWER') and not is_follower:
        raise ValueError(
            f'Caller specified player prop as leader, but had follower pName (prop: {player_info})'
        )

    return Player(is_follower, position, rotation)
