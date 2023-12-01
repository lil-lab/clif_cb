"""
CerealBar game. This stores only the static environment information for the game board; the instructions are 
stored elsewhere.
"""
from __future__ import annotations

import json

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from environment.static_environment import StaticEnvironment


@dataclass_json
@dataclass
class Game:
    """A CerealBar game.
    
    Attributes:
        self.game_id
            The unique game ID.
        self.environment
            The static environment information for this game.
    """
    game_id: str
    environment: StaticEnvironment

    def to_json(self, indent: bool = False):
        if indent:
            return json.dumps(self.to_dict(), indent=2)
        return json.dumps(self.to_dict())
