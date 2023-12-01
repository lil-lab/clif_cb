"""Configurations for doing rollouts using a model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass
class RolloutConfig:
    """Configuration for rollouts
    
    Attributes:
        self.restrict_to_neighbors: If the prediction space should only include voxels reachable in a single action.
        self.game_config: The game config specifying how the game simulator should be used.
        self.max_num_steps: The maximum number of steps that can be taken in a rollout.
    """
    restrict_to_neighbors: bool

    game_config: GameConfig

    max_num_steps: int = 25

    use_sampling: bool = False

    ensemble_inference: bool = False

    ensemble_softmax_normalization: bool = True

    voting_ensemble: bool = False

    def validate(self):
        pass


@dataclass
class GameConfig:
    """A configuration for a game.

    Attributes:
        self.allow_player_intersections: Whether players can intersect. Should be true for pre-training data, 
        but not for online games.
        self.keep_track_of_turns: Whether to detect when an agent is moving out-of-turn and throw an exception if so.
        self.auto_end_turn: Whether to automatically end each player's turn (and execute leader actions automatically).
        self.check_valid_state: Whether to detect when the game is in an invalid state (e.g., unexpected set was made).
        self.generate_new_cards: Whether to generate new cards when an unexpected set is made.
        self.log_fn: The logging function.
    """
    allow_player_intersections: bool

    keep_track_of_turns: bool = True
    auto_end_turn: bool = True
    check_valid_state: bool = False
    generate_new_cards: bool = True
    log_fn: Optional[Callable[[str], Any]] = None

    start_with_leader: bool = False
