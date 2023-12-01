"""Feedback signals in online games."""
from __future__ import annotations

from dataclasses import dataclass

from environment.player import Player
from inference.predicted_action_distribution import ActionPredictions
from inference.predicted_voxel import VoxelPredictions

from tqdm import tqdm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, List, Optional, Union


@dataclass
class ActionFeedback:
    """All feedback signals for a single sampled action."""
    num_positive: int
    num_negative: int
    reboot: bool

    weight: float = 1.

    def is_neutral(self) -> bool:
        return (self.num_positive == 0 and self.num_negative == 0
                or self.num_positive == self.num_negative) and not self.reboot

    def polarity(self) -> int:
        """Returns -1, 0, or 1 based on the feedback."""
        if self.is_neutral():
            return 0
        if self.reboot:
            return -1

        if self.num_positive > self.num_negative:
            return 1

        # Should only happen if num_neg > num_pos.
        return -1


def load_feedback_from_file(
        filepath: str) -> Dict[str, Dict[str, Dict[str, ActionFeedback]]]:
    feedback_data: Dict[str, Dict[str, Dict[str, ActionFeedback]]] = dict()

    print('loading feedback for each move...')
    with open(filepath) as infile:
        lines: List[str] = infile.readlines()[1:]
        with tqdm(total=len(lines)) as pbar:
            for line in lines:
                worker_id, game_id, move_id, instruction_id, num_pos, num_neg, canceled = line.split(
                    '\t')
                canceled = canceled.strip()

                if game_id not in feedback_data:
                    feedback_data[game_id] = dict()

                if instruction_id not in feedback_data[game_id]:
                    feedback_data[game_id][instruction_id] = dict()

                feedback_data[game_id][instruction_id][
                    move_id] = ActionFeedback(int(num_pos), int(num_neg),
                                              eval(canceled))
                pbar.update(1)
    return feedback_data


@dataclass
class SampledActionAnnotation:
    """Relevant information about a sampled action in-game from a model, including the distribution from which the 
    action was sampled and the feedback it got."""
    feedback: ActionFeedback
    probability_dist: Optional[Union[VoxelPredictions, ActionPredictions]]

    # The sampled voxel, used for VIN.
    sampled_goal_voxel: Player
