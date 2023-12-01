"""Metrics for position prediction and instruction following."""
from __future__ import annotations

from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Set


class Metric(Enum):
    # Voxel prediction
    VOXEL_LOSS: str = 'VOXEL_LOSS'
    VOXEL_ACCURACY: str = 'VOXEL_ACCURACY'
    POSITION_ACCURACY_SUM: str = 'POSITION_ACCURACY_SUM'
    POSITION_ACCURACY_VOXEL: str = 'POSITION_ACCURACY_VOXEL'
    STOP_PRECISION: str = 'STOP_PRECISION'
    STOP_RECALL: str = 'STOP_RECALL'
    ENTROPY: str = 'ENTROPY'
    PROP_ARGMAX_SUBSEQUENT: str = 'PROP_ARGMAX_SUBSEQUENT'
    PROP_PRED_OBSTACLE: str = 'PROP_PRED_OBSTACLE'
    NEIGHBOR_ACTION_ACC: str = 'NEIGHBOR_ACTION_ACC'

    # Action prediction
    ACTION_LOSS: str = 'ACTION_LOSS'
    ACTION_ACCURACY: str = 'ACTION_ACCURACY'

    # Feedback training
    EXPECTED_FEEDBACK: str = 'EXPECTED_FEEDBACK'
    EXPECTED_FEEDBACK_POS: str = 'EXPECTED_FEEDBACK_POS'
    EXPECTED_FEEDBACK_NEG: str = 'EXPECTED_FEEDBACK_NEG'
    EXPECTED_ROT_INVARIANT_FEEDBACK: str = 'EXPECTED_ROT_INVARIANT_FEEDBACK'
    PROP_GOOD_ACTION: str = 'PROP_GOOD_ACTION'
    PROP_NEG_FB_SWITCH: str = 'PROP_NEG_FB_SWITCH'
    PROP_POS_FB_SAME: str = 'PROP_POS_FB_SAME'

    ACTION_EF: str = 'ACTION_EF'
    NEG_ACTION_EF: str = 'NEG_ACTION_EF'
    POS_ACTION_EF: str = 'POS_ACTION_EF'

    IPS: str = 'IPS'
    NUM_PASSING_IPS: str = 'NUM_PASSING_IPS'

    # Rollouts
    SEQUENCE_LENGTH: str = 'SEQUENCE_LENGTH'
    PATH_REDUNDANCY: str = 'PATH_REDUNDANCY'
    PROP_SHIFTING_TARGETS: str = 'PROP_SHIFTING_TARGETS'
    CARD_ACCURACY: str = 'CARD_ACCURACY'
    SUCCESS_STOP_DISTANCE: str = 'SUCCESS_STOP_DISTANCE'
    EXACT_ENVIRONMENT_ACCURACY: str = 'EXACT_ENVIRONMENT_ACCURACY'
    RELAXED_ENVIRONMENT_ACCURACY: str = 'RELAXED_ENVIRONMENT_ACCURACY'
    EXACT_SEQUENCE_ACCURACY: str = 'EXACT_SEQUENCE_ACCURACY'
    PROP_NO_STOP: str = 'PROP_NO_STOP'
    STOP_DISTANCE: str = 'STOP_DISTANCE'

    def __str__(self):
        return self.value.lower()


# Metrics which are proportions, and should be multiplied by 100 to display correctly.
PROP_METRICS: Set[Metric] = {
    Metric.VOXEL_ACCURACY, Metric.POSITION_ACCURACY_VOXEL,
    Metric.POSITION_ACCURACY_SUM, Metric.STOP_PRECISION, Metric.STOP_RECALL,
    Metric.PROP_ARGMAX_SUBSEQUENT, Metric.PROP_PRED_OBSTACLE,
    Metric.NEIGHBOR_ACTION_ACC, Metric.CARD_ACCURACY,
    Metric.EXACT_ENVIRONMENT_ACCURACY, Metric.RELAXED_ENVIRONMENT_ACCURACY,
    Metric.EXACT_SEQUENCE_ACCURACY, Metric.PROP_SHIFTING_TARGETS,
    Metric.ACTION_ACCURACY, Metric.PROP_GOOD_ACTION
}


class InstructionFollowingErrorType(Enum):
    """
    Types of errors an instruction-following agent is likely to make, given supervised examples where correct 
    behavior is known.
    """

    # If card state and agent final position is correct.
    CORRECT_CARDS_AND_POS: str = 'CORRECT_CARDS_AND_POS'

    # If card state is correct, but agent final position is not.
    CORRECT_CARDS_WRONG_POS: str = 'CORRECT_CARDS_WRONG_POS'

    # If at least one card is wrong, but the card grabbed in its place shares properties with the one missed.
    WRONG_CARDS_SHARE_PROPERTIES: str = 'WRONG_CARDS_SHARE_PROPERTIES'

    # If at least one card is wrong, and the card grabbed in its place does NOT share properties with the one missed.
    WRONG_CARDS_DO_NOT_SHARE_PROPERTIES: str = 'WRONG_CARDS_DO_NOT_SHARE_PROPERTIES'

    # If at least one of the target cards is missed, but no extra cards were selected.
    MISSES_TARGETS: str = 'MISSES_TARGETS'

    # If at least one additional card is selected in addition to correct targets, and these card(s) share properties
    # with the correct target.
    ADDITIONAL_CARDS_SHARE_PROPERTIES: str = 'ADDITIONAL_CARDS_SHARE_PROPERTIES'

    # If at least one additional card is selected in addition to correct targets, but these card(s) do not share
    # properties with the correct target.
    ADDITIONAL_CARDS_DO_NOT_SHARE_PROPERTIES: str = 'ADDITIONAL_CARDS_DO_NOT_SHARE_PROPERTIES'

    # If the agent stops immediately, but the target sequence requires the agent to move at least one step (e.g.,
    # rotation). Supersedes missing targets.
    STOPS_IMMEDIATELY: str = 'STOPS_IMMEDIATELY'

    def __str__(self):
        return self.value.lower()
