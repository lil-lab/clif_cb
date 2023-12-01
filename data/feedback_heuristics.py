"""Heuristics for re-assigning neutral feedbacks in feedback-annotated data."""
from __future__ import annotations

import math

from data import step_example
from data.feedback import ActionFeedback
from environment import sets
from environment.action import Action

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.step_example import StepExample
    from environment.position import Position
    from typing import List, Set, Tuple

MAX_FB_DISTANCE: int = 8  # The median number of actions in the original game.


def get_fb_assigned(has_pos: bool, has_neg: bool) -> Tuple[bool, int]:
    # If all positive or all negative, this will be the value to set.
    mixed_sequence: bool = True
    feedback_to_set: int = 0
    if has_pos and not has_neg:
        mixed_sequence = False
        feedback_to_set = 1
    elif has_neg and not has_pos:
        mixed_sequence = False
        feedback_to_set = -1

    return mixed_sequence, feedback_to_set


def fill_in_the_blank(original_examples: List[step_example.StepExample],
                      has_pos: bool,
                      has_neg: bool) -> List[step_example.StepExample]:
    mixed_sequence, feedback_to_set = get_fb_assigned(has_pos, has_neg)

    new_step_examples: List[step_example.StepExample] = list()

    prev_nonzero_feedback: int = 0
    distance_from_previous = 0
    prev_action = None
    prev_reboot = False
    has_selected_negative_card = False

    for i, step in enumerate(original_examples):
        prev_card_pos: Set[Position] = {
            card.position
            for card in step.state.cards
        }
        prev_agent_pos: Position = step.state.follower.position
        current_agent_pos: Position = step.sampled_resulting_configuration.position

        stepped_on_card = current_agent_pos != prev_agent_pos and current_agent_pos in prev_card_pos

        resulted_in_invalid_set = stepped_on_card and i < len(original_examples) - 1 and \
                                  sets.is_current_selection_valid(step.state.cards) and not \
                sets.is_current_selection_valid(original_examples[i + 1].state.cards)

        if step.action_annotation.feedback.is_neutral():
            distance_from_previous += 1

            distance = 0
            next_nonzero_feedback = 0
            is_in_neutral_suffix: bool = True
            for future_step in original_examples[i:]:
                if not future_step.action_annotation.feedback.is_neutral():
                    # Found a nonzero future feedback. Grab its polarity.
                    is_in_neutral_suffix = False
                    next_nonzero_feedback = future_step.action_annotation.feedback.polarity(
                    )
                    break
                distance += 1
            if mixed_sequence:
                if is_in_neutral_suffix:
                    # Smear from the past: no more future feedback to grab.
                    #                    print(
                    #                        f'{step.example_id}-{i}\tpast\t{distance_from_previous}\t{prev_nonzero_feedback}'
                    #                        f'\t{prev_action}\t{prev_reboot}\t{has_selected_negative_card}\t{resulted_in_invalid_set}'
                    #                        f'\t{stepped_on_card}\t{step.sampled_action}')
                    feedback_to_set = prev_nonzero_feedback
                else:
                    # Smear from the future.
                    #                    print(
                    #                        f'{step.example_id}-{i}\tfuture\t{distance}\t{next_nonzero_feedback}'
                    #                        f'\t{future_step.sampled_action}\t{future_step.action_annotation.feedback.reboot}'
                    #                        f'\t{has_selected_negative_card}\t{resulted_in_invalid_set}\t{stepped_on_card}'
                    #                        f'\t{step.sampled_action}')
                    feedback_to_set = next_nonzero_feedback
#            else:
#                print(
#                    f'{step.example_id}-{i}\t{"past" if is_in_neutral_suffix else "future"}'
#                    f'\t{distance_from_previous if is_in_neutral_suffix else distance}\t{feedback_to_set}'
#                    f'\t{prev_action if is_in_neutral_suffix else future_step.sampled_action}\t(no origin reboot)'
#                    f'\t{has_selected_negative_card}\t{resulted_in_invalid_set}\t{stepped_on_card}'
#                    f'\t{step.sampled_action}')

            if feedback_to_set == 0:
                raise ValueError('Feedback to set should not be zero!')

            new_feedback: ActionFeedback = ActionFeedback(
                1 if feedback_to_set > 0 else 0,
                1 if feedback_to_set < 0 else 0, False)
            assert new_feedback.polarity() == feedback_to_set

            step.reset_feedback(new_feedback)
            new_step_examples.append(step)
        else:
            new_step_examples.append(step)
            prev_nonzero_feedback = step.action_annotation.feedback.polarity()
            distance_from_previous = 0
            prev_action = step.sampled_action
            prev_reboot = step.action_annotation.feedback.reboot
            feedback_to_set = prev_nonzero_feedback


#            print(
#                f'{step.example_id}-{i}\traw\t0\t{prev_nonzero_feedback}\t{prev_action}\t{prev_reboot}'
#                f'\t{has_selected_negative_card}\t{resulted_in_invalid_set}\t{stepped_on_card}\t{step.sampled_action}'
#            )

        if stepped_on_card and feedback_to_set < 0:
            has_selected_negative_card = True

    return new_step_examples


def adjusted_fitb(original_examples: List[StepExample], has_pos: bool,
                  has_neg: bool) -> List[StepExample]:
    perm_neutral_fb: bool = False

    new_step_examples: List[step_example.StepExample] = list()

    for i, step in enumerate(original_examples):
        if perm_neutral_fb:
            step.reset_feedback(ActionFeedback(0, 0, False))
            new_step_examples.append(step)
            continue

        prev_card_pos: Set[Position] = {
            card.position
            for card in step.state.cards
        }
        prev_agent_pos: Position = step.state.follower.position
        current_agent_pos: Position = step.sampled_resulting_configuration.position

        stepped_on_card = current_agent_pos != prev_agent_pos and current_agent_pos in prev_card_pos

        if step.action_annotation.feedback.is_neutral():
            feedback_to_set: int = 0
            for future_step in original_examples[i:min(len(original_examples
                                                           ), i +
                                                       MAX_FB_DISTANCE)]:
                future_feedback: ActionFeedback = future_step.action_annotation.feedback
                if future_step.sampled_action == Action.STOP and future_feedback.polarity(
                ) < 0:
                    # Don't allow negative STOP feedback to propagate backwards
                    break

                if not future_step.action_annotation.feedback.is_neutral():
                    # Found a nonzero future feedback. Grab its polarity.
                    feedback_to_set = future_step.action_annotation.feedback.polarity(
                    )
                    break

        else:
            feedback_to_set = step.action_annotation.feedback.polarity()
            if feedback_to_set > 0 and stepped_on_card and i < len(original_examples) - 1 and \
                    sets.is_current_selection_valid(step.state.cards) and not \
                    sets.is_current_selection_valid(original_examples[i + 1].state.cards):
                # Ignore positive feedback if invalid set was selected
                feedback_to_set = 0

        new_feedback: ActionFeedback = ActionFeedback(
            1 if feedback_to_set > 0 else 0, 1 if feedback_to_set < 0 else 0,
            False)

        assert new_feedback.polarity() == feedback_to_set

        step.reset_feedback(new_feedback)
        new_step_examples.append(step)

        if stepped_on_card and feedback_to_set < 0:
            # Everything past here may be junk
            perm_neutral_fb = True

    return new_step_examples


def same_targets(
    original_examples: List[step_example.StepExample]
) -> List[step_example.StepExample]:

    new_step_examples: List[step_example.StepExample] = list()
    for i, step in enumerate(original_examples):
        if step.action_annotation.feedback.is_neutral():
            # Find the next / previous step with this target and nonzero feedback
            # Backwards search:
            has_pos: bool = False
            has_neg: bool = False

            search_pos: int = i
            while (search_pos >= 0 and original_examples[search_pos].
                   action_annotation.sampled_goal_voxel
                   == step.action_annotation.sampled_goal_voxel):
                tmp_step: ActionFeedback = original_examples[
                    search_pos].action_annotation.feedback
                if tmp_step.polarity() > 0:
                    has_pos = True
                elif tmp_step.polarity() < 0:
                    has_neg = True

                search_pos -= 1

            # Forwards search:
            search_pos: int = i
            while (search_pos < len(original_examples)
                   and original_examples[search_pos].action_annotation.
                   sampled_goal_voxel
                   == step.action_annotation.sampled_goal_voxel):
                tmp_step: ActionFeedback = original_examples[
                    search_pos].action_annotation.feedback
                if tmp_step.polarity() > 0:
                    has_pos = True
                elif tmp_step.polarity() < 0:
                    has_neg = True

                search_pos += 1

            if has_pos or has_neg:
                if has_pos:
                    new_feedback: ActionFeedback = ActionFeedback(1, 0, False)
                elif has_neg:
                    new_feedback: ActionFeedback = ActionFeedback(0, 1, False)
                else:
                    # Has both.
                    raise ValueError(
                        'Got both pos and neg feedback for this target!')

                step.reset_feedback(new_feedback)

            else:
                new_step_examples.append(step)

        else:
            new_step_examples.append(step)

    return new_step_examples


def _get_coach_weight(distance: int, decay_rate: float, horizon: int) -> float:
    if distance > horizon:
        return 0.
    else:
        # Exponential decay.
        return math.pow(decay_rate, distance)


def coach(original_examples: List[step_example.StepExample], decay_rate: float,
          horizon: int) -> List[step_example.StepExample]:

    new_step_examples: List[step_example.StepExample] = list()
    for i, step in enumerate(original_examples):
        if step.action_annotation.feedback.is_neutral():
            distance: int = 0

            has_pos: bool = False
            has_neg: bool = False

            for j in range(i + 1, len(original_examples)):
                distance += 1
                if not original_examples[
                        j].action_annotation.feedback.is_neutral():
                    polarity: int = original_examples[
                        j].action_annotation.feedback.polarity()

                    if polarity > 0:
                        has_pos = True
                    else:
                        has_neg = True
                    break

            if has_pos or has_neg:
                weight: float = _get_coach_weight(distance, decay_rate,
                                                  horizon)

                if weight:
                    if has_pos:
                        new_feedback: ActionFeedback = ActionFeedback(
                            1, 0, False, weight)
                    elif has_neg:
                        new_feedback: ActionFeedback = ActionFeedback(
                            0, 1, False, weight)
                    else:
                        # Has both.
                        raise ValueError(
                            'Got both pos and neg feedback for this target!')

                    step.reset_feedback(new_feedback)

            new_step_examples.append(step)
        else:
            new_step_examples.append(step)

    return new_step_examples
