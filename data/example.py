"""Examples of instructions.

Example: a single example pairing an instruction with a target action sequence (including environment states).
LeaderTurn: actions taken by a leader during a turn.
ActionStep: a single step taken by a follower, including the environment state before and after taking the step.

"""
from __future__ import annotations

import copy

from dataclasses import dataclass

from config.data_config import FeedbackHeuristicsConfig
from config.training_configs import SupervisedTargetConfig
from data import feedback_heuristics, step_example
from data.dataset_split import DatasetSplit
from data.feedback import SampledActionAnnotation
from environment.action import Action
from environment.card import Card
from environment.observation import Observation
from environment.player import Player
from environment.state import State

from typing import List, Optional, Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from environment.position import Position
    from typing import Dict, Set


@dataclass
class ActionStep:
    """A single step taken by the agent.
    
    Attributes:
        self.previous_observation
            The observation the follower was making before the action was taken.
        self.previous_state
            The state of the board before the action was taken. This may include information about the game not 
            visible to the agent.
        self.target_action
            The action taken.
        self.target_configuration
            The target configuration of the agent; i.e., the rotation and position which should result from executing 
            the action in the current state.
    """
    previous_observation: Observation
    previous_state: State

    target_action: Action
    target_configuration: Player

    feedback_annotation: Optional[SampledActionAnnotation] = None


@dataclass
class Example:
    """An example of an instruction paired with a target action sequence.
    
    Attributes:
        self.instruction
            The instruction given by the leader.
        self.example_id
            The ID of the example, including the game ID and the instruction index.
        self.target_action_sequence
            The target action sequence from the original follower, including states as the agent moves and the action 
            taken.
        self.leader_actions
            The actions taken by the leader while the original follower was executing this instruction, divided into 
            turns.
        self.num_first_turn_steps
            The number of steps remaining in the follower's turn when they first had access to this instruction.
        self.dataset_split
            The split of the dataset this example is in (e.g., train, test).
        self.expected_sets
            The expected sets made by the agent, up until the end of the game.
        self.all_remaining_leader_actions: ALL remaining leader actions in the game, up until the end of the 
            originally-recorded game. Used only for cascaded evaluation.
    
    """
    instruction: str
    example_id: str
    target_action_sequence: List[ActionStep]
    leader_actions: List[List[Action]]
    num_first_turn_steps: int
    dataset_split: DatasetSplit
    expected_sets: List[Tuple[List[Card], List[Card]]]
    all_remaining_leader_actions: Optional[List[List[Action]]]

    step_examples: Optional[List[step_example.StepExample]] = None

    def get_game_id(self) -> str:
        return self.example_id.split('-')[0]

    def get_instruction_idx(self) -> int:
        return int(self.example_id.split('-')[1])

    def get_initial_observation(self) -> Observation:
        return self.target_action_sequence[0].previous_observation

    def get_initial_state(self) -> State:
        return self.target_action_sequence[0].previous_state

    def get_target_state(self) -> State:
        if self.target_action_sequence[-1].target_action != Action.STOP:
            raise ValueError('Final action in target sequence was not stop.')
        return self.target_action_sequence[-1].previous_state

    def set_maximum_memory_age(self, maximum_age: int):
        for step in self.target_action_sequence:
            if step.previous_observation.maximum_memory_age is None:
                step.previous_observation.maximum_memory_age = maximum_age

    def set_fully_observable(self):
        for step in self.target_action_sequence:
            step.previous_observation.fully_observable = True

    def get_previous_configurations_for_step_idx(self,
                                                 idx: int) -> List[Player]:
        return [
            s.previous_state.follower
            for s in self.target_action_sequence[:idx + 1]
        ]

    def construct_supervised_step_examples(
            self, target_config: SupervisedTargetConfig,
            obstacle_positions: Set[Position]
    ) -> List[step_example.StepExample]:
        """Creates one example per step in the target sequence and returns the created step examples."""
        if self.step_examples:
            return self.step_examples

        self.step_examples = list()

        previous_targets: List[Player] = list()
        previously_visited_cards: List[Card] = list()

        for i, step in enumerate(self.target_action_sequence):
            if target_config.directly_predict_actions:
                example: step_example.StepExample = step_example.create_action_supervised_step_example(
                    self.instruction, self.get_game_id(), step,
                    self.example_id, i,
                    self.get_previous_configurations_for_step_idx(i),
                    copy.deepcopy(previously_visited_cards))
            else:
                # Next target configurations.
                future_targets: List[Player] = [
                    s.target_configuration
                    for s in self.target_action_sequence[i + 1:]
                ]

                example: step_example.StepExample = step_example.create_supervised_step_example(
                    self.instruction,
                    self.get_game_id(),
                    step,
                    self.example_id,
                    i,
                    # Previous configurations the agent has visited so far (including the current one).
                    self.get_previous_configurations_for_step_idx(i),
                    future_targets,
                    copy.deepcopy(previous_targets),
                    target_config,
                    obstacle_positions)

                if example.target_action != Action.STOP:
                    # This is either neighbor (if no future configs are used) or heuristically-computed final target (if
                    # future configs are used).
                    previous_targets.append(example.final_target)

            self.step_examples.append(example)

            if step.target_action in {Action.MF, Action.MB}:
                resulting_state: State = self.target_action_sequence[
                    i + 1].previous_state
                previous_state: State = step.previous_state

                resulting_position: Position = resulting_state.follower.position
                previous_cards: Dict[Position, Card] = {
                    card.position: card
                    for card in previous_state.cards
                }
                if resulting_position in previous_cards:
                    previously_visited_cards.append(
                        previous_cards[resulting_position])

        return self.step_examples

    def construct_feedback_step_examples(
            self, source_name: str,
            use_ips: bool) -> List[step_example.StepExample]:
        """Creates one example per step in the target sequence and returns the created step examples."""
        if self.step_examples:
            return self.step_examples

        self.step_examples = list()

        previous_targets: List[Player] = list()
        previously_visited_cards: List[Card] = list()

        for i, step in enumerate(self.target_action_sequence):
            # Model predictions and feedback

            self.step_examples.append(
                step_example.create_feedback_step_example(
                    self.instruction, self.get_game_id(), step,
                    self.example_id, i,
                    self.get_previous_configurations_for_step_idx(i),
                    copy.deepcopy(previous_targets),
                    copy.deepcopy(previously_visited_cards), use_ips))

            if step.target_action != Action.STOP:
                previous_targets.append(
                    step.feedback_annotation.sampled_goal_voxel)

            if step.target_action in {
                    Action.MF, Action.MB
            } and i + 1 < len(self.target_action_sequence):
                resulting_state: State = self.target_action_sequence[
                    i + 1].previous_state
                previous_state: State = step.previous_state

                resulting_position: Position = resulting_state.follower.position
                previous_cards: Dict[Position, Card] = {
                    card.position: card
                    for card in previous_state.cards
                }
                if resulting_position in previous_cards:
                    previously_visited_cards.append(
                        previous_cards[resulting_position])

        return self.step_examples

    def convert_step_examples_to_feedback(
            self) -> List[step_example.StepExample]:
        new_step_examples: List[step_example.StepExample] = list()

        for step in self.step_examples:
            new_step_examples.append(step.convert_to_feedback_example())

        self.step_examples = new_step_examples
        return self.step_examples

    def reannotate_feedback_with_heuristics(
            self, config: FeedbackHeuristicsConfig
    ) -> List[step_example.StepExample]:
        has_neg: bool = False
        has_pos: bool = False
        any_neutral: bool = False

        for step in self.step_examples:
            if step.is_converted_feedback_example:
                # Don't actually modify anything here; just return the current examples. Don't need to reannotate
                # this at all.
                return self.step_examples

            polarity: int = step.action_annotation.feedback.polarity()
            if polarity < 0:
                has_neg = True
            elif polarity > 0:
                has_pos = True
            elif polarity == 0:
                any_neutral = True

        if not any_neutral or not has_neg and not has_pos:
            # All neutral anyway; just return the sequence.
            # Or, return the sequence if none of the annotations are neutral.
            return self.step_examples

        if config.fill_in_the_blank:
            new_step_examples: List[
                step_example.
                StepExample] = feedback_heuristics.fill_in_the_blank(
                    self.step_examples, has_pos, has_neg)
        elif config.new_fitb:
            new_step_examples: List[
                step_example.StepExample] = feedback_heuristics.adjusted_fitb(
                    self.step_examples, has_pos, has_neg)
        elif config.same_targets:
            new_step_examples: List[
                step_example.StepExample] = feedback_heuristics.same_targets(
                    self.step_examples)
        elif config.coach:
            new_step_examples: List[
                step_example.StepExample] = feedback_heuristics.coach(
                    self.step_examples, config.coach_decay_rate,
                    config.coach_horizon)
        else:
            raise NotImplementedError(
                f'Feedback heuristic setting not supported yet: {config}')

        self.step_examples = new_step_examples
        return self.step_examples


def get_examples_for_games(
        examples: List[Example]) -> Dict[str, List[Example]]:
    # Associates each example with its game ID, then sorts examples for each game in order of original execution.
    unsorted_game_examples: Dict[str, List[Example]] = dict()
    for example in examples:
        gid: str = example.get_game_id()
        if gid not in unsorted_game_examples:
            unsorted_game_examples[gid] = list()
        unsorted_game_examples[gid].append(example)

    return {
        gid: sorted(game_examples, key=lambda x: x.get_instruction_idx())
        for gid, game_examples in unsorted_game_examples.items()
    }
