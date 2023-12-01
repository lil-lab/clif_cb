"""
A step example maps an instruction, observation, and prefix of an action sequence to a target set of positions and 
a target action.
"""
from __future__ import annotations

from dataclasses import dataclass

from config.training_configs import SupervisedTargetConfig
from data.feedback import ActionFeedback, SampledActionAnnotation
from environment.action import Action
from environment.observation import Observation
from environment.player import Player
from environment.state import State
from simulation.planner import find_path_between_positions

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from data.example import ActionStep
    from environment.card import Card
    from environment.position import Position
    from typing import Set, Tuple

ORIGINAL_DATA_PREFIX: str = 'original_data_'


@dataclass
class StepExample:
    """A single step example, mapping an instruction and observation to target set of positions and observations."""
    instruction: str
    game_id: str

    example_id: str
    step_idx: int

    observation: Observation
    state: State
    previous_configurations: List[Player]
    previous_targets: Optional[List[Player]]
    previously_visited_cards: List[
        Card]  # A list of cards the agent has visited, not including card it starts on (
    # if any)

    is_supervised_example: bool

    # Only relevant for supervised examples.
    should_copy: Optional[bool] = None
    target_action: Optional[Action] = None
    next_target_configuration: Optional[Player] = None
    possible_target_configurations: Optional[Set[Player]] = None
    final_target: Optional[Player] = None
    unique_target_id: str = None

    # Only relevant for feedback examples.
    sampled_action: Optional[Action] = None
    sampled_resulting_configuration: Optional[Player] = None
    action_annotation: Optional[SampledActionAnnotation] = None
    is_converted_feedback_example: bool = False

    # Only relevant during rollouts.
    can_copy: Optional[bool] = None

    use_ips: Optional[bool] = False

    def get_possible_target_positions(self) -> Set[Position]:
        if not self.possible_target_configurations:
            raise ValueError(
                'Cannot get possible target positions when there are no target configurations.'
            )
        if not self.is_supervised_example:
            raise ValueError(
                'There are no targets if the example is not a supervised example.'
            )

        possible_pos: Set[Position] = set()
        for target in self.possible_target_configurations:
            possible_pos.add(target.position)
        return possible_pos

    def convert_to_feedback_example(self) -> StepExample:
        """
        Returns a step example having feedback annotations (rather than configuration target annotations). Only 
        should be used for data which already has ground-truth target configurations.
        """
        return StepExample(
            self.instruction,
            self.game_id,
            self.example_id,
            self.step_idx,
            self.observation,
            self.state,
            self.previous_configurations,
            self.previous_targets,
            self.previously_visited_cards,
            is_supervised_example=False,
            sampled_action=self.target_action,
            sampled_resulting_configuration=self.next_target_configuration,

            # Feedback is all positive; no original probability distributions; target voxel is the final target here.
            action_annotation=SampledActionAnnotation(
                ActionFeedback(1, 0, False), None, self.final_target),
            is_converted_feedback_example=True)

    def reset_feedback(self, new_feedback: ActionFeedback):
        new_annotation: SampledActionAnnotation = SampledActionAnnotation(
            new_feedback, self.action_annotation.probability_dist,
            self.action_annotation.sampled_goal_voxel)

        self.action_annotation = new_annotation


def get_future_configurations(
        current_state: State, current_observation: Observation,
        future_configurations: List[Player],
        restrict_targets_to_shortest_path: bool,
        obstacle_positions: Set[Position]
) -> Tuple[Set[Player], Optional[Player]]:
    # Whether the agent has changed its position
    has_moved: bool = False

    restrict_to_observable_positions: bool = False
    positions_in_memory: Set[Position] = set()
    if not current_observation.fully_observable:
        restrict_to_observable_positions = True
        positions_in_memory = current_observation.get_positions_in_memory()

    future_target_configurations: Set[Player] = set()
    final_target: Optional[Player] = None

    # Get the positions to avoid: obstacles and cards (except for the next card along the trajectory)
    next_target_card_pos: Optional[Position] = None

    card_positions: List[Position] = current_state.get_card_locations()

    for configuration in future_configurations:
        if configuration.position in card_positions and configuration.position != current_state.follower.position:
            next_target_card_pos = configuration.position
            break

    avoid_positions: List[Position] = list(obstacle_positions)
    for card_pos in card_positions:
        if card_pos not in {
                current_state.follower.position, next_target_card_pos
        }:
            # Only consider cards as obstacles if they are (1) not agent's starting position and (2) not the next
            # card target along the demonstration (if it exists)
            avoid_positions.append(card_pos)

    # This does not include the next target configuration.
    for i, configuration in enumerate(future_configurations):
        if restrict_targets_to_shortest_path:
            # Add 2 here: the first future configuration in this list will be 2 demonstration steps away from the
            # current configuration.
            demonstration_path_length: int = i + 2
            shortest_path: Optional[Tuple] = find_path_between_positions(
                avoid_positions, current_state.follower, configuration)

            if configuration.position in avoid_positions:
                raise ValueError(
                    f'Configuration position should not be an avoid position! {configuration.position}'
                )

            if not shortest_path:
                raise ValueError(
                    f'Could not find shortest path between {current_state.follower} and {configuration}!'
                )

            assert len(shortest_path[0]) == len(shortest_path[1])
            shortest_path_length: int = len(shortest_path[0])

            if demonstration_path_length > shortest_path_length:
                # Break here. Do not include the target because at this point the demonstration has definitely
                # diverted from the shortest path, which can only be explained by some additional pressure for
                # the human follower to not follow the shortest path (e.g., following the instruction exactly,
                # even if it's not efficient, or exploring the environment).
                break

        if restrict_to_observable_positions and configuration.position not in positions_in_memory:
            # Stop looking at future configurations once the path is out of memory of the agent.
            break

        final_target = configuration
        future_target_configurations.add(configuration)

        if configuration.position != current_state.follower.position:
            has_moved = True

        if configuration.position in card_positions and has_moved:
            # Don't continue into future steps; last goal should be a card if there are any in the future.
            # Use the current state here (instead of the future step's state) because if there is a card in this
            # position in the future but not now, the agent doesn't have access to it now.

            # If the agent hasn't moved, include these in the target.
            break

    return future_target_configurations, final_target


def create_supervised_step_example(
        instruction: str, game_id: str, step: ActionStep, example_id: str,
        step_idx: int, previous_configurations: List[Player],
        future_configurations: List[Player], previous_targets: List[Player],
        target_config: SupervisedTargetConfig,
        obstacle_positions: Set[Position],
        previously_visited_cards: List[Card]) -> StepExample:
    """Creates a step example.
    
    Args:
        instruction
            The instruction.
        game_id
            The game ID (used to get the static environment information).
        example_id
            The original example ID this step is associated with.
        step_idx
            The step idx of this step in the sequence of steps.
        previous_configurations
            The previous configurations the agent was in, including the current one.
        previous_targets
            Sampled (or gold) targets from the previous steps in the action sequence.
        step
            The step that is being used for the target.
        future_configurations
            The future (target) configurations up to the end of the sequence.
        target_config
            Configuration of how to construct targets.
        obstacle_positions
            Positions in the environment containing obstacles; to be used for shortest-path calculations.
    """
    current_observation: Observation = step.previous_observation
    current_state: State = step.previous_state

    plausible_next_configurations: Set[Player] = {step.target_configuration}
    final_target: Player = step.target_configuration

    if target_config.include_future_configs_in_target:
        already_reached_card_target: bool = False
        subsequent_card: bool = False
        for card in current_state.cards:
            if card.position == step.target_configuration.position:
                subsequent_card: bool = True
                break
        if subsequent_card and step.target_configuration.position != current_state.follower.position:
            # The subsequent action was stepping onto the next target card, so don't add any more future targets.
            already_reached_card_target = True

        if not already_reached_card_target:
            future_targets, final_target_obj = get_future_configurations(
                current_state, current_observation, future_configurations,
                target_config.restrict_targets_to_shortest_path,
                obstacle_positions)

            plausible_next_configurations |= future_targets
            if final_target_obj is not None:
                final_target = final_target_obj

    for configuration in plausible_next_configurations:
        if not configuration.is_follower:
            raise ValueError(
                'All possible next configurations MUST be followers.')

    if target_config.use_only_final_target:
        plausible_next_configurations = {final_target}

    should_copy: bool = False
    if target_config.allow_copy and previous_targets and previous_targets[
            -1] == final_target and step.target_action != Action.STOP:
        should_copy = True

    return StepExample(
        instruction,
        game_id,
        example_id,
        step_idx,
        current_observation,
        current_state,
        previous_configurations,
        previous_targets,
        previously_visited_cards,
        is_supervised_example=True,
        should_copy=should_copy,
        target_action=step.target_action,
        next_target_configuration=step.target_configuration,
        possible_target_configurations=plausible_next_configurations,
        final_target=final_target)


def create_action_supervised_step_example(
        instruction: str, game_id: str, step: ActionStep, example_id: str,
        step_idx: int, previous_configurations: List[Player],
        previously_visited_cards: List[Card]) -> StepExample:
    """Creates a simple action-supervised step example where there are no target configurations, only target actions.
    """
    current_observation: Observation = step.previous_observation
    current_state: State = step.previous_state

    return StepExample(instruction,
                       game_id,
                       example_id,
                       step_idx,
                       current_observation,
                       current_state,
                       previous_configurations,
                       previous_targets=None,
                       previously_visited_cards=previously_visited_cards,
                       is_supervised_example=True,
                       should_copy=False,
                       target_action=step.target_action,
                       next_target_configuration=None,
                       possible_target_configurations=None,
                       final_target=None)


def create_feedback_step_example(instruction: str, game_id: str,
                                 step: ActionStep, example_id: str,
                                 step_idx: int,
                                 previous_configurations: List[Player],
                                 previous_targets: List[Player],
                                 previously_visited_cards: List[Card],
                                 use_ips: bool) -> StepExample:
    """Creates a step example.
    
    Args:
        instruction
            The instruction.
        game_id
            The game ID (used to get the static environment information).
        example_id
            The original example ID this step is associated with.
        step_idx
            The step idx of this step in the sequence of steps.
        previous_configurations
            The previous configurations the agent was in, including the current one.
        step
            The step that is being used for the target.
        previous_targets:
            The previous targets from previous steps in the sequence.
    """
    current_observation: Observation = step.previous_observation
    current_state: State = step.previous_state

    return StepExample(
        instruction,
        game_id,
        example_id,
        step_idx,
        current_observation,
        current_state,
        previous_configurations,
        previous_targets,
        previously_visited_cards,
        is_supervised_example=False,
        sampled_action=step.target_action,
        sampled_resulting_configuration=step.target_configuration,
        action_annotation=step.feedback_annotation,
        use_ips=use_ips)
