"""Tracks training progress using patience and decides to stop when all countdowns have been degraded."""
from __future__ import annotations

import ray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.training_util_configs import PatienceSchedule
    from typing import Dict, List

PG_ACTION: str = 'prop_good_action'
PG_ACTION_POS: str = 'prop_good_pos_feedback'
PG_ACTION_NEG: str = 'prop_good_neg_feedback'
EF_NAME: str = 'combined_expected_feedback'
COMBINED_ACTION_EF: str = 'combined_action_ef'


class AccuracyTracker:
    def __init__(self, patience_config: PatienceSchedule):
        self._current_maxima: Dict[str, float] = None

        self._patience_config: PatienceSchedule = patience_config

        self._countdowns: Dict[str, int] = None
        self._patiences: Dict[str, int] = None

    def _initialize_countdowns(self):
        self._countdowns: Dict[str, int] = {
            name: self._patience_config.initial_patience
            for name in self._current_maxima
        }

        self._patiences: Dict[str, int] = {
            name: self._patience_config.initial_patience
            for name in self._current_maxima
        }

    def initialize_maxima(self, accuracies: Dict[str, float]):
        if self._current_maxima:
            raise ValueError('Maxima are already set!')
        self._current_maxima = accuracies
        self._initialize_countdowns()

    def initialized(self) -> bool:
        return self._current_maxima is not None

    def check_improvement(self, new_values: Dict[str, float]) -> List[str]:
        if not self.initialized():
            raise ValueError(
                'Must set initial values before checking for improvement.')

        if set(new_values.keys()) != set(self._current_maxima.keys()):
            new_keys: str = ', '.join(
                [str(key) for key in list(new_values.keys())])
            old_keys: str = ', '.join(
                [str(key) for key in list(self._current_maxima.keys())])
            raise ValueError(
                f'Got a different set of metrics from checking improvements than the initial evaluation.'
                f'Got keys: {new_keys}; expected keys {old_keys}')

        improved_metrics: List[str] = list()
        for metric_name, new_value in new_values.items():
            if self._countdowns[
                    metric_name] > 0 and new_value > self._current_maxima[
                        metric_name]:

                # Reset the maximum for this metric
                self._current_maxima[metric_name] = new_value

                # Update the countdown for this metric
                self._patiences[
                    metric_name] *= self._patience_config.patience_update_factor
                self._countdowns[metric_name] = self._patiences[metric_name]

                improved_metrics.append(metric_name)

            # Subtract one from the countdown
            self._countdowns[metric_name] -= 1

        if EF_NAME in new_values:
            # Reset this to take on any component countdowns too.
            component_countdowns = {
                countdown
                for name, countdown in self._countdowns.items()
                if 'expected_feedback' in name.lower()
            }
            max_countdown: int = max(component_countdowns)

            self._countdowns[EF_NAME] = max_countdown
        if PG_ACTION in new_values:
            component_countdowns = {
                countdown
                for name, countdown in self._countdowns.items()
                if 'good_action' in name.lower()
            }
            max_countdown: int = max(component_countdowns)
            self._countdowns[PG_ACTION] = max_countdown
        if COMBINED_ACTION_EF in new_values:
            component_countdowns = {
                countdown
                for name, countdown in self._countdowns.items()
                if 'action_ef' in name.lower()
            }
            max_countdown: int = max(component_countdowns)
            self._countdowns[COMBINED_ACTION_EF] = max_countdown

        return improved_metrics

    def has_improving_metrics(self) -> bool:
        for countdown in self._countdowns.values():
            if countdown > 0:
                return True
        return False

    def get_countdowns(self) -> Dict[str, int]:
        return self._countdowns


@ray.remote
class PatienceTracker:
    def __init__(self, patience_schedule: PatienceSchedule):
        self._alive: bool = True
        self._num_epochs: int = 0
        self._accuracy_tracker: AccuracyTracker = AccuracyTracker(
            patience_schedule)

    def done_training(self) -> bool:
        return not self._alive

    def get_num_epochs(self) -> int:
        return self._num_epochs

    def get_countdowns(self) -> Dict[str, int]:
        return self._accuracy_tracker.get_countdowns()

    def increment_epochs(self):
        self._num_epochs += 1

    def improved_metric_after_epoch(self,
                                    new_values: Dict[str, float]) -> List[str]:

        if not self._accuracy_tracker.initialized():
            self._accuracy_tracker.initialize_maxima(new_values)

            # All improved, so just return all values.
            improved_metrics: List[str] = list(new_values.keys())

        else:
            improved_metrics: List[
                str] = self._accuracy_tracker.check_improvement(new_values)

        self._alive = self._accuracy_tracker.has_improving_metrics()

        return improved_metrics
