"""Metadata used for all experiments."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ExperimentMetadata:
    """Defines an experiment's metadata.
    
    Attributes:
        self.experiment_name: The unique name of the experiment to run.
        self.debug: Debug mode. If this is True, then only one example will be loaded from disk.
        self.save_rootdir: The location where experiments are saved.
    """
    experiment_name: str

    debug: bool = False

    save_rootdir: str = 'experiments/'

    def validate(self):
        if not os.path.exists(self.save_rootdir):
            raise ValueError('Save directory %s does not exist.' %
                             self.save_rootdir)

        if not self.experiment_name:
            raise ValueError('Experiment name must be set.')

    def get_experiment_directory(self) -> str:
        """Gets the full path to the experiment save directory."""
        return os.path.join(self.save_rootdir, self.experiment_name)
