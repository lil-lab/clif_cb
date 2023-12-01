"""Utilities for interacting with wandb."""
import wandb


def initialize_wandb_with_name(project: str, name: str):
    return wandb.init(project=project, name=name, reinit=True, resume='allow')
