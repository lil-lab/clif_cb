"""Main script for training and evaluating simple CerealBar agents."""
from __future__ import annotations

import os

from absl import app, flags

from evaluation import evaluate
from learning import training
from data import browser

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', None, 'Mode to run (e.g., training, evaluation.)')
flags.DEFINE_string('config_filepath', None, 'Filepath for the configuration.')

flags.mark_flag_as_required('mode')


def _check_for_path():
    if not os.path.exists(FLAGS.config_filepath):
        raise FileNotFoundError(
            f'Did not find config filepath {FLAGS.config_filepath}')


def main(argv):
    mode: str = FLAGS.mode
    config_path: str = FLAGS.config_filepath

    if mode == 'TRAIN':
        _check_for_path()
        training.train(config_path)
    elif mode == 'EVAL':
        _check_for_path()
        evaluate.load_config_and_evaluate(config_path)
    elif mode == 'BROWSE':
        browser.browse_data()
    else:
        raise ValueError(f'Mode {mode} not supported.')


if __name__ == '__main__':
    app.run(main)
