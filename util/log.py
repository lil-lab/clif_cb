"""Logging utils."""
from __future__ import annotations

import logging
import os


def setup(save_directory: str, name: str = 'meta.log'):
    """Sets up the logging module for a process by piping output to the console and supressing logs from other
    packages.

    :param save_directory: The director where the log file should be saved.
    :param name: The name of the logging file.
    """
    # Configure the logger

    # Need to remove existing handlers first
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    filename = os.path.join(save_directory, name)
    logging.basicConfig(
        format='%(asctime)s %(name)-6s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        level=logging.INFO)

    # Remove the console handler, because this will just send every child message.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Then add the filehandler for the root logger
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    logging.getLogger('').addHandler(fh)
    logging.info('Created a meta logger with path %s' % (filename))

    # Ignore some modules' logging messages
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('moviepy').setLevel(logging.WARNING)


def teardown_logger(name: str = ''):
    for filehandler in logging.getLogger(name).handlers:
        filehandler.close()
        logging.getLogger(name).removeHandler(filehandler)
