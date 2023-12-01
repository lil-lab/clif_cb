"""Visualizes and browses instruction examples in CerealBar."""
from __future__ import annotations

from absl import flags

import copy
import csv
import numpy as np
import random
import mss
import mss.tools
import pyperclip
import os
import time

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from config.data_config import DataConfig, TokenizerConfig, FeedbackHeuristicsConfig
from config.evaluation import UnityStandaloneConfig
from config.rollout import GameConfig
from config.training_configs import SupervisedTargetConfig
from data.dataset_split import DatasetSplit
from data.loading import load_recorded_data
from environment.position import EDGE_WIDTH, Position
from environment.static_environment import StaticEnvironment
from simulation.server_socket import ServerSocket
from simulation.unity_game import UnityGame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data.example import Example
    from data.dataset import DatasetCollection, GamesCollection
    from typing import List

flags.DEFINE_string('dataset_id', None, 'Dataset id to run on.')
flags.DEFINE_string('file_prefix', None,
                    'The file name prefix for example list.')
flags.DEFINE_bool(
    'save_gifs', None,
    'Whether to save gifs for later. Automates running through game.')
flags.DEFINE_bool(
    'load_from_batchfile', None,
    'Loads from an MTurk batch file generated after getting annotations.')
FLAGS = flags.FLAGS

ALL_DIDS: List[str] = [
    '1_29', '1_30', '1_31', '2_1', '2_2', '2_3', '2_5', '2_7', '2_9', '2_12',
    '2_16'
]

IMG_WIDTH = 645  # 1300
IMG_HEIGHT = 475  # 935
PROGRESS_BAR = 20
TOP = 78
LEFT = 560
SCREENSHOT_REGION = (1115, 160, IMG_WIDTH, IMG_HEIGHT)
FONT = ImageFont.truetype("~/Library/Fonts/eHyperlegible-bold.ttf", 64)


def _add_progressbar(img: Image, count: int, max_size: int):
    img_width, img_height = img.size
    width_diff = int(img_width / max_size)
    overlay = Image.new('RGBA', img.size, (255, 255, 255) + (0, ))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(0, img_height - PROGRESS_BAR),
                    (count * width_diff, img_height)],
                   fill=(255, 255, 255) + (128, ),
                   outline="black")
    return Image.alpha_composite(img, overlay)


def _add_white_overlay(img: Image, opacity: int):
    overlay = Image.new('RGBA', img.size, (255, 255, 255) + (opacity, ))
    return Image.alpha_composite(img, overlay)


def _take_screenshot():
    # return pyautogui.screenshot(region=SCREENSHOT_REGION)

    with mss.mss() as sct:
        region = {
            'top': TOP,
            'left': LEFT,
            'width': IMG_WIDTH,
            'height': IMG_HEIGHT
        }
        img = sct.grab(region)
        return Image.frombytes('RGB', img.size, img.bgra, "raw",
                               "BGRX").convert('RGBA')


def _visualize_example(example: Example, game_server: ServerSocket,
                       games: GamesCollection, save_gifs: bool):
    environment: StaticEnvironment = games.games[
        example.get_game_id()].environment

    if save_gifs and os.path.exists(
            f'annotation/gifs/{example.example_id}.gif'):
        return

    st = time.time()

    game: UnityGame = UnityGame(
        environment, example.target_action_sequence[0].previous_state,
        example.instruction, example.num_first_turn_steps,
        GameConfig(allow_player_intersections=True), example.leader_actions,
        example.expected_sets, game_server)

    print(example.instruction)
    pyperclip.copy(f'{example.example_id}\t{example.instruction}')

    example.construct_supervised_step_examples(
        SupervisedTargetConfig(True, True, False, False),
        environment.get_obstacle_positions())

    example_copy = copy.deepcopy(example)
    example_copy.reannotate_feedback_with_heuristics(
        FeedbackHeuristicsConfig(fill_in_the_blank=True))

    screenshots = list()
    if save_gifs:
        """
        all_target_dist: np.ndarray = np.zeros((EDGE_WIDTH, EDGE_WIDTH))
        for step in example.target_action_sequence:
            all_target_dist[step.target_configuration.position.x][
                step.target_configuration.position.y] = 1
        game.send_map_probability_dist(all_target_dist, 2)

        time.sleep(0.5)

        still_img = pyautogui.screenshot(region=SCREENSHOT_REGION)
        still_img.save(f'annotation/stills/{example.example_id}.png')

        game: UnityGame = UnityGame(
            environment, example.target_action_sequence[0].previous_state,
            example.instruction, example.num_first_turn_steps,
            GameConfig(allow_player_intersections=True),
            example.leader_actions, example.expected_sets, game_server)
        """

        time.sleep(0.5)

        initial_screenshot = _take_screenshot()
        screenshots.append(
            _add_progressbar(initial_screenshot, 0,
                             len(example.target_action_sequence)))

    for i, step in enumerate(example.target_action_sequence):
        """
        final_target: Position = step.final_target.position
        target_dist: np.ndarray = np.zeros((EDGE_WIDTH, EDGE_WIDTH))
        target_dist[final_target.x][final_target.y] = 1
        game.send_map_probability_dist(target_dist)
        """

        pos = step.feedback_annotation.feedback.num_positive
        neg = step.feedback_annotation.feedback.num_negative
        rb_str = '(reboot)' if step.feedback_annotation.feedback.reboot else ''

        if step.feedback_annotation.sampled_goal_voxel:
            final_target: Position = step.feedback_annotation.sampled_goal_voxel.position
            target_dist: np.ndarray = np.zeros((EDGE_WIDTH, EDGE_WIDTH))
            target_dist[final_target.x][final_target.y] = 1
            game.send_map_probability_dist(target_dist)

        heuristic_step = example_copy.step_examples[
            i].action_annotation.feedback

        if not save_gifs:
            input(
                f'{i} {step.target_action} {pos - neg} = {pos} - {neg} {rb_str} '
                f'(reannotated: {heuristic_step.num_positive - heuristic_step.num_negative})'
            )

        game.execute_follower_action(step.target_action)
        time.sleep(0.1)

        if save_gifs:
            screenshot = _take_screenshot()

            screenshot = _add_progressbar(screenshot, i + 1,
                                          len(example.target_action_sequence))
            screenshots.append(screenshot)
    game.finish_all_leader_actions()

    if save_gifs:
        if time.time() - st > 60:
            print('Detected timeout! Closing.')
            game_server.close()
            exit()
        screenshots[0].save(f'annotation/gifs/{example.example_id}.gif',
                            save_all=True,
                            append_images=screenshots[1:] + [
                                _add_white_overlay(screenshots[-1], 85),
                                _add_white_overlay(screenshots[-1], 85)
                            ],
                            optimize=False,
                            duration=500,
                            loop=0)


def browse_data():
    """Loads validation data and displays examples, including computed targets.."""
    #    data: DatasetCollection = load_training_data(DataConfig(TokenizerConfig()),
    #                                                 debug=True,
    #                                                 val_only=True)

    filter_examples = list()
    mturk_ratings = dict()
    num_correct = 0
    num_unsure = 0
    num_incorrect = 0
    if FLAGS.load_from_batchfile:
        with open(f'{FLAGS.file_prefix}.csv') as infile:
            csv_reader = csv.reader(infile, delimiter=',', quotechar='"')
            headers = list()
            for i, line in enumerate(csv_reader):
                if i == 0:
                    headers = line[33:]
                else:
                    exid = line[28]

                    filter_examples.append(exid)
                    mturk_ratings[exid] = {
                        h: v
                        for h, v in zip(headers, line[33:])
                    }
                    if mturk_ratings[exid]['Answer.correctness'] == 'correct':
                        num_correct += 1
                    elif mturk_ratings[exid][
                            'Answer.correctness'] == 'incorrect':
                        num_incorrect += 1
                    elif mturk_ratings[exid]['Answer.correctness'] == 'unsure':
                        num_unsure += 1
            print(f'Loaded {len(filter_examples)} to display')
    elif FLAGS.file_prefix:
        for did in ALL_DIDS:
            with open(f'{FLAGS.file_prefix}.txt') as infile:
                filter_examples.extend(line.strip().replace('_', '-')
                                       for line in infile.readlines()
                                       if line.strip())

    data: DatasetCollection = load_recorded_data(
        DataConfig(TokenizerConfig()),
        ALL_DIDS,
        'game_recordings/', {did: True
                             for did in ALL_DIDS},
        limit_to_examples=set(filter_examples))

    standalone_config: UnityStandaloneConfig = UnityStandaloneConfig()

    game_server: ServerSocket = ServerSocket(standalone_config.ip_address,
                                             standalone_config.port)
    game_server.start_unity()

    examples: List[Example] = list()
    for did in ALL_DIDS:
        examples.extend(
            data.online_datasets[DatasetSplit.TRAIN][did].instruction_examples)
        examples.extend(data.online_datasets[DatasetSplit.VALIDATION]
                        [did].instruction_examples)

    # random.Random(72).shuffle(examples)

    if filter_examples:
        ex_dict = dict()
        for example in examples:
            ex_dict[example.example_id] = example

        reordered = list()

        for exid in filter_examples:
            reordered.append(ex_dict[exid])

        examples = reordered

    print(f'Displaying {len(examples)} examples')

    for example in examples:
        print(f'---------- {example.example_id} ----------')
        _visualize_example(example, game_server, data.games, FLAGS.save_gifs)
        if mturk_ratings:
            rating = mturk_ratings[example.example_id]
            print(f'\tCorrectness: {rating["Answer.correctness"]}')
            if rating["Answer.errors.extra_card"] == "true":
                print(f'\t\tSelected additional card')
            if rating["Answer.errors.missed_card"] == "true":
                print(f'\t\tMissed target card')
            if rating["Answer.errors.inefficient"] == "true":
                print(f'\t\tInefficient path')
            if rating["Answer.errors.wrong_order"] == "true":
                print(f'\t\tWrong order of tasks')
            if rating["Answer.errors.wrong_stop"] == "true":
                print(f'\t\tWrong stop position')
            if rating["Answer.errors.wrong_turn"] == "true":
                print(f'\t\tWrong turning direction')
            if rating["Answer.errors.other"] == "true":
                print(f'\t\tOther error')
            if rating["Answer.text_input"]:
                print(f'\tComment: {rating["Answer.text_input"]}')

        if not FLAGS.save_gifs:
            input('Press enter to show next example.')

    game_server.close()
