# -*- coding: UTF-8 -*-

import argparse
import os
import logging
from utils import InitLogging, LogThanExitIfFailed
import subprocess

SCENES = {
    'intermediate': ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train'],
    'advanced': ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']
}


def checkSceneExists(scene_dir: str, scene_names: list):
    for scene_name in scene_names:
        scene_path = os.path.join(scene_dir, scene_name)
        LogThanExitIfFailed(os.path.isdir(scene_path), 'scene ' + scene_path + ' is not exists')


def reconstructionScene(image_dir: str, work_dir: str):
    if os.path.isdir(work_dir) is False:
        os.mkdir(work_dir)

    colmap_command_line = [
        'colmap', 'automatic_reconstructor',
        '--image_path', image_dir,
        '--workspace_path', work_dir
    ]

    subprocess.run(colmap_command_line, check=True)

if __name__ == '__main__':
    InitLogging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--intermediate_dir', type=str, default=None, help='intermedia scene directory')
    parser.add_argument('--advanced_dir', type=str, default=None, help='advanced scene directory')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='dataset directory, specail this for intermediate and advanced set evalution at same time')
    parser.add_argument('work_dir', type=str, help='workspace directory')

    options = parser.parse_args()

    if options.data_dir:
        if options.intermediate_dir is None:
            options.intermediate_dir = os.path.join(options.data_dir, 'intermediate')
        if options.advanced_dir is None:
            options.advanced_dir = os.path.join(options.data_dir, 'advanced')

    LogThanExitIfFailed((options.intermediate_dir is not None) or (options.advanced_dir is not None),
                        "you must at least special one of [intermediate_dir, advanced_dir, data_dir]")

    intermediate_work_dir = os.path.join(options.work_dir, 'intermediate')
    if options.intermediate_dir is not None:
        checkSceneExists(options.intermediate_dir, SCENES['intermediate'])
        if os.path.isdir(intermediate_work_dir) is False:
            os.mkdir(intermediate_work_dir)

    advanced_work_dir = os.path.join(options.work_dir, 'advanced')
    if options.advanced_dir is not None:
        checkSceneExists(options.advanced_dir, SCENES['advanced'])
        if os.path.isdir(advanced_work_dir) is False:
            os.mkdir(advanced_work_dir)

    if options.intermediate_dir is not None:
        for scene_name in SCENES['intermediate']:
            reconstructionScene(os.path.join(options.intermediate_dir, scene_name), os.path.join(intermediate_work_dir, scene_name))

    if options.advanced_dir is not None:
        for scene_name in SCENES['advanced']:
            reconstructionScene(os.path.join(options.advanced_dir, scene_name), os.path.join(advanced_work_dir, scene_name))

    logging.info('done')
