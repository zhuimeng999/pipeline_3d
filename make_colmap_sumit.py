# -*- coding: UTF-8 -*-

import argparse
import os
from utils import InitLogging, LogThanExitIfFailed

SCENES = {
    'intermediate': ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train'],
    'advanced': ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']
}


def checkSceneExists(scene_dir: str, scene_names: list):
    for scene_name in scene_names:
        scene_path = os.path.join(scene_dir, scene_name)
        LogThanExitIfFailed(os.path.isdir(scene_path), 'scene ' + scene_path + ' is not exists')


if __name__ == '__main__':
    InitLogging()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--intermediate_dir', type=str, default=None, help='intermedia scene directory')
    parser.add_argument('--advanced_dir', type=str, default=None, help='advanced scene directory')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='dataset directory, specail this for intermediate and advanced set evalution at same time')

    options = parser.parse_args()

    if options.data_dir:
        if options.intermediate_dir is None:
            options.intermediate_dir = os.path.join(options.data_dir, 'intermediate')
        if options.advanced_dir is None:
            options.advanced_dir = os.path.join(options.data_dir, 'advanced')

    LogThanExitIfFailed((options.intermediate_dir is None) and (options.advanced_dir is None),
                        "you must at least special one of [intermediate_dir, advanced_dir, data_dir]")
    if options.intermediate_dir is not None:
        checkSceneExists(os.path.join(options.intermediate_dir, 'intermediate'), SCENES['intermediate'])
    if options.advanced_dir is not None:
        checkSceneExists(os.path.join(options.intermediate_dir, 'advanced'), SCENES['advanced'])
