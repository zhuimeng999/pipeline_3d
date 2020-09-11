# -*- coding: UTF-8 -*-

import argparse
import os
import logging
from pipeline.utils import InitLogging, LogThanExitIfFailed
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
        'python', os.path.join(os.path.dirname(__file__), '../pipeline/pipeline_run.py'),
        image_dir, work_dir,
        '--sfm', 'openmvg',
        '--mvs', 'mvsnet',
        '--fuse', 'mvsnet',
        '--mvs_max_w', '640',
        '--mvs_max_h', '512',
        '--mvs_max_d', '128'
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) + ':' + env['PYTHONPATH']
    subprocess.run(colmap_command_line, check=True, env=env)


def run_colmap_converter(image_dir, result_dir, log_output, script_dir):
    conver_to_log_command_line = ['python', 'convert_to_logfile.py',
                                  os.path.join(result_dir, 'sparse/0/camera.bin'),
                                  log_output,
                                  image_dir,
                                  'COLMAP',
                                  'jpg']
    subprocess.run(conver_to_log_command_line, check=True,
                   cwd=os.path.join(script_dir, 'TanksAndTemples/python_toolbox'))


if __name__ == '__main__':
    InitLogging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--intermediate_dir', type=str, default=None, help='intermedia scene directory')
    parser.add_argument('--advanced_dir', type=str, default=None, help='advanced scene directory')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='dataset directory, specail this for intermediate and advanced set evalution at same time')
    parser.add_argument('--script_dir', type=str, default=None,
                        help='convert_to_logfile.py')
    parser.add_argument('work_dir', type=str, help='workspace directory')

    options = parser.parse_args()

    if options.data_dir:
        if options.intermediate_dir is None:
            options.intermediate_dir = os.path.join(options.data_dir, 'intermediate')
        if options.advanced_dir is None:
            options.advanced_dir = os.path.join(options.data_dir, 'advanced')
    if options.script_dir is None:
        options.script_dir = os.path.join(options.data_dir, '../../')
        LogThanExitIfFailed(os.path.isdir(options.script_dir),
                            "you must provide the TanksAndTemples eval script directory")
    # convert_colmap_spec = importlib.util.spec_from_file_location('convert_colmap', options.colmap_converter_path)
    # convert_colmap = importlib.util.module_from_spec(convert_colmap_spec)
    # convert_colmap_spec.loader.exec_module(convert_colmap)
    # fun = getattr(convert_colmap, 'convert_COLMAP_to_log')

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

    submit_dir = os.path.join(options.work_dir, 'submit')
    subprocess.run(['rm', '-rv', submit_dir], check=False)
    os.mkdir(submit_dir)

    if options.intermediate_dir is not None:
        for scene_name in SCENES['intermediate']:
            reconstructionScene(os.path.join(options.intermediate_dir, scene_name),
                                os.path.join(intermediate_work_dir, scene_name))
            subprocess.run(['cp', '-v', os.path.join(intermediate_work_dir, scene_name, 'dense/0/fused.ply'),
                            os.path.join(submit_dir, scene_name + '.ply')], check=True)
            run_colmap_converter(os.path.join(options.intermediate_dir, scene_name),
                                 os.path.join(intermediate_work_dir, scene_name),
                                 os.path.join(submit_dir, scene_name + '.log'),
                                 options.script_dir)

    if options.advanced_dir is not None:
        for scene_name in SCENES['advanced']:
            reconstructionScene(os.path.join(options.advanced_dir, scene_name),
                                os.path.join(advanced_work_dir, scene_name))
            subprocess.run(['cp', '-v', os.path.join(advanced_work_dir, scene_name, 'dense/0/fused.ply'),
                            os.path.join(submit_dir, scene_name + '.ply')], check=True)
            run_colmap_converter(os.path.join(options.advanced_dir, scene_name),
                                 os.path.join(advanced_work_dir, scene_name),
                                 os.path.join(submit_dir, scene_name + '.log'),
                                 options.script_dir)

    logging.info('done')
