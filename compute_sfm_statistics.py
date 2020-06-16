# -*- coding: UTF-8 -*-

import argparse
import logging
import os
import pathlib
import subprocess

from sfm_run import get_sfm_parser
from third_party.colmap.read_write_model import read_model
from utils import InitLogging

module_logger = logging.getLogger('module_logger')


def ConvertTheiasfmToColmap(theiasfm_reconstruction_filename, sfm_colmap_path):
    theiasfm_to_colmap_command_line = ['export_colmap_files',
                                       '-input_reconstruction_file', theiasfm_reconstruction_filename,
                                       '-output_folder', sfm_colmap_path,
                                       '--logtostderr']
    subprocess.run(theiasfm_to_colmap_command_line, check=True)


def FindOrConvertSfmResultToColmap(options):
    sfm_model_path = None
    if options.alg_type == 'colmap':
        if options.build_id is None:
            sparse_path = pathlib.Path(os.path.join(options.sfm_path, 'sparse'))
            if len(list(sparse_path.iterdir())) != 3:
                logging.fatal('there are many reconstruction resule in %s, you must special a build_id',
                              sparse_path.as_posix(), )
            sfm_model_path = os.path.join(options.sfm_path, 'sparse/0')
        else:
            sfm_model_path = os.path.join(options.sfm_path, 'sparse', str(options.build_id))
    elif options.alg_type == 'openmvg':
        pass
    elif options.alg_type == 'theiasfm':
        sfm_model_path = os.path.join(options.sfm_path, 'sfm_colmap')
        if not os.path.isdir(sfm_model_path):
            os.mkdir(sfm_model_path)

        if options.build_id is None:
            reconstructions = list(pathlib.Path(options.sfm_path).glob('reconstruction.bin*'))
            if len(reconstructions) != 1:
                logging.fatal('there are many reconstruction resule in %s, you must special a build_id',
                              reconstructions)
            theiasfm_reconstruction_filename = str(reconstructions[0].absolute().as_posix())
        else:
            theiasfm_reconstruction_filename = os.path.join(options.sfm_path,
                                                            'reconstruction.bin-' + str(options.build_id))
        ConvertTheiasfmToColmap(theiasfm_reconstruction_filename, sfm_model_path)

    elif options.alg_type == 'mve':
        pass
    else:
        logging.fatal('unknown algorithm type: %s', options.alg_type)
    return sfm_model_path


if __name__ == '__main__':
    InitLogging()
    parser = argparse.ArgumentParser()
    parser.add_argument('sfm_path', help='sfm reconstruction result directory')
    parser.add_argument('--alg_type', default=None, choices=['colmap', 'openmvg', 'theiasfm', 'mve'])
    parser.add_argument('--build_id', default=None, type=int)
    options = parser.parse_args()
    if options.alg_type is None:
        with open(os.path.join(options.sfm_path, 'sfm_run_options.txt'), 'r') as f:
            line = map(lambda x: x.strip(), filter(lambda x: x.rstrip()[0] != '#', f))
            options.alg_type = get_sfm_parser().parse_args(list(line)).alg_type
            module_logger.info('use algorithm type get from sfm config file: %s', options.alg_type)

    sfm_model_path = FindOrConvertSfmResultToColmap(options)
    module_logger.info('find reconstruction on: %s', sfm_model_path)
    cameras, images, points3D = read_model(sfm_model_path, '.bin' if options.alg_type == 'colmap' else '.txt')
    module_logger.info('\nNum views: %d\nNum 3D points: %d', len(images), len(points3D))

