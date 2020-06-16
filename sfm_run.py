# -*- coding: UTF-8 -*-

"""
    USAGE: %s [flags] input_path output_path
"""

import subprocess
from utils import SetupOneGpu, InitLogging
import os, sys
import argparse
import logging
import pathlib

module_logger = logging.getLogger('module_logger')

def sfm_mve(options):
    makescene_command_line = ['makescene', '-i', options.images_path, os.path.join(options.output_path, 'view')]
    subprocess.run(makescene_command_line, check=True)
    sfmrecon_command_line = ['sfmrecon', os.path.join(options.output_path, 'view')]
    subprocess.run(sfmrecon_command_line, check=True)

def sfm_theiasfm(options):
    tmp = pathlib.Path(options.images_path)
    images = None
    for f in tmp.iterdir():
        if f.suffix in ['.JPG', '.jpg', '.PNG', '.png'] and f.is_file():
            images = '*' + f.suffix
            break
    assert isinstance(images, str)

    with open('data/build_reconstruction_flags.txt', 'r') as f:
        content = f.read()
    theiasfm_flagfile = os.path.join(options.output_path, 'theiasfm_flagfile.txt')
    matching_work_directory = os.path.join(options.output_path, 'matching')
    if os.path.isdir(matching_work_directory) is False:
        os.mkdir(matching_work_directory)

    content = content.replace('--images=', '--images=' + os.path.join(options.images_path, images)).replace(
        '--output_matches_file=', '--output_matches_file=' + os.path.join(options.output_path, 'matches.txt')).replace(
        '--output_reconstruction=', '--output_reconstruction=' + os.path.join(options.output_path, 'reconstruction.ply')).replace(
        '--matching_working_directory=', '--matching_working_directory=' + matching_work_directory)\

    if options.reconstruction_estimator == 'INCREMENTAL':
        content = content.replace(
        '--reconstruction_estimator=GLOBAL', '--reconstruction_estimator=INCREMENTAL')
    with open(theiasfm_flagfile, 'w') as f:
        f.write(content)

    theiasfm_command_line = ['build_reconstruction', '--flagfile', theiasfm_flagfile]
    subprocess.run(theiasfm_command_line, check=True)


def sfm_openmvg(options):
    sensor_db = 'data/sensor_width_camera_database.txt'

    image_listing_command_line = ['openMVG_main_SfMInit_ImageListing',
                                  '--imageDirectory', options.images_path,
                                  '--sensorWidthDatabase', sensor_db,
                                  '--outputDirectory', options.output_path]
    subprocess.run(image_listing_command_line, check=True)

    ComputeFeatures_command_line = ['openMVG_main_ComputeFeatures',
                                    '--input_file', os.path.join(options.output_path, 'sfm_data.json'),
                                    '--outdir', options.output_path]
    subprocess.run(ComputeFeatures_command_line, check=True)

    ComputeMatches_command_line = ['openMVG_main_ComputeMatches',
                                   '--input_file', os.path.join(options.output_path, 'sfm_data.json'),
                                   '--out_dir', options.output_path]
    subprocess.run(ComputeMatches_command_line, check=True)

    if options.reconstruction_estimator == 'INCREMENTAL':
        IncrementalSfM_command_line = ['openMVG_main_IncrementalSfM',
                                       '--input_file', os.path.join(options.output_path, 'sfm_data.json'),
                                       '--matchdir', options.output_path,
                                       '--outdir', options.output_path]
        subprocess.run(IncrementalSfM_command_line, check=True)
    else:
        IncrementalSfM_command_line = ['openMVG_main_GlobalSfM',
                                       '--input_file', os.path.join(options.output_path, 'sfm_data.json'),
                                       '--matchdir', options.output_path,
                                       '--outdir', options.output_path]
        subprocess.run(IncrementalSfM_command_line, check=True)


def sfm_colmap(options):
    if options.reconstruction_estimator != 'INCREMENTAL':
        logging.fatal('colmap only support INCREMENTAL sfm reconstruction')
    colmap_command_line = ['colmap', 'automatic_reconstructor',
                           '--workspace_path', options.output_path,
                           '--image_path', options.images_path,
                           '--sparse', str(1),
                           '--dense', str(0)]
    logging.info('run colmap with args:\n %s', colmap_command_line)
    subprocess.run(colmap_command_line, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_path', type=str, help='input images directory')
    parser.add_argument('output_path', type=str, help='output directory')
    parser.add_argument('--reconstruction_estimator', default='INCREMENTAL', choices=['INCREMENTAL', 'GLOBAL'])
    subparser = parser.add_subparsers(help='algorithm to use, support are {colmap|openmvg|theiasfm|mve}', metavar='algorithm', dest='alg_type', required=True)
    parser_colmap = subparser.add_parser('colmap')
    parser_openmvg = subparser.add_parser('openmvg')
    parser_theiasfm = subparser.add_parser('theiasfm')
    parser_theiasfm = subparser.add_parser('mve')

    options = parser.parse_args()

    with open(os.path.join(options.output_path, 'sfm_run_options.txt'), 'w') as f:
        f.write('\n'.join(map(lambda x: x if x.startswith('-') else '\'' + x + '\'', sys.argv[1:])))

    module_logger.info('select gpu %d, has free memory %d MiB', *SetupOneGpu())
    if options.alg_type == 'openmvg':
        sfm_openmvg(options)
    elif options.alg_type == 'theiasfm':
        sfm_theiasfm(options)
    elif options.alg_type == 'mve':
        sfm_mve(options)
    else:
        sfm_colmap(options)


if __name__ == '__main__':
    InitLogging()
    main()
