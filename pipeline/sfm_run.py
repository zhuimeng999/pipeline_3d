# -*- coding: UTF-8 -*-

"""
    USAGE: %s [flags] input_path output_path
"""

import subprocess
from pipeline.utils import SetupFreeGpu, InitLogging, LogThanExitIfFailed
import os, sys
import argparse
import pathlib
import logging
from pipeline.common_options import GLOBAL_OPTIONS as FLAGS


def sfm_colmap(images_dir, work_dir):
    '''
    for fast https://github.com/colmap/colmap/issues/116 :
    --Mapper.ba_global_images_ratio 1.2
    --Mapper.ba_global_points_ratio 1.2
    --Mapper.ba_global_max_num_iterations 20
    --Mapper.ba_global_max_refinement 3
    --Mapper.ba_global_points_freq 200000
    '''

    if pathlib.Path(work_dir).joinpath('sparse/0').is_dir():
        logging.warning('skip colmap sfm run, because directory %s already exist', pathlib.Path(work_dir).joinpath('sparse/0'))
        return
    LogThanExitIfFailed(FLAGS.sfm_global is False,
                        'colmap only support INCREMENTAL sfm reconstruction')
    colmap_command_line = ['colmap', 'automatic_reconstructor',
                           '--workspace_path', work_dir,
                           '--image_path', images_dir,
                           '--sparse', str(1),
                           '--dense', str(0),
                           '--num_threads', str(FLAGS.num_cpu)]
    logging.info('run colmap with args:\n %s', colmap_command_line)
    subprocess.run(colmap_command_line, check=True)


def sfm_openmvg(images_dir, work_dir):
    if os.path.isfile(os.path.join(work_dir, 'sfm_data.bin')):
        logging.warning('sfm_data.bin already exsits, skip openmvg run in directory %s', work_dir)
        return
    sensor_db = 'data/sensor_width_camera_database.txt'

    image_listing_command_line = ['openMVG_main_SfMInit_ImageListing',
                                  '--imageDirectory', images_dir,
                                  '--sensorWidthDatabase', sensor_db,
                                  '--outputDirectory', work_dir]
    subprocess.run(image_listing_command_line, check=True)

    matches_dir = os.path.join(work_dir, 'matches')
    if os.path.isdir(matches_dir) is False:
        os.mkdir(matches_dir)

    ComputeFeatures_command_line = ['openMVG_main_ComputeFeatures',
                                    '--input_file', os.path.join(work_dir, 'sfm_data.json'),
                                    '--outdir', matches_dir]
    subprocess.run(ComputeFeatures_command_line, check=True)

    ComputeMatches_command_line = ['openMVG_main_ComputeMatches',
                                   '--input_file', os.path.join(work_dir, 'sfm_data.json'),
                                   '--out_dir', matches_dir]
    subprocess.run(ComputeMatches_command_line, check=True)

    if FLAGS.sfm_global is False:
        IncrementalSfM_command_line = ['openMVG_main_IncrementalSfM',
                                       '--input_file', os.path.join(work_dir, 'sfm_data.json'),
                                       '--matchdir', matches_dir,
                                       '--outdir', work_dir]
        subprocess.run(IncrementalSfM_command_line, check=True)
    else:
        IncrementalSfM_command_line = ['openMVG_main_GlobalSfM',
                                       '--input_file', os.path.join(work_dir, 'sfm_data.json'),
                                       '--matchdir', matches_dir,
                                       '--outdir', work_dir]
        subprocess.run(IncrementalSfM_command_line, check=True)


def sfm_theiasfm(options, images_dir, work_dir):
    tmp = pathlib.Path(images_dir)
    images = None
    for f in tmp.iterdir():
        if f.suffix in ['.JPG', '.jpg', '.PNG', '.png'] and f.is_file():
            images = '*' + f.suffix
            break
    assert isinstance(images, str)

    with open('data/build_reconstruction_flags.txt', 'r') as f:
        content = f.read()
    theiasfm_flagfile = os.path.join(work_dir, 'theiasfm_flagfile.txt')
    matching_work_directory = os.path.join(work_dir, 'matching')
    if os.path.isdir(matching_work_directory) is False:
        os.mkdir(matching_work_directory)

    content = content.replace('--images=', '--images=' + os.path.join(images_dir, images)).replace(
        '--output_reconstruction=', '--output_reconstruction=' + os.path.join(work_dir, 'reconstruction.bin')).replace(
        '--matching_working_directory=', '--matching_working_directory=' + matching_work_directory).replace(
        '--intrinsics_to_optimize=NONE', '--intrinsics_to_optimize=FOCAL_LENGTH|PRINCIPAL_POINTS|RADIAL_DISTORTION').replace(
        '--num_threads=16','--num_threads=' + str(options.num_cpu))

    if options.sfm_global is False:
        content = content.replace(
            '--reconstruction_estimator=GLOBAL', '--reconstruction_estimator=INCREMENTAL')
    with open(theiasfm_flagfile, 'w') as f:
        f.write(content)

    theiasfm_command_line = ['build_reconstruction', '--flagfile', theiasfm_flagfile]
    subprocess.run(theiasfm_command_line, check=True)


def sfm_mve(options, images_dir, work_dir):
    makescene_command_line = ['makescene', '-i', images_dir, os.path.join(work_dir, 'view')]
    subprocess.run(makescene_command_line, check=True)
    sfmrecon_command_line = ['sfmrecon', os.path.join(work_dir, 'view')]
    subprocess.run(sfmrecon_command_line, check=True)


def sfm_run_helper(alg, images_dir, sfm_work_dir):
    this_module = sys.modules[__name__]
    sfm_run_fun = getattr(this_module, 'sfm_' + alg)
    sfm_run_fun(images_dir, sfm_work_dir)

def get_sfm_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_path', type=str, help='input images directory')
    parser.add_argument('output_path', type=str, help='output directory')
    parser.add_argument('--reconstruction_estimator', default='INCREMENTAL', choices=['INCREMENTAL', 'GLOBAL'])
    subparser = parser.add_subparsers(help='algorithm to use, support are {colmap|openmvg|theiasfm|mve}',
                                      metavar='algorithm', dest='alg_type')
    parser_colmap = subparser.add_parser('colmap')
    parser_openmvg = subparser.add_parser('openmvg')
    parser_theiasfm = subparser.add_parser('theiasfm')
    parser_theiasfm = subparser.add_parser('mve')

    return parser


def main():
    options = get_sfm_parser().parse_args()
    logging.info('select gpu %s', SetupFreeGpu(options.num_gpu))

    with open(os.path.join(options.output_path, 'sfm_run_options.txt'), 'w') as f:
        f.write('\n'.join(map(lambda x: x if ' ' not in x else '\'' + x + '\'', sys.argv[1:])))

    if options.alg_type == 'openmvg':
        sfm_openmvg(options)
    elif options.alg_type == 'theiasfm':
        sfm_theiasfm(options)
    elif options.alg_type == 'mve':
        sfm_mve(options)
    elif options.alg_type == 'colmap':
        sfm_colmap(options)
    else:
        LogThanExitIfFailed(False, 'you must provide a algorithm to use')


if __name__ == '__main__':
    InitLogging()
    main()