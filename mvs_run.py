# -*- coding: UTF-8 -*-

import os, sys
import argparse, logging
from utils import SetupFreeGpu, InitLogging
import subprocess
from algorithm_wrapper.mvsnet_wrapper import run_mvsnet_predict


def maybe_convert_sfm_result():
    pass


def mvs_colmap(options, mvs_work_dir):
    colmap_patch_match_command_line = ['colmap', 'patch_match_stereo',
                                       '--workspace_path', mvs_work_dir]
    subprocess.run(colmap_patch_match_command_line, check=True)

def mvs_openmvs(options, mvs_work_dir):
    openmvs_command_line = ['DensifyPointCloud',
                            '--working-folder', mvs_work_dir,
                                       os.path.join(mvs_work_dir, 'scene.mvs')]
    subprocess.run(openmvs_command_line, check=True)

def mvs_pmvs(options, mvs_work_dir):
    pmvs_command_line = ['pmvs2', os.path.join(mvs_work_dir, 'pmvs/'),
                                       'option-all']
    subprocess.run(pmvs_command_line, check=True)


def mvs_mve(options, mvs_work_dir):
    pmvs_command_line = ['dmrecon', '-s2', os.path.join(mvs_work_dir, 'view')]
    subprocess.run(pmvs_command_line, check=True)

def mvs_mvsnet(options, mvs_work_dir):
    run_mvsnet_predict(mvs_work_dir)


def mvs_run_helper(alg, options, mvs_work_dir):
    this_module = sys.modules[__name__]
    mvs_run_fun = getattr(this_module, 'mvs_' + alg)
    mvs_run_fun(options, mvs_work_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sfm_path', type=str, help='sfm result directory')
    parser.add_argument('--sfm_algorithm', default=None, choices=['colmap', 'openmvg', 'theiasfm', 'mve'], help='sfm algorithm type')
    parser.add_argument('output_path', type=str, help='output directory')
    subparser = parser.add_subparsers(dest='alg_type', metavar='algorithm', help='algorithm to use, support are {colmap|openmvs|cmvs-pmvs|mve}')
    parser_colmap = subparser.add_parser('colmap')
    parser_openmvs = subparser.add_parser('openmvs')
    parser_cpmvs = subparser.add_parser('cmvs-pmvs')
    parser_mve = subparser.add_parser('mve')

    options = parser.parse_args()

    InitLogging()
    logging.info('select gpu %s', SetupFreeGpu(options.num_gpu))


    maybe_convert_sfm_result()

    if options.alg_type == 'openmvs':
        mvs_openmvs(options)
    elif options.alg_type == 'cmvs-pmvs':
        mvs_cpmvs(options)
    elif options.alg_type == 'mve':
        mvs_mve(options)
    else:
        mvs_colmap(options)
