# -*- coding: UTF-8 -*-

import os, sys
import argparse, logging
from utils import SetupFreeGpu, InitLogging, mvs_network_check
import subprocess
from algorithm_wrapper.mvsnet_wrapper import run_mvsnet_predict, run_rmvsnet_predict
from algorithm_wrapper.pointmvsnet_wrapper import run_pointmvsnet_predict
from common_options import get_common_options_parser


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
    run_mvsnet_predict(options, mvs_work_dir)


def mvs_rmvsnet(options, mvs_work_dir):
    run_rmvsnet_predict(options, mvs_work_dir)


def mvs_pointmvsnet(options, mvs_work_dir):
    run_pointmvsnet_predict(options, mvs_work_dir)


def mvs_run_helper(alg, options, mvs_work_dir):
    this_module = sys.modules[__name__]
    mvs_run_fun = getattr(this_module, 'mvs_' + alg)
    mvs_run_fun(options, mvs_work_dir)


if __name__ == '__main__':
    InitLogging()

    parser = argparse.ArgumentParser(parents=[get_common_options_parser()])
    parser.add_argument('mvs_work_dir', type=str, help='working directory')
    parser.add_argument('--sfm', default='colmap', choices=['colmap', 'openmvg', 'theiasfm', 'mve'],
                        help='sfm algorithm')

    mvs_algorithm_list = ['colmap', 'openmvs', 'pmvs', 'cmvs', 'mve',
                          'mvsnet', 'rmvsnet', 'pointmvsnet']
    parser.add_argument('--mvs', type=mvs_network_check, default='colmap', choices=mvs_algorithm_list,
                        help='mvs algorithm')

    options = parser.parse_args()

    logging.info('select gpu %s', SetupFreeGpu(options.num_gpu))

    mvs_run_helper(options.mvs, options, options.mvs_work_dir)
