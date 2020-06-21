# -*- coding: UTF-8 -*-

import os
import sys
import logging
import argparse
import shutil

from utils import SetupFreeGpu, InitLogging, mvs_network_check
from sfm_run import sfm_run_helper
from sfm_converter import sfm_convert_helper
from mvs_run import mvs_run_helper
from common_options import get_common_options_parser


if __name__ == '__main__':
    InitLogging()
    parser = argparse.ArgumentParser(parents=[get_common_options_parser()])
    parser.add_argument('images_dir', type=str, help='images directory to process')
    parser.add_argument('workspace_dir', type=str, help='working directory')
    parser.add_argument('--sfm', default='colmap', choices=['colmap', 'openmvg', 'theiasfm', 'mve'],
                        help='sfm algorithm')

    mvs_algorithm_list = ['colmap', 'openmvs', 'pmvs', 'cmvs', 'mve',
                          'mvsnet', 'rmvsnet', 'pointmvsnet']
    parser.add_argument('--mvs', type=mvs_network_check, default='colmap', choices=mvs_algorithm_list,
                        help='mvs algorithm')
    parser.add_argument('--auto_rerun', type=bool, default=False, help='auto run left pipeline if a step is missing')

    options = parser.parse_args()

    logging.info('select gpu %s', SetupFreeGpu(options.num_gpu))

    if options.auto_rerun is True:
        force_run = -1000
    else:
        force_run = 0

    sfm_name = 'sfm_' + options.sfm
    sfm_work_dir = os.path.join(options.workspace_dir, sfm_name)
    if os.path.exists(sfm_work_dir) is False:
        os.mkdir(sfm_work_dir)
        sfm_run_helper(options.sfm, options, options.images_dir, sfm_work_dir)
        force_run = force_run + 1

    mvs_name = 'mvs_' + options.sfm + '2' + options.mvs
    mvs_work_dir = os.path.join(options.workspace_dir, mvs_name)
    if (os.path.exists(mvs_work_dir) is False) or (force_run > 0):
        shutil.rmtree(mvs_work_dir, ignore_errors=True)
        os.mkdir(mvs_work_dir)
        logging.info('convert sfm(%s) result to mvs(%s)', options.sfm, options.mvs)
        sfm_convert_helper(options.sfm, options.mvs, options, sfm_work_dir, options.images_dir, mvs_work_dir)
        logging.info('run mvs(%s)', options.mvs)
        mvs_run_helper(options.mvs, options, mvs_work_dir)
        force_run = force_run + 1
