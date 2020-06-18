# -*- coding: UTF-8 -*-

import os
import sys
import logging
import argparse
import shutil

from utils import SetupFreeGpu, InitLogging
from sfm_run import sfm_colmap, sfm_openmvg, sfm_theiasfm, sfm_mve
from sfm_normalize import sfm_normalize_colmap
from mvs_run import mvs_colmap

def run_sfm_alg(options, images_dir, sfm_work_dir):
    if options.sfm == 'colmap':
        sfm_colmap(images_dir, sfm_work_dir)
    else:
        raise

def run_sfm_normalize_alg(options, sfm_work_dir, sfm_normalize_work_dir):
    if options.sfm == 'colmap':
        sfm_normalize_colmap(sfm_work_dir, sfm_normalize_work_dir, options.images_dir)
    else:
        raise

def run_mvs_alg(options, sfm_normalize_work_dir, mvs_work_dir):
    if options.mvs == 'colmap':
        mvs_colmap(sfm_normalize_work_dir, mvs_work_dir)
    else:
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str, help='images directory to process')
    parser.add_argument('workspace_dir', type=str, help='working directory')
    parser.add_argument('--sfm', default='colmap', choices=['colmap', 'openmvg', 'theiasfm', 'mve'], help='sfm algorithm')
    parser.add_argument('--mvs', default='colmap', choices=['colmap', 'openmvs', 'pmvs', 'cmvs', 'mve'], help='mvs algorithm')
    parser.add_argument('--auto_rerun', type=bool, default=False, help='auto run left pipeline if a step is missing')
    parser.add_argument('--num_gpu', type=int, default=1, help='how many gpu to use')

    options = parser.parse_args()

    InitLogging()
    logging.info('select gpu %s', SetupFreeGpu(options.num_gpu))

    if options.auto_rerun is True:
        force_run = -1000
    else:
        force_run = 0

    sfm_name = 'sfm_' + options.sfm
    sfm_work_dir = os.path.join(options.workspace_dir, sfm_name)
    if os.path.exists(sfm_work_dir) is False:
        os.mkdir(sfm_work_dir)
        run_sfm_alg(options, options.images_dir, sfm_work_dir)
        force_run = force_run + 1

    sfm_normalize_mame = 'sfm_normalize_' + options.sfm
    sfm_normalize_work_dir = os.path.join(options.workspace_dir, sfm_normalize_mame)
    if (os.path.exists(sfm_normalize_work_dir)) is False or (force_run > 0):
        shutil.rmtree(sfm_normalize_work_dir, ignore_errors=True)
        os.mkdir(sfm_normalize_work_dir)
        run_sfm_normalize_alg(options, sfm_work_dir, sfm_normalize_work_dir)
        force_run = force_run + 1

    if options.sfm != options.mvs:
        logging.info('convert from sfm %s format to mvs %s format', options.sfm, options.mvs)
        sfm_normalize_convert_mame = 'sfm_normalize_' + options.mvs
        sfm_normalize_convert_work_dir = os.path.join(options.workspace_dir, sfm_normalize_convert_mame)
        convert(sfm_normalize_work_dir, sfm_normalize_convert_work_dir)
        sfm_normalize_work_dir = sfm_normalize_convert_work_dir

    mvs_name = 'mvs_' + options.mvs
    mvs_work_dir = os.path.join(options.workspace_dir, mvs_name)
    if (os.path.exists(mvs_work_dir) is False) or (force_run > 0):
        shutil.rmtree(mvs_work_dir, ignore_errors=True)
        os.mkdir(mvs_work_dir)
        run_mvs_alg(options, sfm_normalize_work_dir, mvs_work_dir)
        force_run = force_run + 1