# -*- coding: UTF-8 -*-

import os
import logging
import shutil

from pipeline.utils import SetupFreeGpu, InitLogging, mvs_network_check
from pipeline.sfm_run import sfm_run_helper
from pipeline.sfm_converter import sfm_convert_helper
from pipeline.mvs_run import mvs_run_helper
from pipeline.mvs_converter import mvs_convert_helper
from pipeline.fuse_run import fuse_run_helper
from pipeline.common_options import GLOBAL_OPTIONS as FLAGS


if __name__ == '__main__':
    InitLogging()

    FLAGS.add_argument('images_dir', type=str, help='images directory to process')
    FLAGS.add_argument('workspace_dir', type=str, help='working directory')
    FLAGS.add_argument('--sfm', default='colmap', choices=['colmap', 'openmvg', 'theiasfm', 'mve'],
                        help='sfm algorithm')

    mvs_algorithm_list = ['colmap', 'openmvs', 'pmvs', 'cmvs', 'mve',
                          'mvsnet', 'rmvsnet', 'pointmvsnet']
    FLAGS.add_argument('--mvs', type=mvs_network_check, default='colmap', choices=mvs_algorithm_list,
                        help='mvs algorithm')
    FLAGS.add_argument('--fuse', type=mvs_network_check, default='colmap', choices=mvs_algorithm_list,
                        help='fuse algorithm')
    FLAGS.add_argument('--auto_rerun', type=bool, default=False, help='auto run left pipeline if a step is missing')

    FLAGS.parse_args()

    logging.info('select gpu %s', SetupFreeGpu(FLAGS.num_gpu))

    if FLAGS.auto_rerun is True:
        force_run = -1000
    else:
        force_run = 0

    sfm_name = 'sfm_' + FLAGS.sfm
    sfm_work_dir = os.path.join(FLAGS.workspace_dir, sfm_name)
    if os.path.exists(sfm_work_dir) is False:
        os.mkdir(sfm_work_dir)
        sfm_run_helper(FLAGS.sfm, FLAGS.images_dir, sfm_work_dir)
        force_run = force_run + 1

    mvs_name = 'mvs_' + FLAGS.sfm + '2' + FLAGS.mvs
    mvs_work_dir = os.path.join(FLAGS.workspace_dir, mvs_name)
    if (os.path.exists(mvs_work_dir) is False) or (force_run > 0):
        shutil.rmtree(mvs_work_dir, ignore_errors=True)
        os.mkdir(mvs_work_dir)
        logging.info('convert sfm(%s) result to mvs(%s)', FLAGS.sfm, FLAGS.mvs)
        sfm_convert_helper(FLAGS.sfm, FLAGS.mvs, sfm_work_dir, FLAGS.images_dir, mvs_work_dir)
        logging.info('run mvs(%s)', FLAGS.mvs)
        mvs_run_helper(FLAGS.mvs, mvs_work_dir)
        force_run = force_run + 1

    fuse_name = 'fuse_' + FLAGS.sfm + '2' + FLAGS.mvs + '2' + FLAGS.fuse
    fuse_work_dir = os.path.join(FLAGS.workspace_dir, fuse_name)
    if (os.path.exists(fuse_work_dir) is False) or (force_run > 0):
        shutil.rmtree(fuse_work_dir, ignore_errors=True)
        os.mkdir(fuse_work_dir)
        logging.info('convert mvs(%s) result to fuse(%s)', FLAGS.mvs, FLAGS.fuse)
        mvs_convert_helper(FLAGS.mvs, FLAGS.fuse, mvs_work_dir, fuse_work_dir)
        logging.info('run fuse(%s)', FLAGS.fuse)
        fuse_run_helper(FLAGS.fuse, fuse_work_dir)
        force_run = force_run + 1
