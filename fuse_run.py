# -*- coding: UTF-8 -*-

import os, sys
import logging
from utils import SetupFreeGpu, InitLogging, mvs_network_check
import subprocess
from algorithm_wrapper.mvsnet_wrapper import run_mvsnet_fuse
from algorithm_wrapper.pointmvsnet_wrapper import run_pointmvsnet_fuse
from common_options import GLOBAL_OPTIONS as FLAGS


def fuse_colmap(fuse_work_dir):
    colmap_fuse_command_line = ['colmap', 'stereo_fusion',
                                '--workspace_path', os.path.join(fuse_work_dir, 'mvs_result'),
                                '--output_path', os.path.join(fuse_work_dir, 'fused.ply')]
    subprocess.run(colmap_fuse_command_line, check=True)


def fuse_openmvs(fuse_work_dir):
    openmvs_command_line = ['DensifyPointCloud',
                            '--working-folder', mvs_work_dir,
                            os.path.join(mvs_work_dir, 'scene.mvs')]
    subprocess.run(openmvs_command_line, check=True)


def fuse_pmvs(fuse_work_dir):
    pmvs_command_line = ['pmvs2', os.path.join(mvs_work_dir, 'pmvs/'),
                         'option-all']
    subprocess.run(pmvs_command_line, check=True)


def fuse_mve(fuse_work_dir):
    pmvs_command_line = ['dmrecon', '-s2', os.path.join(mvs_work_dir, 'view')]
    subprocess.run(pmvs_command_line, check=True)


def fuse_mvsnet(fuse_work_dir):
    run_mvsnet_fuse(os.path.join(fuse_work_dir, 'mvs_result'))
    os.rename(os.path.join(fuse_work_dir, 'mvs_result/points_mvsnet'), os.path.join(fuse_work_dir, 'fused'))


def fuse_pointmvsnet(fuse_work_dir):
    run_pointmvsnet_fuse(fuse_work_dir)


def fuse_run_helper(alg, fuse_work_dir):
    this_module = sys.modules[__name__]
    mvs_run_fun = getattr(this_module, 'fuse_' + alg)
    mvs_run_fun(fuse_work_dir)


if __name__ == '__main__':
    InitLogging()

    FLAGS.add_argument('fuse_work_dir', type=str, help='working directory')
    FLAGS.add_argument('--sfm', default='colmap', choices=['colmap', 'openmvg', 'theiasfm', 'mve'],
                       help='sfm algorithm')

    fuse_algorithm_list = ['colmap', 'openmvs', 'pmvs', 'cmvs', 'mve',
                           'mvsnet', 'rmvsnet', 'pointmvsnet']
    FLAGS.add_argument('--mvs', type=mvs_network_check, default='colmap', choices=fuse_algorithm_list,
                       help='mvs algorithm')
    FLAGS.add_argument('--fuse', type=mvs_network_check, default='colmap', choices=fuse_algorithm_list,
                       help='fuse algorithm')

    FLAGS.parse_args()

    logging.info('select gpu %s', SetupFreeGpu(FLAGS.num_gpu))

    fuse_run_helper(FLAGS.mvs, FLAGS.fuse_work_dir)
