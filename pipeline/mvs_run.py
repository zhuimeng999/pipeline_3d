# -*- coding: UTF-8 -*-

import os, sys
import logging
from pipeline.utils import SetupFreeGpu, InitLogging, mvs_network_check
import subprocess
from algorithm_wrapper.mvsnet_wrapper import run_mvsnet_predict, run_rmvsnet_predict
from algorithm_wrapper.pointmvsnet_wrapper import run_pointmvsnet_predict
from pipeline.common_options import GLOBAL_OPTIONS as FLAGS


def mvs_colmap(mvs_work_dir):
    colmap_patch_match_command_line = ['colmap', 'patch_match_stereo',
                                       '--workspace_path', mvs_work_dir]
    subprocess.run(colmap_patch_match_command_line, check=True)


def mvs_openmvs(mvs_work_dir):
    openmvs_command_line = ['DensifyPointCloud',
                            '--working-folder', mvs_work_dir,
                            os.path.join(mvs_work_dir, 'scene.mvs')]
    subprocess.run(openmvs_command_line, check=True)


def mvs_pmvs(mvs_work_dir):
    pmvs_command_line = ['pmvs2', os.path.join(mvs_work_dir, 'pmvs/'),
                         'option-all']
    subprocess.run(pmvs_command_line, check=True)


def mvs_mve(mvs_work_dir):
    pmvs_command_line = ['dmrecon', '-s2', os.path.join(mvs_work_dir, 'view')]
    subprocess.run(pmvs_command_line, check=True)


def mvs_mvsnet(mvs_work_dir):
    run_mvsnet_predict(mvs_work_dir)


def mvs_rmvsnet(mvs_work_dir):
    run_rmvsnet_predict(mvs_work_dir)


def mvs_pointmvsnet(mvs_work_dir):
    run_pointmvsnet_predict(mvs_work_dir)


def mvs_pmvsnet(mvs_work_dir):
    from algorithm_wrapper.mvsnet_wrapper import get_mvsnet_options
    esmnet_path = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PMVSNet')))
    cl = ['python', os.path.join(esmnet_path, 'test.py'),
          '--data_dir', mvs_work_dir,
          '--output_dir', mvs_work_dir,
          '--ckpt_dir', FLAGS.mvs_use_ckpt]
    subprocess.run(cl + get_mvsnet_options(), check=True)


def mvs_run_helper(alg, mvs_work_dir):
    DONE = os.path.join(mvs_work_dir, 'DONE')
    if os.path.isfile(DONE):
        logging.warning('step mvs for %s in directpry %s already done, skipped', alg, mvs_work_dir)
        return
    this_module = sys.modules[__name__]
    mvs_run_fun = getattr(this_module, 'mvs_' + alg)
    mvs_run_fun(mvs_work_dir)
    with open(DONE, 'w') as f:
        f.write(' '.join(sys.argv))


if __name__ == '__main__':
    InitLogging()

    FLAGS.add_argument('mvs_work_dir', type=str, help='working directory')
    FLAGS.add_argument('--sfm', default='colmap', choices=['colmap', 'openmvg', 'theiasfm', 'mve'],
                       help='sfm algorithm')

    mvs_algorithm_list = ['colmap', 'openmvs', 'pmvs', 'cmvs', 'mve',
                          'mvsnet', 'rmvsnet', 'pointmvsnet']
    FLAGS.add_argument('--mvs', type=mvs_network_check, default='colmap', choices=mvs_algorithm_list,
                       help='mvs algorithm')

    FLAGS.parse_args()

    logging.info('select gpu %s', SetupFreeGpu(FLAGS.num_gpu))

    mvs_run_helper(FLAGS.mvs, FLAGS.mvs_work_dir)
