# -*- coding: UTF-8 -*-

import os
import logging
import pathlib
import subprocess

from pipeline.utils import SetupFreeGpu, InitLogging, mvs_network_check, LogThanExitIfFailed
from pipeline.sfm_run import sfm_run_helper
from pipeline.sfm_converter import sfm_convert_helper
from pipeline.mvs_run import mvs_run_helper
from pipeline.mvs_converter import mvs_convert_helper
from pipeline.fuse_run import fuse_run_helper
from pipeline.common_options import GLOBAL_OPTIONS as FLAGS


def run_colmap_converter(image_dir, result_dir, log_output, script_dir):
    conver_to_log_command_line = ['python', 'convert_to_logfile.py',
                                  os.path.join(result_dir, 'sparse/0/camera.bin'),
                                  log_output,
                                  image_dir,
                                  'COLMAP',
                                  'jpg']
    subprocess.run(conver_to_log_command_line, check=True,
                   cwd=os.path.join(script_dir, 'TanksAndTemples/python_toolbox'))

class TanksAndTemplesDataset:
    SCENES = {
        'intermediate': ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train'],
        'advanced': ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']
    }

    def __init__(self, scene_dir, sfm_dir, mvs_dir, fuse_dir):
        self.scene_dir = scene_dir
        self.sfm_dir = sfm_dir
        self.mvs_dir = mvs_dir
        self.fuse_dir = fuse_dir

        sfm_work_infos = []
        mvs_work_infos = []
        fuse_work_infos = []
        for k, v in self.SCENES.items():
            for scene_name in v:
                scene_path = os.path.join(self.scene_dir, k, scene_name)
                sfm_path = os.path.join(self.sfm_dir, k, scene_name)
                mvs_path = os.path.join(self.mvs_dir, k, scene_name)
                fuse_path = os.path.join(self.fuse_dir, k, scene_name)
                LogThanExitIfFailed(os.path.isdir(scene_path), 'scene ' + scene_path + ' is not exists')
                sfm_work_infos.append((scene_path, sfm_path))
                mvs_work_infos.append((scene_path, sfm_path, mvs_path))
                fuse_work_infos.append((scene_name, mvs_path, fuse_path))

        self.sfm_work_infos = sfm_work_infos
        self.mvs_work_infos = mvs_work_infos
        self.fuse_work_infos = fuse_work_infos

    def get_sfm_infos(self):
        return self.sfm_work_infos

    def get_mvs_infos(self):
        return self.mvs_work_infos

    def get_fuse_infos(self):
        return self.fuse_work_infos


class TanksAndTemplesTrainDataset:
    SCENES = []

    def __init__(self, scene_dir, sfm_dir, mvs_dir, fuse_dir):
        self.scene_dir = scene_dir
        self.sfm_dir = sfm_dir
        self.mvs_dir = mvs_dir
        self.fuse_dir = fuse_dir

        sfm_work_infos = []
        mvs_work_infos = []
        fuse_work_infos = []
        for scene_name in self.SCENES:
            scene_path = os.path.join(self.scene_dir, scene_name)
            sfm_path = os.path.join(self.sfm_dir, scene_name)
            mvs_path = os.path.join(self.mvs_dir, scene_name)
            fuse_path = os.path.join(self.fuse_dir, scene_name)
            LogThanExitIfFailed(os.path.isdir(scene_path), 'scene ' + scene_path + ' is not exists')
            sfm_work_infos.append((scene_path, sfm_path))
            mvs_work_infos.append((scene_path, sfm_path, mvs_path))
            fuse_work_infos.append((scene_name, scene_path, mvs_path, fuse_path))

        self.sfm_work_infos = sfm_work_infos
        self.mvs_work_infos = mvs_work_infos
        self.fuse_work_infos = fuse_work_infos

    def get_sfm_infos(self):
        return self.sfm_work_infos

    def get_mvs_infos(self):
        return self.mvs_work_infos

    def get_fuse_infos(self):
        return self.fuse_work_infos


if __name__ == '__main__':
    InitLogging()

    FLAGS.add_argument('data_dir', type=str, default=None,
                       help='dataset directory, specail this for intermediate and advanced set evalution at same time')
    FLAGS.add_argument('sfm_dir', type=str, help='workspace directory')
    FLAGS.add_argument('mvs_dir', type=str, help='working directory')
    FLAGS.add_argument('fuse_dir', type=str, help='working directory')
    FLAGS.add_argument('--dataset', type=str, default='tt', help='dataset type')
    FLAGS.add_argument('--script_dir', type=str, default=None,
                       help='convert_to_logfile.py')
    FLAGS.add_argument('--sfm', default='colmap', choices=['colmap', 'openmvg', 'theiasfm', 'mve'],
                       help='sfm algorithm')

    mvs_algorithm_list = ['colmap', 'openmvs', 'pmvs', 'cmvs', 'mve',
                          'mvsnet', 'rmvsnet', 'pointmvsnet']
    FLAGS.add_argument('--mvs', type=mvs_network_check, default='colmap', choices=mvs_algorithm_list,
                       help='mvs algorithm')
    FLAGS.add_argument('--fuse', type=mvs_network_check, default='colmap', choices=mvs_algorithm_list,
                       help='fuse algorithm')
    FLAGS.add_argument('--eval_dir', type=str, default=None, help='whether to eval output')

    FLAGS.parse_args()

    logging.info('select gpu %s', SetupFreeGpu(FLAGS.num_gpu))

    if FLAGS.mvs in ['mvsnet', 'rmvsnet']:
        if FLAGS.mvs_max_w is None:
            FLAGS.mvs_max_w = 1152 if FLAGS.mvs == 'mvsnet' else 1600
        if FLAGS.mvs_max_h is None:
            FLAGS.mvs_max_h = 864 if FLAGS.mvs == 'mvsnet' else 1200
        if FLAGS.mvs_max_d is None:
            FLAGS.mvs_max_d = 192 if FLAGS.mvs == 'mvsnet' else 256

    if FLAGS.script_dir is None:
        FLAGS.script_dir = os.path.join(FLAGS.data_dir, '../../')
        LogThanExitIfFailed(os.path.isdir(FLAGS.script_dir),
                            "you must provide the TanksAndTemples eval script directory")
    LogThanExitIfFailed(FLAGS.eval_dir is None or os.path.isdir(FLAGS.eval_dir),
                        "you must provide the TanksAndTemples eval output directory")

    if FLAGS.dataset == 'tt':
        ds = TanksAndTemplesDataset(FLAGS.data_dir, FLAGS.sfm_dir, FLAGS.mvs_dir, FLAGS.fuse_dir)
    elif FLAGS.dataset == 'ttt':
        ds = TanksAndTemplesTrainDataset(FLAGS.data_dir, FLAGS.sfm_dir, FLAGS.mvs_dir, FLAGS.fuse_dir)

    for images_dir, sfm_work_dir in ds.get_sfm_infos():
        pathlib.Path(sfm_work_dir).mkdir(parents=True, exist_ok=True)
        sfm_run_helper(FLAGS.sfm, images_dir, sfm_work_dir)

    for images_dir, sfm_work_dir, mvs_work_dir in ds.get_mvs_infos():
        pathlib.Path(mvs_work_dir).mkdir(parents=True, exist_ok=True)
        sfm_convert_helper(FLAGS.sfm, FLAGS.mvs, sfm_work_dir, images_dir, mvs_work_dir)
        mvs_run_helper(FLAGS.mvs, mvs_work_dir)

    for scene_name, mvs_work_dir, fuse_work_dir in ds.get_fuse_infos():
        pathlib.Path(fuse_work_dir).mkdir(parents=True, exist_ok=True)
        mvs_convert_helper(FLAGS.mvs, FLAGS.fuse, mvs_work_dir, fuse_work_dir)
        fuse_run_helper(FLAGS.fuse, fuse_work_dir)

    if FLAGS.eval_dir is not None:
        for scene_name, scene_path, mvs_work_dir, fuse_work_dir in ds.get_fuse_infos():
            subprocess.run(['cp', '-v', os.path.join(fuse_work_dir, 'fused.ply'),
                            os.path.join(FLAGS.eval_dir, scene_name + '.ply')], check=True)
            run_colmap_converter(scene_path,
                                 mvs_work_dir,
                                 os.path.join(FLAGS.eval_dir, scene_name + '.log'),
                                 FLAGS.script_dir)

