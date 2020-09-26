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


class BaseDataset:
    # SCENES = ['Barn', 'Caterpillar', 'Church', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']

    def __init__(self, scene_dir, sfm_dir, mvs_dir, fuse_dir):
        self.scene_dir = scene_dir
        self.sfm_dir = sfm_dir
        self.mvs_dir = mvs_dir
        self.fuse_dir = fuse_dir

        self.scene_names = list(map(lambda x: x.name, filter(lambda x: x.is_dir(), pathlib.Path(scene_dir).iterdir())))
        self.scene_names.sort()
        logging.info('all scenes: %s', self.scene_names)
        sfm_work_infos = []
        mvs_work_infos = []
        fuse_work_infos = []
        for scene_name in self.scene_names:
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
    FLAGS.add_argument('--work_dir', type=str, default=None)
    FLAGS.add_argument('--sfm_dir', type=str, default=None, help='workspace directory')
    FLAGS.add_argument('--mvs_dir', type=str, default=None, help='working directory')
    FLAGS.add_argument('--fuse_dir', type=str, default=None, help='working directory')
    FLAGS.add_argument('--eval_dir', type=str, default=None, help='whether to eval output')
    FLAGS.add_argument('--tt', type=bool, default=False, help='dataset type')
    FLAGS.add_argument('--report', default=False, action='store_true', help='whether to generate report')
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

    FLAGS.parse_args()

    if FLAGS.work_dir is not None:
        FLAGS.sfm_dir = FLAGS.sfm_dir or os.path.join(FLAGS.work_dir, 'sfm_' + FLAGS.sfm)
        FLAGS.mvs_dir = FLAGS.mvs_dir or os.path.join(FLAGS.work_dir, 'mvs_' + FLAGS.sfm + '2' + FLAGS.mvs)
        FLAGS.fuse_dir = FLAGS.fuse_dir or os.path.join(FLAGS.work_dir,
                                                        'fuse_' + FLAGS.sfm + '2' + FLAGS.mvs + '2' + FLAGS.fuse)
        FLAGS.eval_dir = FLAGS.eval_dir or os.path.join(FLAGS.work_dir,
                                                        'submit_' + FLAGS.sfm + '2' + FLAGS.mvs + '2' + FLAGS.fuse)
    else:
        LogThanExitIfFailed((FLAGS.sfm_dir is not None) and (FLAGS.sfm_dir is not None) and
                            (FLAGS.sfm_dir is not None) and (FLAGS.eval_dir is not None),
                            'you must provide work_dir or (sfm_dir, mvs_dir, fuse_dir, eval_dir)')
    logging.info(FLAGS.sfm_dir)
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

    if FLAGS.tt:
        ds = TanksAndTemplesDataset(FLAGS.data_dir, FLAGS.sfm_dir, FLAGS.mvs_dir, FLAGS.fuse_dir)
    else:
        ds = BaseDataset(FLAGS.data_dir, FLAGS.sfm_dir, FLAGS.mvs_dir, FLAGS.fuse_dir)

    for images_dir, sfm_work_dir in ds.get_sfm_infos():
        pathlib.Path(sfm_work_dir).mkdir(parents=True, exist_ok=True)
        sfm_run_helper(FLAGS.sfm, images_dir, sfm_work_dir)

    for images_dir, sfm_work_dir, mvs_work_dir in ds.get_mvs_infos():
        pathlib.Path(mvs_work_dir).mkdir(parents=True, exist_ok=True)
        sfm_convert_helper(FLAGS.sfm, FLAGS.mvs, sfm_work_dir, images_dir, mvs_work_dir)
        mvs_run_helper(FLAGS.mvs, mvs_work_dir)

    for scene_name, scene_path, mvs_work_dir, fuse_work_dir in ds.get_fuse_infos():
        pathlib.Path(fuse_work_dir).mkdir(parents=True, exist_ok=True)
        mvs_convert_helper(FLAGS.mvs, FLAGS.fuse, mvs_work_dir, fuse_work_dir)
        fuse_run_helper(FLAGS.fuse, fuse_work_dir)

    for scene_name, scene_path, mvs_work_dir, fuse_work_dir in ds.get_fuse_infos():
        pathlib.Path(FLAGS.eval_dir).mkdir(parents=True, exist_ok=True)
        fused_path = os.path.join(FLAGS.eval_dir, scene_name + '.ply')
        log_path = os.path.join(FLAGS.eval_dir, scene_name + '.log')
        if os.path.isfile(fused_path) is False:
            subprocess.run(['cp', '-v', os.path.join(fuse_work_dir, FLAGS.fuse + '_fused.ply'),
                            fused_path], check=True)
        else:
            logging.warning('skip copy file %s, already exist', fused_path)
        if os.path.isfile(log_path) is False:
            conver_to_log_command_line = ['python', 'convert_to_logfile.py',
                                          os.path.join(mvs_work_dir, 'sparse/camera.bin'),
                                          log_path,
                                          scene_path,
                                          'COLMAP',
                                          'jpg']
            subprocess.run(conver_to_log_command_line, check=True,
                           cwd=os.path.join(FLAGS.script_dir, 'TanksAndTemples/python_toolbox'))
        else:
            logging.warning('skip generate log file %s, already exist', log_path)

    if FLAGS.report:
        report = {}
        for scene_name, scene_path, mvs_work_dir, fuse_work_dir in ds.get_fuse_infos():
            fused_path = os.path.join(FLAGS.eval_dir, scene_name + '.ply')
            log_path = os.path.join(FLAGS.eval_dir, scene_name + '.log')
            report_path = os.path.join(FLAGS.eval_dir, scene_name + '_report.txt')
            if os.path.isfile(report_path):
                logging.warning('skip generate report for scene %s, because %s already exist', scene_name, report_path)
                with open(report_path, 'r') as f:
                    outputs = f.readlines()
            else:
                eval_command_line = ['python', '-u', 'run.py',
                                     '--dataset-dir', os.path.join(FLAGS.script_dir, 'eval', scene_name),
                                     '--traj-path', log_path,
                                     '--ply-path', fused_path]
                proc = subprocess.Popen(eval_command_line, stdout=subprocess.PIPE,
                                        cwd=os.path.join(FLAGS.script_dir, 'TanksAndTemples/python_toolbox/evaluation'))

                outputs = []
                try:
                    while proc.poll() is None:
                        line = proc.stdout.readline()
                        line = line if isinstance(line, str) else line.decode()
                        print(line , end='')
                        outputs.append(line)

                    line = proc.stdout.readlines()
                    line = list(map(lambda x: x if isinstance(x, str) else x.decode(), line))
                    print(''.join(line), end='')
                    outputs = outputs + line
                except:
                    proc.kill()
                    proc.wait()
                    raise
                LogThanExitIfFailed(proc.returncode == 0,
                                    'command ' + ' '.join(eval_command_line) + ' return ' + str(proc.returncode))
                with open(report_path, 'w') as f:
                    f.write(''.join(outputs))

            report[scene_name] = [-1, -1, -1]
            for line in outputs:
                if line.startswith('precision :'):
                    report[scene_name][0] = float(line.split(':')[1])
                if line.startswith('recall :'):
                    report[scene_name][1] = float(line.split(':')[1])
                if line.startswith('f-score :'):
                    report[scene_name][2] = float(line.split(':')[1])
        from tabulate import tabulate

        total_sumery = tabulate(report, headers='keys', showindex=['precision', 'recall', 'f-score'])
        with open(os.path.join(FLAGS.eval_dir, 'total_sumery.txt'), 'w') as f:
            import sys
            f.write(' '.join(sys.argv))
            f.write('\n')
            f.write(total_sumery)
            f.write('\n')
        print(total_sumery)