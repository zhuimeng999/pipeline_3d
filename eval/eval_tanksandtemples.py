# -*- coding: UTF-8 -*-

import os
import subprocess
from pipeline.utils import InitLogging
from pipeline.common_options import GLOBAL_OPTIONS as FLAGS


if __name__ == '__main__':
    InitLogging()
    FLAGS.add_argument('tt_dir', help='TanksAndTemples directory')
    FLAGS.add_argument('colmap_sparse_dir', help='colmap sparse reconstrution result directory')
    FLAGS.add_argument('ply_path', help='ply file path')
    FLAGS.add_argument('--dataset_name', default='Ignatius')
    FLAGS.add_argument('--work_dir', default=None, help='working directory')
    FLAGS.parse_args()
    if FLAGS.work_dir is None:
        FLAGS.work_dir = os.path.dirname(FLAGS.ply_path)
    if os.path.exists(FLAGS.work_dir) is False:
        os.mkdir(FLAGS.work_dir)
    conver_to_log_command_line = ['python', 'convert_to_logfile.py',
                                  os.path.join(FLAGS.colmap_sparse_dir, 'cameras.bin'),
                                  os.path.join(FLAGS.work_dir, 'cameras.log'),
                                  os.path.join(FLAGS.tt_dir, FLAGS.dataset_name, 'images'),
                                  'COLMAP',
                                  'jpg']
    subprocess.run(conver_to_log_command_line, check=True, cwd=os.path.join(FLAGS.tt_dir, 'TanksAndTemples/python_toolbox'))
    eval_command_line = ['python', 'run.py',
                         '--dataset-dir', os.path.join(FLAGS.tt_dir, FLAGS.dataset_name),
                         '--traj-path', os.path.join(FLAGS.work_dir, 'cameras.log'),
                         '--ply-path', FLAGS.ply_path]
    subprocess.run(eval_command_line, check=True, cwd=os.path.join(FLAGS.tt_dir, 'TanksAndTemples/python_toolbox/evaluation'))