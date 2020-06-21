# -*- coding: UTF-8 -*-

import os
import subprocess
from third_party.colmap.read_write_model import read_model, write_model
from common_options import GLOBAL_OPTIONS as FLAGS


def get_mvsnet_path() -> str:
    mvsnet_path = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MVSNet')))
    if os.path.exists(mvsnet_path):
        return mvsnet_path
    else:
        return ''


def get_mvsnet_options():
    mvsnet_options = []
    if FLAGS.mvs_max_w is not None:
        mvsnet_options.append('--max_w')
        mvsnet_options.append(str(FLAGS.mvs_max_w))
    if FLAGS.mvs_max_h is not None:
        mvsnet_options.append('--max_h')
        mvsnet_options.append(str(FLAGS.mvs_max_h))
    if FLAGS.mvs_max_d is not None:
        mvsnet_options.append('--max_d')
        mvsnet_options.append(str(FLAGS.mvs_max_d))
    return mvsnet_options


def export_colmap_to_mvsnet(output_dir):
    if os.path.exists(os.path.join(output_dir, 'sparse/cameras.txt')) is False:
        sparse_dir = os.path.join(output_dir, 'sparse')
        write_model(*read_model(sparse_dir, '.bin'), sparse_dir, '.txt')
    mvsnet_path = get_mvsnet_path()
    colmap2mvsnet_path = os.path.join(mvsnet_path, 'mvsnet/colmap2mvsnet.py')
    colmap2mvsnet_command_line = ['python', colmap2mvsnet_path,
                                  '--dense_folder', output_dir]
    if FLAGS.mvs_max_d is not None:
        colmap2mvsnet_command_line.append('--max_d')
        colmap2mvsnet_command_line.append(str(FLAGS.mvs_max_d))
    subprocess.run(colmap2mvsnet_command_line, check=True)


def run_mvsnet_predict(output_dir):
    mvsnet_path = get_mvsnet_path()
    base_command = 'source ~/anaconda3/bin/activate mvsnet;'
    mvsnet_command_line = ['python', os.path.join(mvsnet_path, 'mvsnet/test.py'),
                           '--dense_folder', output_dir,
                           '--regularization', '3DCNNs',
                           '--interval_scale', '1.06',
                           '--pretrained_model_ckpt_path',
                           os.path.join(mvsnet_path, 'pretrain/tf_model_eth3d/3DCNNs/model.ckpt'),
                           '--ckpt_step', '150000']

    base_command = base_command + '"' + '" "'.join(mvsnet_command_line + get_mvsnet_options()) + '"'
    subprocess.run(['bash', '-c', base_command], cwd=os.path.join(mvsnet_path, 'mvsnet'))


def run_rmvsnet_predict(output_dir):
    mvsnet_path = get_mvsnet_path()
    base_command = 'source ~/anaconda3/bin/activate mvsnet;'
    mvsnet_command_line = ['python', os.path.join(mvsnet_path, 'mvsnet/test.py'),
                           '--dense_folder', output_dir,
                           '--regularization', 'GRU',
                           '--interval_scale', '0.8',
                           '--pretrained_model_ckpt_path',
                           os.path.join(mvsnet_path, 'pretrain/tf_model_eth3d/GRU/model.ckpt'),
                           '--ckpt_step', '150000']
    base_command = base_command + '"' + '" "'.join(mvsnet_command_line + get_mvsnet_options()) + '"'
    subprocess.run(['bash', '-c', base_command], cwd=os.path.join(mvsnet_path, 'mvsnet'))
