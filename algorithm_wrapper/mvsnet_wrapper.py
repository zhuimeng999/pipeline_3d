# -*- coding: UTF-8 -*-

import os
import subprocess
from third_party.colmap.read_write_model import read_model, write_model


def get_mvsnet_path() -> str:
    mvsnet_path = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MVSNet')))
    if os.path.exists(mvsnet_path):
        return mvsnet_path
    else:
        return ''


def export_colmap_to_mvsnet(output_dir):
    if os.path.exists(os.path.join(output_dir, 'sparse/cameras.txt')) is False:
        sparse_dir = os.path.join(output_dir, 'sparse')
        write_model(*read_model(sparse_dir, '.bin'), sparse_dir, '.txt')
    mvsnet_path = get_mvsnet_path()
    colmap2mvsnet_path = os.path.join(mvsnet_path, 'mvsnet/colmap2mvsnet.py')
    colmap2mvsnet_command_line = ['python', colmap2mvsnet_path,
                                  '--dense_folder', output_dir]
    subprocess.run(colmap2mvsnet_command_line, check=True)


def run_mvsnet_predict(options, output_dir):
    mvsnet_path = get_mvsnet_path()
    base_command = 'source ~/anaconda3/bin/activate mvsnet;'
    mvsnet_command_line = ['python', os.path.join(mvsnet_path, 'mvsnet/test.py'),
                           '--dense_folder', output_dir,
                           '--regularization', '3DCNNs',
                           '--max_w', '1152',
                           '--max_h', '864',
                           '--max_d', '192',
                           '--interval_scale', '1.06',
                           '--pretrained_model_ckpt_path', os.path.join(mvsnet_path, 'pretrain/tf_model_eth3d/3DCNNs/model.ckpt'),
                           '--ckpt_step', '150000']
    base_command = base_command + '"'+ '" "'.join(mvsnet_command_line)+'"'
    subprocess.run(['bash', '-c', base_command], start_new_session=True, cwd=os.path.join(mvsnet_path, 'mvsnet'))

def run_rmvsnet_predict(options, output_dir):
    mvsnet_path = get_mvsnet_path()
    base_command = 'source ~/anaconda3/bin/activate mvsnet;'
    mvsnet_command_line = ['python', os.path.join(mvsnet_path, 'mvsnet/test.py'),
                           '--dense_folder', output_dir,
                           '--regularization', 'GRU',
                           '--max_w', '1600',
                           '--max_h', '1200',
                           '--max_d', '256',
                           '--interval_scale', '0.8',
                           '--pretrained_model_ckpt_path', os.path.join(mvsnet_path, 'pretrain/tf_model_eth3d/GRU/model.ckpt'),
                           '--ckpt_step', '150000']
    base_command = base_command + '"'+ '" "'.join(mvsnet_command_line)+'"'
    subprocess.run(['bash', '-c', base_command], start_new_session=True, cwd=os.path.join(mvsnet_path, 'mvsnet'))
