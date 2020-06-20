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


def run_mvsnet_predict(output_dir):
    mvsnet_path = get_mvsnet_path()
    mvsnet_test = os.path.join(mvsnet_path, 'mvsnet/test.py')
    base_command = 'source ~/anaconda3/bin/activate mvsnet;'
    base_command = base_command + 'poython ' + mvsnet_test
    subprocess.run('source ~/anaconda3/bin/activate mvsnet; python ' + mvsnet_test, shell=True)
