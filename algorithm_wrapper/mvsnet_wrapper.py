# -*- coding: UTF-8 -*-

import os
import pathlib
import logging
import subprocess
from third_party.colmap.read_write_model import read_model, write_model
from pipeline.common_options import GLOBAL_OPTIONS as FLAGS


def get_mvsnet_path() -> str:
    mvsnet_path = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MVSNet')))
    if os.path.exists(mvsnet_path):
        return mvsnet_path
    else:
        return ''

def get_fusibile_path() -> str:
    return os.path.join(get_mvsnet_path(), 'fusibile/fusibile')


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
    if FLAGS.mvs_view_num is not None:
        mvsnet_options.append('--view_num')
        mvsnet_options.append(str(FLAGS.mvs_view_num))
    return mvsnet_options


def export_colmap_to_mvsnet(output_dir):
    if os.path.exists(os.path.join(output_dir, 'sparse/cameras.txt')) is False:
        sparse_dir = os.path.join(output_dir, 'sparse')
        write_model(*read_model(sparse_dir, '.bin'), sparse_dir, '.txt')
    origin_images = list(os.listdir(os.path.join(output_dir, 'images')))

    mvsnet_path = get_mvsnet_path()
    colmap2mvsnet_path = os.path.join(mvsnet_path, 'mvsnet/colmap2mvsnet.py')
    colmap2mvsnet_command_line = ['python', colmap2mvsnet_path,
                                  '--dense_folder', output_dir]
    if FLAGS.mvs_max_d is not None:
        colmap2mvsnet_command_line.append('--max_d')
        colmap2mvsnet_command_line.append(str(FLAGS.mvs_max_d))
    logging.info('run colmap2mvsnet with command: %s', ' '.join(colmap2mvsnet_command_line))
    subprocess.run(colmap2mvsnet_command_line, check=True)
    for image in origin_images:
        os.remove(os.path.join(output_dir, 'images', image))



def run_mvsnet_predict(output_dir):
    mvsnet_path = get_mvsnet_path()
    base_command = 'source ~/anaconda3/bin/activate mvsnet;'
    mvsnet_command_line = ['python', os.path.join(mvsnet_path, 'mvsnet/test.py'),
                           '--dense_folder', output_dir,
                           '--regularization', '3DCNNs',
                           '--interval_scale', '1.06',
                           '--pretrained_model_ckpt_path',
                           os.path.join(mvsnet_path, 'pretrain/tf_model_blendedmvs/3DCNNs/model.ckpt'),
                           '--ckpt_step', '150000']

    base_command = base_command + '"' + '" "'.join(mvsnet_command_line + get_mvsnet_options()) + '"'
    logging.info('run mvsnet with command: %s', base_command)
    subprocess.run(['bash', '-c', base_command], cwd=os.path.join(mvsnet_path, 'mvsnet'), check=True)


def run_mvsnet_fuse(output_dir):
    mvsnet_path = get_mvsnet_path()
    base_command = 'source ~/anaconda3/bin/activate mvsnet;'
    mvsnet_fuse_command_line = ['python', os.path.join(mvsnet_path, 'mvsnet/depthfusion.py'),
                           '--dense_folder', os.path.join(output_dir, 'mvs_result'),
                           '--fusibile_exe_path', get_fusibile_path()]
    mvsnet_fuse_command_line.append('--prob_threshold')
    mvsnet_fuse_command_line.append(str(FLAGS.fuse_prob_threshold or (0.8 if FLAGS.mvs == 'mvsnet' else 0.3)))

    base_command = base_command + '"' + '" "'.join(mvsnet_fuse_command_line) + '"'
    subprocess.run(['bash', '-c', base_command], cwd=os.path.join(mvsnet_path, 'mvsnet'), check=True)
    subprocess.run(['rm', '-rf', os.path.join(output_dir, 'fused')], check=False)
    pointcloud_path = list(pathlib.Path(os.path.join(output_dir, 'mvs_result/points_mvsnet')).glob('**/*.ply'))
    if len(pointcloud_path) != 1:
        logging.critical('error %s', pointcloud_path)
        exit(1)
    new_fused = pathlib.Path(os.path.join(output_dir, 'mvsnet_fused.ply'))
    # if new_fused.is_file():
    #     new_fused.unlink()
    pointcloud_path[0].rename(new_fused)
    os.rename(os.path.join(output_dir, 'mvs_result/points_mvsnet'), os.path.join(output_dir, 'fused'))


def run_rmvsnet_predict(output_dir):
    mvsnet_path = get_mvsnet_path()
    base_command = 'source ~/anaconda3/bin/activate mvsnet;'
    mvsnet_command_line = ['python', os.path.join(mvsnet_path, 'mvsnet/test.py'),
                           '--dense_folder', output_dir,
                           '--regularization', 'GRU',
                           '--interval_scale', '0.8',
                           '--pretrained_model_ckpt_path',
                           os.path.join(mvsnet_path, 'pretrain/tf_model_blendedmvs/GRU/model.ckpt'),
                           '--ckpt_step', '150000']
    base_command = base_command + '"' + '" "'.join(mvsnet_command_line + get_mvsnet_options()) + '"'
    logging.info('run rmvsnet with command: %s', base_command)
    subprocess.run(['bash', '-c', base_command], cwd=os.path.join(mvsnet_path, 'mvsnet'), check=True)
