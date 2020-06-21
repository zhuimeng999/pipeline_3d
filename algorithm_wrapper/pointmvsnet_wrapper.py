# -*- coding: UTF-8 -*-

import os
import argparse
import subprocess
import pathlib
import cv2
import yaml

from common_options import GLOBAL_OPTIONS as FLAGS
from algorithm_wrapper.mvsnet_wrapper import get_fusibile_path


def get_pointmvsnet_path() -> str:
    pointmvsnet_path = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PointMVSNet')))
    if os.path.exists(pointmvsnet_path):
        return pointmvsnet_path
    else:
        return ''


def fix_mvsnet_to_pointmvsnet(dataset_dir):
    os.symlink(os.path.join(dataset_dir, 'cams'), os.path.join(dataset_dir, 'Cameras'))
    images_dir = os.path.join(dataset_dir, 'Eval')
    os.mkdir(images_dir)
    images_dir = os.path.join(images_dir, 'Rectified')
    os.mkdir(images_dir)
    images_dir = os.path.join(images_dir, 'scan1')
    os.mkdir(images_dir)
    original_image_dir = os.path.join(dataset_dir, 'images')
    for image_path in pathlib.Path(original_image_dir).glob('*.jpg'):
        basename = os.path.basename(image_path.stem)
        if len(basename) != 8:
            continue
        idx = int(basename)
        image_data = cv2.imread(image_path.absolute().as_posix())
        cv2.imwrite(os.path.join(images_dir, 'rect_{:03d}_3_r5000.png'.format(idx + 1)), image_data)
    os.symlink(os.path.join(dataset_dir, 'pair.txt'), os.path.join(dataset_dir, 'Cameras/pair.txt'))


def run_pointmvsnet_predict(mvs_work_dir):
    config_filepath = os.path.join(mvs_work_dir, 'pointmvsnet_cfg.yaml')
    with open(os.path.join(os.path.dirname(__file__), '../data/dtu_wde3.yaml'), 'r') as f_in:
        with open(config_filepath, 'w') as f_out:
            params = yaml.load(f_in)
            params['DATA']['TEST']['ROOT_DIR'] = mvs_work_dir
            params['OUTPUT_DIR'] = os.path.join(mvs_work_dir, 'depths')
            if FLAGS.mvs_max_w is not None:
                params['DATA']['TEST']['IMG_WIDTH'] = FLAGS.mvs_max_w
            if FLAGS.mvs_max_h is not None:
                params['DATA']['TEST']['IMG_HEIGHT'] = FLAGS.mvs_max_h
            if FLAGS.mvs_max_d is not None:
                params['DATA']['TEST']['NUM_VIRTUAL_PLANE'] = FLAGS.mvs_max_d
            yaml.dump(params, f_out)
    pointmvsnet_path = get_pointmvsnet_path()
    base_command = 'source ~/anaconda3/bin/activate PointMVSNet;'
    mvsnet_command_line = ['python', 'pointmvsnet/test.py',
                           '--cfg', config_filepath]
    if FLAGS.num_gpu == 0:
        mvsnet_command_line.append('--cpu')
    mvsnet_command_line = mvsnet_command_line + ['TEST.WEIGHT', 'outputs/dtu_wde3/model_pretrained.pth']
    base_command = base_command + '"' + '" "'.join(mvsnet_command_line) + '"'
    subprocess.run(['bash', '-c', base_command], cwd=pointmvsnet_path, check=True)


def run_pointmvsnet_fuse(output_dir):
    pointmvsnet_path = get_pointmvsnet_path()
    base_command = 'source ~/anaconda3/bin/activate PointMVSNet;'
    mvsnet_fuse_command_line = ['python', os.path.join(pointmvsnet_path, 'tools/depthfusion.py'),
                                '--eval_folder', os.path.join(output_dir, 'mvs_result'),
                                '--fusibile_exe_path', get_fusibile_path(),
                                '-f', 'depths',
                                '-n', 'flow3']

    base_command = base_command + '"' + '" "'.join(mvsnet_fuse_command_line) + '"'
    subprocess.run(['bash', '-c', base_command], env={'PYTHONPATH': pointmvsnet_path}, cwd=pointmvsnet_path, check=True)


if __name__ == '__main__':
    FLAGS.add_argument('work_dir', type=str, help='woring directory')
    FLAGS.parse_args()
    fix_mvsnet_to_pointmvsnet(FLAGS.work_dir)
