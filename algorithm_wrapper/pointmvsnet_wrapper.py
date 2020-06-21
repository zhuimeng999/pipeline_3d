# -*- coding: UTF-8 -*-

import os
import argparse
import subprocess
import pathlib
import cv2
import yaml


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


def run_pointmvsnet_predict(options, mvs_work_dir):
    config_filepath = os.path.join(mvs_work_dir, 'pointmvsnet_cfg.yaml')
    with open(os.path.join(os.path.dirname(__file__), '../data/dtu_wde3.yaml'), 'r') as f_in:
        with open(config_filepath, 'w') as f_out:
            params = yaml.load(f_in)
            params['DATA']['TEST']['ROOT_DIR'] = mvs_work_dir
            yaml.dump(params, f_out)
    pointmvsnet_path = get_pointmvsnet_path()
    base_command = 'source ~/anaconda3/bin/activate PointMVSNet;'
    mvsnet_command_line = ['python', 'pointmvsnet/test.py',
                           '--cfg', config_filepath]
    if options.num_gpu == 0:
        mvsnet_command_line.append('--cpu')
    mvsnet_command_line = mvsnet_command_line + ['TEST.WEIGHT', 'outputs/dtu_wde3/model_pretrained.pth']
    base_command = base_command + '"' + '" "'.join(mvsnet_command_line) + '"'
    subprocess.run(['bash', '-c', base_command], start_new_session=True, cwd=pointmvsnet_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', type=str, help='woring directory')
    options = parser.parse_args()
    fix_mvsnet_to_pointmvsnet(options.work_dir)