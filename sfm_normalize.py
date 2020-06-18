# -*- coding: UTF-8 -*-

import argparse
import logging
import os
import pathlib
import subprocess

from utils import InitLogging, GetFileFromBuildId

def sfm_normalize_colmap(old_colmap_dir, new_colmap_dir, image_path, build_id:int = None):
    select_dir = GetFileFromBuildId(os.path.join(old_colmap_dir, 'sparse'), "*", build_id)
    colmap_undistored_command_line = ['colmap', 'image_undistorter',
                                      '--image_path', image_path,
                                      '--input_path', select_dir,
                                      '--output_path', new_colmap_dir]
    subprocess.run(colmap_undistored_command_line)

def normalize_mve(mve_dir):
    logging.info('their is no need to run mve normalize, since it is already normalized')
    pass

if __name__ == '__main__':
    InitLogging()
    parser = argparse.ArgumentParser