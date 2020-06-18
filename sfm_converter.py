# -*- coding: UTF-8 -*-

import argparse
import logging
import os
import pathlib
import subprocess

from utils import InitLogging, LogThanExitIfFailed
from third_party.colmap.read_write_model import write_model, read_images_text, read_points3D_text, write_points3D_text
from load_mve_sfm import load_mve_sfm

def FixedOpenmvgToColmapError(sfm_colmap_dir):
    images = read_images_text(os.path.join(sfm_colmap_dir, 'images.txt'))
    points3Ds = read_points3D_text(os.path.join(sfm_colmap_dir, 'points3D.txt'))

    points3Ds_count = {}
    for track_id, track in points3Ds.items():
        points3Ds_count[track_id] = 0

    for image_id, image in images.items():
        for i in range(len(image.point3D_ids)):
            pid = image.point3D_ids[i]
            track = points3Ds[pid]
            cnt = points3Ds_count[pid]
            track.image_ids[cnt] = image_id
            track.point2D_idxs[cnt] = i
            points3Ds_count[pid] = points3Ds_count[pid] + 1
    for track_id, track in points3Ds.items():
        assert points3Ds_count[track_id] == len(track.point2D_idxs)

    write_points3D_text(points3Ds, os.path.join(sfm_colmap_dir, 'points3D.txt'))

def openmvg2colmap(openmvg_dir, colmap_dir, build_id:int =None):
    os.mkdir(colmap_dir)
    assert build_id is None
    openmvg_to_colmap_command_line = ['openMVG_main_openMVG2Colmap',
                                      '--sfmdata', os.path.join(openmvg_dir, 'sfm_data.bin'),
                                      '--outdir', colmap_dir]
    subprocess.run(openmvg_to_colmap_command_line, check=True)
    logging.info(
        'there are errors with openMVG_main_openMVG2Colmap, the 3D->2D map partialily wrong, we can fix this')
    FixedOpenmvgToColmapError(sfm_colmap_dir=colmap_dir)

def mve2colmap(mve_dir, colmap_dir):
    write_model(*load_mve_sfm(mve_dir), colmap_dir, '.txt')

def theiasfm2colmap(theiasfm_dir, colmap_dir, build_id:int =None):
    os.mkdir(colmap_dir)

    if build_id is None:
        reconstructions = list(pathlib.Path(theiasfm_dir).glob('reconstruction.bin*'))
        LogThanExitIfFailed(len(reconstructions) == 1,
                            'there are many reconstruction resule in %s, you must special a build_id',
                            reconstructions)
        theiasfm_reconstruction_filename = str(reconstructions[0].absolute().as_posix())
    else:
        theiasfm_reconstruction_filename = os.path.join(theiasfm_dir,
                                                        'reconstruction.bin-' + str(build_id))
    theiasfm_to_colmap_command_line = ['export_colmap_files',
                                       '-input_reconstruction_file', theiasfm_reconstruction_filename,
                                       '-output_folder', colmap_dir,
                                       '--logtostderr']
    subprocess.run(theiasfm_to_colmap_command_line, check=True)

if __name__ == '__main__':
    InitLogging()
    parser = argparse.ArgumentParser()