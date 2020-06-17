# -*- coding: UTF-8 -*-

import argparse
import logging
import os
import pathlib
import subprocess
import numpy as np

from sfm_run import get_sfm_parser
from third_party.colmap.read_write_model import read_model
from utils import InitLogging, LogThanExitIfFailed


class RadiaCamera:
    def __init__(self, f, cx, cy, radia1, radia2):
        self.radia2 = radia2
        self.radia1 = radia1
        self.cy = cy
        self.cx = cx
        self.f = f

    def __call__(self, p):
        r2 = p[0] * p[0] + p[1] * p[1]
        coef = 1 + r2 * self.radia1 + r2 * r2 * self.radia2
        ret = p * coef
        return np.array((ret[0] * self.f + self.cx, ret[1] * self.f + self.cy))


def ConvertTheiasfmToColmap(theiasfm_reconstruction_filename, sfm_colmap_path):
    theiasfm_to_colmap_command_line = ['export_colmap_files',
                                       '-input_reconstruction_file', theiasfm_reconstruction_filename,
                                       '-output_folder', sfm_colmap_path,
                                       '--logtostderr']
    subprocess.run(theiasfm_to_colmap_command_line, check=True)


def FindOrConvertSfmResultToColmap(options):
    sfm_model_path = os.path.join(options.sfm_path, 'sfm_colmap')
    if options.alg_type == 'colmap':
        if options.build_id is None:
            sparse_path = pathlib.Path(os.path.join(options.sfm_path, 'sparse'))
            LogThanExitIfFailed(len(list(sparse_path.iterdir())) != 1,
                                'there are many reconstruction resule in %s, you must special a build_id',
                                sparse_path.as_posix())
            sfm_model_path = os.path.join(options.sfm_path, 'sparse/0')
        else:
            sfm_model_path = os.path.join(options.sfm_path, 'sparse', str(options.build_id))
    elif os.path.isdir(sfm_model_path) is True:
        pass
    elif options.alg_type == 'openmvg':
        pass
    elif options.alg_type == 'theiasfm':
        os.mkdir(sfm_model_path)

        if options.build_id is None:
            reconstructions = list(pathlib.Path(options.sfm_path).glob('reconstruction.bin*'))
            LogThanExitIfFailed(len(reconstructions) != 1,
                                'there are many reconstruction resule in %s, you must special a build_id',
                                reconstructions)
            theiasfm_reconstruction_filename = str(reconstructions[0].absolute().as_posix())
        else:
            theiasfm_reconstruction_filename = os.path.join(options.sfm_path,
                                                            'reconstruction.bin-' + str(options.build_id))
        ConvertTheiasfmToColmap(theiasfm_reconstruction_filename, sfm_model_path)

    elif options.alg_type == 'mve':
        pass
    else:
        LogThanExitIfFailed(False, 'unknown algorithm type: %s', options.alg_type)
    return sfm_model_path


def format_camera(cameras):
    new_camers = {}
    for camera_id, camera in cameras.items():
        LogThanExitIfFailed(camera.model not in ['RADIAL', 'SIMPLE_RADIAL'], 'unsupport camera type: %s', camera.model)
        if len(camera.params) == 4:
            radia2 = 0
        else:
            radia2 = camera.params[4]
        new_camers[camera_id] = RadiaCamera(camera.params[0], camera.params[1], camera.params[2], camera.params[3],
                                            radia2)
    return new_camers


def ProjectPoint(camera, image, p_in):
    p = image.qvec2rotmat().dot(p_in) + image.tvec
    return camera(p[:2] / p[2]), p[2]


def PrintReprojectionErrors(cameras, images, points3D):
    reprojection_errors = []
    num_projections_behind_camera = 0

    for track_id, track in points3D.items():
        for view_id, feature_id in zip(track.image_ids, track.point2D_idxs):
            image = images[view_id]
            camera = cameras[image.camera_id]
            assert track_id == image.point3D_ids[feature_id]
            feature = image.xys[feature_id]

            projection, z = ProjectPoint(camera, image, track.xyz)
            if z < 0:
                num_projections_behind_camera = num_projections_behind_camera + 1;

            # Compute reprojection error.
            reprojection_error = np.linalg.norm(feature - projection)
            reprojection_errors.append(reprojection_error)

    if len(reprojection_errors) == 0:
        logging.info("No estimated 3d points were found. Cannot compute "
                     "reprojection error statistics.")
        return

    reprojection_errors.sort()
    mean_reprojection_error = sum(reprojection_errors) / len(reprojection_errors)
    median_reprojection_error = reprojection_errors[len(reprojection_errors) // 2]

    logging.info(
        "\nNum observations: %d\nNum reprojections behind camera: %d\nMean reprojection error = %f\nMedian reprojection_error = %f",
        len(reprojection_errors), num_projections_behind_camera, mean_reprojection_error, median_reprojection_error)


def PrintTrackLengthHistogram(cameras, images, points3D):
    track_lengths = []
    for track_id, track in points3D.items():
        track_lengths.append(len(track.image_ids))

    # Exit if there were no tracks found.
    if (len(track_lengths) == 0):
        logging.info("No valid tracks were present in the reconstruction.")

    # Compute the mean and median track lengths.
    mean_track_length = sum(track_lengths) / len(track_lengths)
    logging.info("Mean track length: %f", mean_track_length)

    # Sort the median track length.
    median_track_length = np.median(track_lengths)
    logging.info("Median track length: %d", median_track_length)

    # Display the track length histogram.
    histogram_bins = [2, 3, 4, 5, 6, 7, 8,
                      9, 10, 15, 20, 25, 50, max(track_lengths)]
    histogram = np.histogram(track_lengths, bins=histogram_bins, range=(0, 1000))
    logging.info("Track length histogram = \n%s", histogram)


if __name__ == '__main__':
    InitLogging()

    parser = argparse.ArgumentParser()
    parser.add_argument('sfm_path', help='sfm reconstruction result directory')
    parser.add_argument('--alg_type', default=None, choices=['colmap', 'openmvg', 'theiasfm', 'mve'])
    parser.add_argument('--build_id', default=None, type=int)
    options = parser.parse_args()
    if options.alg_type is None:
        with open(os.path.join(options.sfm_path, 'sfm_run_options.txt'), 'r') as f:
            line = map(lambda x: x.strip(), filter(lambda x: x.rstrip()[0] != '#', f))
            options.alg_type = get_sfm_parser().parse_args(list(line)).alg_type
            logging.info('use algorithm type get from sfm config file: %s', options.alg_type)

    sfm_model_path = FindOrConvertSfmResultToColmap(options)
    logging.info('find reconstruction on: %s', sfm_model_path)
    cameras, images, points3D = read_model(sfm_model_path, '.bin' if options.alg_type == 'colmap' else '.txt')
    logging.info('Num views: %d', len(images))
    logging.info('Num 3D points: %d', len(points3D))
    new_cameras = format_camera(cameras)
    PrintReprojectionErrors(new_cameras, images, points3D)
    PrintTrackLengthHistogram(new_cameras, images, points3D)
