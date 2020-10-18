# -*- coding: UTF-8 -*-

import os
import pathlib
import argparse
import numpy as np
import logging
import math
from tqdm import tqdm
import collections
from third_party.colmap.read_write_model import read_model
from pipeline.utils import LogThanExitIfFailed, InitLogging
import pickle


class RadiaCamera:
    def __init__(self, f, cx, cy, radia1, radia2):
        self.radia2 = radia2
        self.radia1 = radia1
        self.cy = cy
        self.cx = cx
        self.f = f
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)

    def __call__(self, p):
        r2 = p[0] * p[0] + p[1] * p[1]
        coef = 1.0 + r2 * self.radia1 + r2 * r2 * self.radia2
        ret = p * coef
        return np.array((ret[0] * self.f + self.cx, ret[1] * self.f + self.cy))


class PinholeCamera:
    def __init__(self, fx, fy, cx, cy):
        self.cy = cy
        self.cx = cx
        self.fx = fx
        self.fy = fy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)

    def __call__(self, p):
        return np.array((p[0] * self.fx + self.cx, p[1] * self.fy + self.cy))


class OpencvCamera:
    def __init__(self, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6

    def __call__(self, p):
        u2 = p[0] * p[0]
        v2 = p[1] * p[1]
        r2 = u2 + v2
        r4 = r2 * r2
        r6 = r4 * r4
        uv = p[0] * p[1]
        coef = (1.0 + r2 * self.k1 + r4 * self.k2 + r6 * self.k3) / (1.0 + r2 * self.k4 + r4 * self.k5 + r6 * self.k6)
        delta1 = 2 * self.p1 * uv + (r2 + 2. * u2) * self.p2
        delta2 = 2 * self.p2 * uv + (r2 + 2. * v2) * self.p1
        ret = p * coef
        p[0] = p[0] + delta1
        p[1] = p[1] + delta2
        return np.array((ret[0] * self.fx + self.cx, ret[1] * self.fy + self.cy))


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
        openmvg2colmap(options.sfm_path, sfm_model_path)
    elif options.alg_type == 'theiasfm':
        theiasfm2colmap(options.sfm_path, sfm_model_path)
    elif options.alg_type == 'mve':
        mve2colmap(options.sfm_path, sfm_model_path)
    else:
        LogThanExitIfFailed(False, 'unknown algorithm type: %s', options.alg_type)
    return sfm_model_path


def format_camera(cameras):
    new_camers = {}
    for camera_id, camera in cameras.items():
        if camera.model in ['RADIAL', 'SIMPLE_RADIAL']:
            if len(camera.params) == 4:
                radia2 = 0
            else:
                radia2 = camera.params[4]
            new_camers[camera_id] = RadiaCamera(camera.params[0], camera.params[1], camera.params[2], camera.params[3],
                                                radia2)
        elif camera.model in ['PINHOLE', 'SIMPLE_PINHOLE']:
            if len(camera.params) == 4:
                new_camers[camera_id] = PinholeCamera(camera.params[0], camera.params[1], camera.params[2],
                                                      camera.params[3])
            else:
                assert len(camera.params) == 3
                new_camers[camera_id] = PinholeCamera(camera.params[0], camera.params[0], camera.params[1],
                                                      camera.params[2])
        elif camera.model in ['FULL_OPENCV']:
            new_camers[camera_id] = OpencvCamera(*camera.params)
        else:
            LogThanExitIfFailed(False, 'unsupport camera type: %s', camera.model)
        new_camers[camera_id].height = camera.height
        new_camers[camera_id].width = camera.width

    return new_camers


def format_images(images):
    for image_id, image in images.items():
        image.mat = image.qvec2rotmat()
        image.centor = -image.mat.dot(image.tvec)


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
            # feature_id = np.where(image.point3D_ids == track_id)
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
    LogThanExitIfFailed(len(track_lengths) > 0,
                        "No valid tracks were present in the reconstruction.")

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


def colmap_view_select(cameras, images, points3D):
    point_shared = {}
    angles = {}
    for image_id, image in images.items():
        point_shared[image_id] = {}
        angles[image_id] = {}
        for image_id_y, image in images.items():
            point_shared[image_id][image_id_y] = 0
            angles[image_id][image_id_y] = []

    for track_id, track in tqdm(points3D.items()):
        for id_x in track.image_ids:
            for id_y in track.image_ids:
                if id_x == id_y:
                    continue
                d1 = np.square(track.xyz - images[id_x].centor).sum()
                d2 = np.square(track.xyz - images[id_y].centor).sum()
                d3 = np.square(images[id_x].centor - images[id_y].centor).sum()
                tmp = 2. * np.sqrt(d1*d2)
                if tmp == 0.:
                    continue
                angle = np.abs(np.arccos((d1 + d2 - d3)/tmp))
                angle = np.min([angle, math.pi - angle])
                point_shared[id_x][id_y] += 1
                angles[id_x][id_y].append(angle)
                point_shared[id_y][id_x] += 1
                angles[id_y][id_x].append(angle)

    for image_id, image in images.items():
        for image_id_y, image in images.items():
            if image_id == image_id_y:
                continue
            if len(angles[image_id][image_id_y]) < 100:
                point_shared[image_id][image_id_y] = 0
                continue
            percentile = int(np.round(75*len(angles[image_id][image_id_y])/100))
            angle = np.partition(angles[image_id][image_id_y], percentile)[percentile]
            if angle < (1.*math.pi/180.):
                point_shared[image_id][image_id_y] = 0

    with open('/tmp/colmap_dump.pikle', 'wb') as f:
        pickle.dump(point_shared, f)


def mvsnet_view_select(cameras, images, points3D):
    point_shared = {}
    for image_id, image in images.items():
        point_shared[image_id] = {}
        for image_id_y, image in images.items():
            point_shared[image_id][image_id_y] = 0.

    for track_id, track in tqdm(points3D.items()):
        for id_x in track.image_ids:
            for id_y in track.image_ids:
                if id_x == id_y:
                    continue
                d1 = np.square(track.xyz - images[id_x].centor).sum()
                d2 = np.square(track.xyz - images[id_y].centor).sum()
                d3 = np.square(images[id_x].centor - images[id_y].centor).sum()
                tmp = 2. * np.sqrt(d1*d2)
                if tmp == 0.:
                    continue
                angle = np.abs(np.arccos((d1 + d2 - d3)/tmp))
                angle = 180. * np.min([angle, math.pi - angle])/math.pi
                if angle < 5:
                    kernel = (angle - 5.)/1.
                else:
                    kernel = (angle - 5.)/10.
                score = math.exp(-kernel*kernel/2)
                point_shared[id_x][id_y] += score
                point_shared[id_y][id_x] += score

    with open('/tmp/mvsnet_dump.pikle', 'wb') as f:
        pickle.dump(point_shared, f)


def disparity_compute(cameras, images, points3D):
    with open('/tmp/mvsnet_dump.pikle', 'rb') as f:
        point_shared = pickle.load(f)
    for image_id, image in images.items():
        camera = cameras[image.camera_id]
        order_map = sorted(point_shared[image_id].items(), key = lambda x: x[1], reverse=True)
        for i in range(10):
            image_id_y = order_map[i][0]
            image_y = images[image_id_y]
            R = image_y.mat.dot(image.mat.T)
            T = image_y.tvec - R.dot(image.tvec)
            centor = T[:2]/T[2]

            n3 = []
            d = []
            for h in range(camera.height):
                for w in range(camera.width):
                    ref_pos = camera.K_inv.dot([w + 0.5, h + 0.5, 1.])
                    ref_pos = ref_pos/np.linalg.norm(ref_pos)
                    src_pos = R.dot(ref_pos)
                    projection = src_pos[:2]/src_pos[2]
                    direction =
                    distance = np.linalg.norm(centor - projection)
                    n3.append(src_pos[2])
                    d.append(distance)
            print('size: ', camera.width/camera.fx, ' ', camera.height/camera.fy, ' resolution: ', 1./camera.fx)
            print('n3: ', np.min(n3), ' ', np.max(n3), ' ', np.median(n3), ' ', np.mean(n3))
            print('d: ', np.min(d), ' ', np.max(d), ' ', np.median(d), ' ', np.mean(d))


if __name__ == '__main__':
    InitLogging()

    parser = argparse.ArgumentParser()
    parser.add_argument('sfm_path', help='sfm reconstruction result directory')
    options = parser.parse_args()

    cameras, images, points3D = read_model(options.sfm_path, '.bin')
    logging.info('Num views: %d', len(images))
    logging.info('Num 3D points: %d', len(points3D))
    new_cameras = format_camera(cameras)
    format_images(images)
    # PrintReprojectionErrors(new_cameras, images, points3D)
    # PrintTrackLengthHistogram(new_cameras, images, points3D)
    # colmap_view_select(new_cameras, images, points3D)
    # mvsnet_view_select(new_cameras, images, points3D)
    disparity_compute(new_cameras, images, points3D)
