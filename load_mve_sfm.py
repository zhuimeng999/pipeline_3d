# -*- coding: UTF-8 -*-

import os
import logging
import numpy as np
import struct
import cv2

from utils import LogThanExitIfFailed, InitLogging
from third_party.colmap.read_write_model import Camera, Image, Point3D, rotmat2qvec, read_model


def read_bundler(filepath, images_xys, format='PHOTOSYNTHER'):
    cameras = {}
    images = {}
    points3D = {}
    image_dir = os.path.dirname(filepath)
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        if format == 'PHOTOSYNTHER':
            LogThanExitIfFailed(line == 'drews 1.0', 'wrong format type: expect "drews 1.0", got %s', line)
        elif format == 'BUNDLER':
            LogThanExitIfFailed(line == '# Bundle file v0.3', 'wrong format type: expect "# Bundle file v0.3", got %s',
                                line)
        else:
            LogThanExitIfFailed(False, 'wrong format type')

        num_images, num_points = map(int, f.readline().split())

        for i in range(num_images):
            height, width, _ = cv2.imread(os.path.join(image_dir, 'views', 'view_%04d.mve' % i, 'original.jpg')).shape
            max_dim = max(height, width)
            params = np.array(tuple(map(float, f.readline().split())))
            cameras[i] = Camera(id=i, model='RADIAL',
                                width=width,
                                height=height,
                                params=np.array(
                                    (params[0] * max_dim, width / 2. - 0.5, height / 2 - 0.5, params[1], params[2])))
            row1 = np.array(tuple(map(float, f.readline().split())))
            row2 = np.array(tuple(map(float, f.readline().split())))
            row3 = np.array(tuple(map(float, f.readline().split())))
            qvec = rotmat2qvec(np.stack([row1, row2, row3]))
            tvec = np.array(tuple(map(float, f.readline().split())))
            images[i] = Image(
                id=i, qvec=qvec, tvec=tvec,
                camera_id=i, name="%04d.jpg" % i,
                xys=images_xys[i] * max_dim + np.array([width / 2., height / 2.]) - 0.5,
                point3D_ids=np.ones(len(images_xys[i]), dtype=np.int) * -1)

        for i in range(num_points):
            xyz = np.array(tuple(map(float, f.readline().split())))
            rgb = np.array(tuple(map(int, f.readline().split())))
            elems = f.readline().split()
            image_ids = np.array(tuple(map(int, elems[1::3])))
            point2D_idxs = np.array(tuple(map(int, elems[2::3])))
            track_lengths = int(elems[0])
            assert track_lengths == len(image_ids)
            assert track_lengths == len(point2D_idxs)
            points3D[i] = Point3D(id=i, xyz=xyz, rgb=rgb,
                                  error=0., image_ids=image_ids,
                                  point2D_idxs=point2D_idxs)
            for image_id, point2D_id in zip(image_ids, point2D_idxs):
                images[image_id].point3D_ids[point2D_id] = i
        left = f.read()
        assert (left is None) or (left == '')
    assert len(images_xys) == len(images)
    return cameras, images, points3D


def read_mve_feature(filepath):
    PREBUNDLE_SIGNATURE = b"MVE_PREBUNDLE\n"
    integer_struct = struct.Struct('i')
    images_xys = []
    with open(filepath, 'rb') as f:
        assert f.read(len(PREBUNDLE_SIGNATURE)) == PREBUNDLE_SIGNATURE
        num_views = integer_struct.unpack(f.read(integer_struct.size))[0]

        vec_struct = struct.Struct('ff')
        colol_struct = struct.Struct('BBB')
        for i in range(num_views):
            num_positions = integer_struct.unpack(f.read(integer_struct.size))[0]
            xys = np.array(list(vec_struct.iter_unpack(f.read(vec_struct.size * num_positions))), dtype=np.float)
            images_xys.append(xys)
            num_colors = integer_struct.unpack(f.read(integer_struct.size))[0]
            for color in colol_struct.iter_unpack(f.read(colol_struct.size * num_colors)):
                pass

        num_pairs = integer_struct.unpack(f.read(integer_struct.size))[0]
        match_struct = struct.Struct('iii')
        correspondence_struct = struct.Struct('ii')
        for i in range(num_pairs):
            _, _, num_matches = match_struct.unpack(f.read(match_struct.size))
            for c in correspondence_struct.iter_unpack(f.read(correspondence_struct.size * num_matches)):
                pass
        left = f.read()
        assert (left is None) or (left == b'')
    return images_xys


def load_mve_sfm(sfm_mve_dir):
    images_xys = read_mve_feature(os.path.join(sfm_mve_dir, 'view/prebundle.sfm'))
    return read_bundler(os.path.join(sfm_mve_dir, 'view/synth_0.out'), images_xys)


def save_mve_sfm(sfm_colmap_dir, image_dir, sfm_mve_dir):
    mve_meta_template = '''# MVE view meta data is stored in INI-file syntax.
# This file is generated, formatting will get lost.

[camera]
focal_length = %f
pixel_aspect = 1
principal_point = %f %f
rotation =%s
translation =%s 

[view]
id = %d
name = %s'''
    mve_dir = os.path.join(sfm_mve_dir, 'view')
    os.mkdir(mve_dir)
    view_dir = os.path.join(mve_dir, 'views')
    os.mkdir(view_dir)

    cameras, images, points3D = read_model(sfm_colmap_dir, '.bin')

    with open(os.path.join(mve_dir, 'synth_0.out'), 'w') as f:
        f.write('drews 1.0\n')
        f.write('%d %d\n' % (len(images), len(points3D)))
        new_image_id = {}
        cnt = 0
        new_images = sorted(list(images.items()), key= lambda x: x[0])
        for image_id, image in new_images:
            camera = cameras[image.camera_id]
            max_dim = max(camera.width, camera.height)

            f.write('%f %f %f\n' % (camera.params[0] / max_dim, 0., 0.))
            R_matrix = image.qvec2rotmat()
            f.write('%f %f %f\n' % (R_matrix[0, 0], R_matrix[0, 1], R_matrix[0, 2]))
            f.write('%f %f %f\n' % (R_matrix[1, 0], R_matrix[1, 1], R_matrix[1, 2]))
            f.write('%f %f %f\n' % (R_matrix[2, 0], R_matrix[2, 1], R_matrix[2, 2]))
            f.write('%f %f %f\n' % (image.tvec[0], image.tvec[1], image.tvec[2]))
            new_image_id[image_id] = cnt
            cnt = cnt + 1

        for _, point3D in points3D.items():
            f.write('%f %f %f\n' % (point3D.xyz[0], point3D.xyz[1], point3D.xyz[2]))
            f.write('%d %d %d\n' % (point3D.rgb[0], point3D.rgb[1], point3D.rgb[2]))
            f.write('%d' % len(point3D.image_ids))
            for image_id, point2D_id in zip(point3D.image_ids, point3D.point2D_idxs):
                f.write(' %d %d 0' % (new_image_id[image_id], point2D_id))
            f.write('\n')
        for image_id, image in new_images:
            camera = cameras[image.camera_id]
            max_dim = max(camera.width, camera.height)
            if image is None:
                continue

            view_image_dir = os.path.join(view_dir, 'view_%04d.mve' % new_image_id[image_id])
            os.mkdir(view_image_dir)
            os.rename(os.path.join(image_dir, image.name), os.path.join(view_image_dir, 'undistorted.png'))

            R_str = ''
            for v in np.nditer(image.qvec2rotmat()):
                R_str = R_str + ' ' + str(v)
            T_str = ''
            for v in np.nditer(image.tvec):
                T_str = T_str + ' ' + str(v)

            assert camera.model == 'PINHOLE'
            assert abs(camera.params[0] - camera.params[1]) < 1e-8
            with open(os.path.join(view_image_dir, 'meta.ini'), 'w') as f:
                f.write(mve_meta_template % (
                camera.params[0] / max_dim, camera.params[2] / camera.width, camera.params[3] / camera.height,
                R_str, T_str, new_image_id[image_id], image.name))


if __name__ == '__main__':
    InitLogging()
    load_mve_sfm('/home/lucius/data/workspace/Ignatius_mve')
