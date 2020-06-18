# -*- coding: UTF-8 -*-

import os
import logging
import numpy as np
import struct
import cv2

from utils import LogThanExitIfFailed, InitLogging
from third_party.colmap.read_write_model import Camera, Image, Point3D, rotmat2qvec, write_model


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
            height, width, _= cv2.imread(os.path.join(image_dir, 'views', 'view_%04d.mve' % i, 'original.jpg')).shape
            max_dim = max(height, width)
            params = np.array(tuple(map(float, f.readline().split())))
            cameras[i] = Camera(id=i, model='RADIAL',
                                width=width,
                                height=height,
                                params=np.array((params[0]*max_dim, width/2. - 0.5, height/2 - 0.5, params[1], params[2])))
            row1 = np.array(tuple(map(float, f.readline().split())))
            row2 = np.array(tuple(map(float, f.readline().split())))
            row3 = np.array(tuple(map(float, f.readline().split())))
            qvec = rotmat2qvec(np.stack([row1, row2, row3]))
            tvec = np.array(tuple(map(float, f.readline().split())))
            images[i] = Image(
                id=i, qvec=qvec, tvec=tvec,
                camera_id=i, name="%4d" % i,
                xys=images_xys[i]*max_dim + np.array([width/2., height/2.]) - 0.5, point3D_ids=np.ones(len(images_xys[i]), dtype=np.int)*-1)

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


if __name__ == '__main__':
    InitLogging()
    load_mve_sfm('/home/lucius/data/workspace/Ignatius_mve')
