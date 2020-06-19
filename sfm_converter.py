# -*- coding: UTF-8 -*-

import argparse
import logging
import os
import sys
import pathlib
import subprocess

from utils import InitLogging, LogThanExitIfFailed, GetFileFromBuildId
from third_party.colmap.read_write_model import write_model, read_images_text, read_points3D_text, write_points3D_text
from load_mve_sfm import load_mve_sfm, save_mve_sfm
from common_options import get_common_options_parser
from algorithm_wrapper.mvsnet_wrapper import export_colmap_to_mvsnet


def create_colmap_sparse_directory(base_dir):
    if os.path.exists(base_dir) is False:
        os.mkdir(base_dir)
    tmp_work_dir = os.path.join(base_dir, 'sparse')
    if os.path.exists(tmp_work_dir) is False:
        os.mkdir(tmp_work_dir)
    tmp_work_dir = os.path.join(tmp_work_dir, '0')
    if os.path.exists(tmp_work_dir) is False:
        os.mkdir(tmp_work_dir)
    return tmp_work_dir


def colmap2colmap(in_colmap_dir, in_images_dir, out_colmap_dir, build_id: int = None):
    select_dir = GetFileFromBuildId(os.path.join(in_colmap_dir, 'sparse'), "*", build_id)
    colmap_undistored_command_line = ['colmap', 'image_undistorter',
                                      '--image_path', in_images_dir,
                                      '--input_path', select_dir,
                                      '--output_path', out_colmap_dir]
    subprocess.run(colmap_undistored_command_line, check=True)


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


def mve2others(in_mve_dir, in_images_dir, out_others_dir, build_id: int = None, out_type='colmap'):
    assert build_id is None
    distorted_convert_dir = os.path.join(out_others_dir, 'tmp')
    tmp_work_dir = create_colmap_sparse_directory(distorted_convert_dir)
    cameras, images, points3D = load_mve_sfm(in_mve_dir)
    write_model(cameras, images, points3D, tmp_work_dir, '.txt')
    original_images_dir = os.path.join(distorted_convert_dir, 'original_images')
    os.mkdir(original_images_dir)
    for image_id, image in images.items():
        original_image_path = os.path.join(in_mve_dir, 'view/views', 'view_%04d.mve' % image_id, 'original.jpg')
        os.symlink(os.path.abspath(original_image_path), os.path.join(original_images_dir, image.name))

    sfm_convert_helper('colmap', out_type, distorted_convert_dir, original_images_dir, out_others_dir)


def openmvg2colmap(in_openmvg_dir, in_images_dir, out_colmap_dir, build_id: int = None):
    distorted_convert_dir = os.path.join(out_colmap_dir, 'tmp')
    tmp_work_dir = create_colmap_sparse_directory(distorted_convert_dir)
    assert build_id is None
    openmvg_to_colmap_command_line = ['openMVG_main_openMVG2Colmap',
                                      '--sfmdata', os.path.join(in_openmvg_dir, 'sfm_data.bin'),
                                      '--outdir', tmp_work_dir]
    subprocess.run(openmvg_to_colmap_command_line, check=True)
    logging.info(
        'there are errors with openMVG_main_openMVG2Colmap, the 3D->2D map partialily wrong, we can fix this')
    FixedOpenmvgToColmapError(sfm_colmap_dir=tmp_work_dir)
    colmap2colmap(distorted_convert_dir, in_images_dir, out_colmap_dir)


def theiasfm2colmap(in_theiasfm_dir, in_images_dir, out_colmap_dir, build_id: int = None):
    select_file = GetFileFromBuildId(in_theiasfm_dir, "reconstruction.bin*", build_id)
    distorted_convert_dir = os.path.join(out_colmap_dir, 'tmp')
    tmp_work_dir = create_colmap_sparse_directory(distorted_convert_dir)
    theiasfm_to_colmap_command_line = ['export_colmap_files',
                                       '-input_reconstruction_file', select_file,
                                       '-output_folder', tmp_work_dir,
                                       '--logtostderr']
    subprocess.run(theiasfm_to_colmap_command_line, check=True)
    colmap2colmap(distorted_convert_dir, in_images_dir, out_colmap_dir)


def mve2colmap(in_mve_dir, in_images_dir, out_colmap_dir, build_id: int = None):
    mve2others(in_mve_dir, in_images_dir, out_colmap_dir, build_id=build_id, out_type='colmap')


def colmap2openmvs(in_colmap_dir, in_images_dir, out_openmvs_dir, build_id: int = None):
    colmap2colmap(in_colmap_dir, in_images_dir, out_openmvs_dir, build_id)
    interface_colmap_command_line = ['InterfaceCOLMAP',
                                     '--working-folder', out_openmvs_dir,
                                     '--input-file', out_openmvs_dir,
                                     '--image-folder', os.path.join(out_openmvs_dir, 'images'),
                                     '--output-file', os.path.join(out_openmvs_dir, 'scene.mvs')]
    subprocess.run(interface_colmap_command_line, check=True)


def openmvg2openmvs(in_openmvg_dir, in_images_dir, out_openmvs_dir, build_id: int = None):
    openmvg2openmvs_command_line = ['openMVG_main_openMVG2openMVS',
                                    '--sfmdata', os.path.join(in_openmvg_dir, 'sfm_data.bin'),
                                    '--outfile', os.path.join(out_openmvs_dir, 'scene.mvs'),
                                    '--outdir', os.path.join(out_openmvs_dir, 'images')]
    subprocess.run(openmvg2openmvs_command_line, check=True)


def theiasfm2openmvs(in_theiasfm_dir, in_images_dir, out_openmvs_dir, build_id: int = None):
    theiasfm2colmap(in_theiasfm_dir, in_images_dir, out_openmvs_dir, build_id)
    interface_colmap_command_line = ['InterfaceCOLMAP',
                                     '--working-folder', out_openmvs_dir,
                                     '--input-file', out_openmvs_dir,
                                     '--image-folder', os.path.join(out_openmvs_dir, 'images'),
                                     '--output-file', os.path.join(out_openmvs_dir, 'scene.mvs')]
    subprocess.run(interface_colmap_command_line, check=True)


def mve2openmvs(in_mve_dir, in_images_dir, out_openmvs_dir, build_id: int = None):
    mve2colmap(in_mve_dir, in_images_dir, out_openmvs_dir, build_id)
    interface_colmap_command_line = ['InterfaceCOLMAP',
                                     '--working-folder', out_openmvs_dir,
                                     '--input-file', out_openmvs_dir,
                                     '--image-folder', os.path.join(out_openmvs_dir, 'images'),
                                     '--output-file', os.path.join(out_openmvs_dir, 'scene.mvs')]
    subprocess.run(interface_colmap_command_line, check=True)


def colmap2pmvs(in_colmap_dir, in_images_dir, out_pmvs_dir, build_id: int = None):
    select_dir = GetFileFromBuildId(os.path.join(in_colmap_dir, 'sparse'), "*", build_id)
    colmap_undistored_command_line = ['colmap', 'image_undistorter',
                                      '--image_path', in_images_dir,
                                      '--input_path', select_dir,
                                      '--output_path', out_pmvs_dir,
                                      '--output_type', 'PMVS']
    subprocess.run(colmap_undistored_command_line, check=True)


def openmvg2pmvs(in_openmvg_dir, in_images_dir, out_openmvs_dir, build_id: int = None):
    openmvg2openmvs_command_line = ['openMVG_main_openMVG2PMVS',
                                    '--sfmdata', os.path.join(in_openmvg_dir, 'sfm_data.bin'),
                                    '--outdir', out_openmvs_dir]
    subprocess.run(openmvg2openmvs_command_line, check=True)
    logging.info('change generate file name in order to keep colmap consistence')
    gen_pmvs_dir = os.path.join(out_openmvs_dir, 'pmvs')
    os.rename(os.path.join(out_openmvs_dir, 'PMVS'), gen_pmvs_dir)
    os.rename(os.path.join(gen_pmvs_dir, 'pmvs_options.txt'), os.path.join(gen_pmvs_dir, 'option-all'))


def theiasfm2pmvs(in_theiasfm_dir, in_images_dir, out_pmvs_dir, build_id: int = None):
    tmp = pathlib.Path(in_images_dir)
    images = None
    for f in tmp.iterdir():
        if f.suffix in ['.JPG', '.jpg', '.PNG', '.png'] and f.is_file():
            images = '*' + f.suffix
            break
    assert isinstance(images, str)
    select_file = GetFileFromBuildId(in_theiasfm_dir, "reconstruction.bin*", build_id)
    gen_pmvs_dir = os.path.join(out_pmvs_dir, 'pmvs')
    theiasfm2pmvs_command_line = ['export_reconstruction_to_pmvs',
                                  '-images', os.path.join(in_images_dir, images),
                                  '-pmvs_working_directory', gen_pmvs_dir,
                                  '-reconstruction', select_file,
                                  '--logtostderr']
    subprocess.run(theiasfm2pmvs_command_line, check=True)
    with open(os.path.join(gen_pmvs_dir, 'pmvs_options.txt'), 'r') as f_in:
        with open(os.path.join(gen_pmvs_dir, 'option-all'), 'w') as f_out:
            f_out.write(f_in.read().replace('CPU 1', 'CPU ' + str(os.cpu_count())))
    os.remove(os.path.join(gen_pmvs_dir, 'pmvs_options.txt'))


def mve2pmvs(in_mve_dir, in_images_dir, out_pmvs_dir, build_id: int = None):
    mve2others(in_mve_dir, in_images_dir, out_pmvs_dir, build_id=build_id, out_type='pmvs')


def colmap2mve(in_colmap_dir, in_images_dir, out_mve_dir, build_id: int = None):
    colmap2colmap(in_colmap_dir, in_images_dir, out_mve_dir, build_id)
    save_mve_sfm(os.path.join(out_mve_dir, 'sparse'), os.path.join(out_mve_dir, 'images'), out_mve_dir)

def openmvg2mve(in_openmvg_dir, in_images_dir, out_mve_dir, build_id: int = None):
    openmvg2mve_command_line = ['openMVG_main_openMVG2MVE2',
                                '--sfmdata', os.path.join(in_openmvg_dir, 'sfm_data.bin'),
                                '--outdir', out_mve_dir]
    subprocess.run(openmvg2mve_command_line, check=True)
    logging.info('change generate directory name in order to keep colmap consistence')
    os.rename(os.path.join(out_mve_dir, 'MVE'), os.path.join(out_mve_dir, 'view'))


def theiasfm2mve(in_theiasfm_dir, in_images_dir, out_mve_dir, build_id: int = None):
    theiasfm2colmap(in_theiasfm_dir, in_images_dir, out_mve_dir, build_id)
    save_mve_sfm(os.path.join(out_mve_dir, 'sparse'), os.path.join(out_mve_dir, 'images'), out_mve_dir)

def mve2mve(in_mve_dir, in_images_dir, out_mve_dir, build_id: int = None):
    os.rmdir(out_mve_dir)
    os.symlink(in_mve_dir, out_mve_dir)


def sfm_convert_helper(src_alg, target_alg, in_alg_dir, in_images_dir, out_alg_dir, build_id: int = None):
    this_module = sys.modules[__name__]
    if target_alg in ['mvsnet', 'rmvsnet', 'pointmvsnet']:
        convert_fun = getattr(this_module, src_alg + '2' + 'colmap')
        convert_fun(in_alg_dir, in_images_dir, out_alg_dir, build_id)
        export_colmap_to_mvsnet(out_alg_dir)
    else:
        convert_fun = getattr(this_module, src_alg + '2' + target_alg)
        convert_fun(in_alg_dir, in_images_dir, out_alg_dir, build_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_common_options_parser()])
    parser.add_argument('input_dir', type=str, help='input directory')
    parser.add_argument('images_dir', type=str, help='images directory')
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('--sfm', default='colmap', choices=['colmap', 'openmvg', 'theiasfm', 'mve'],
                        help='sfm algorithm')
    parser.add_argument('--mvs', default='colmap', choices=['colmap', 'openmvs', 'pmvs', 'cmvs', 'mve'],
                        help='mvs algorithm')
    options = parser.parse_args()
    InitLogging()

    sfm_convert_helper(options.sfm, options.mvs, options.input_dir, options.images_dir, options.output_dir)

