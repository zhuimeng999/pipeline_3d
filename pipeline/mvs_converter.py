# -*- coding: UTF-8 -*-

import os
import sys

from pipeline.utils import InitLogging, mvs_network_check
from pipeline.common_options import GLOBAL_OPTIONS as FLAGS


def mvs_colmap2colmap(in_colmap_dir, out_colmap_dir, build_id: int = None):
    os.symlink(in_colmap_dir, os.path.join(out_colmap_dir, 'mvs_result'))


def mvs_mvsnet2mvsnet(in_mvsnet_dir, out_mvsnet_dir, build_id: int = None):
    os.symlink(in_mvsnet_dir, os.path.join(out_mvsnet_dir, 'mvs_result'))


def mvs_pointmvsnet2pointmvsnet(in_mvsnet_dir, out_mvsnet_dir, build_id: int = None):
    os.symlink(os.path.join(in_mvsnet_dir, 'Eval'), os.path.join(out_mvsnet_dir, 'mvs_result'))


def mvs_convert_helper(src_alg, target_alg, in_alg_dir, out_alg_dir, build_id: int = None):
    this_module = sys.modules[__name__]
    convert_fun = getattr(this_module, 'mvs_' + src_alg + '2' + target_alg)
    convert_fun(in_alg_dir, out_alg_dir, build_id)


if __name__ == '__main__':
    InitLogging()
    FLAGS.add_argument('input_dir', type=str, help='input directory')
    FLAGS.add_argument('output_dir', type=str, help='output directory')
    mvs_algorithm_list = ['colmap', 'openmvs', 'pmvs', 'cmvs', 'mve',
                          'mvsnet', 'rmvsnet', 'pointmvsnet']
    FLAGS.add_argument('--mvs', type=mvs_network_check, default='colmap', choices=mvs_algorithm_list,
                       help='mvs algorithm')
    FLAGS.add_argument('--fuse', type=mvs_network_check, default='colmap', choices=mvs_algorithm_list,
                       help='fuse algorithm')
    FLAGS.parse_args()

    mvs_convert_helper(FLAGS.sfm, FLAGS.mvs, FLAGS.input_dir, FLAGS.images_dir, FLAGS.output_dir)