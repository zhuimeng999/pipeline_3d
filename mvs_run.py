# -*- coding: UTF-8 -*-

import os, sys
import argparse, logging
from utils import SetupOneGpu, InitLogging

module_logger = logging.getLogger('module_logger')


def maybe_convert_sfm_result():
    pass


def mvs_colmap(options):



if __name__ == '__main__':
    InitLogging()
    module_logger.info('select gpu %d, has free memory %d MiB', *SetupOneGpu())

    parser = argparse.ArgumentParser()
    parser.add_argument('sfm_path', type=str, help='sfm result directory')
    parser.add_argument('sfm_algorithm', default=None, choices=['colmap', 'openmvg', 'theiasfm', 'mve'], help='sfm algorithm type')
    parser.add_argument('output_path', type=str, help='output directory')
    subparser = parser.add_subparsers(dest='alg_type', metavar='algorithm', help='algorithm to use, support are {colmap|openmvs|cmvs-pmvs|mve}')
    parser_colmap = subparser.add_parser('colmap')
    parser_openmvs = subparser.add_parser('openmvs')
    parser_cpmvs = subparser.add_parser('cmvs-pmvs')
    parser_mve = subparser.add_parser('mve')

    options = parser.parse_args()
    maybe_convert_sfm_result()

    if options.alg_type == 'openmvs':
        mvs_openmvs(options)
    elif options.alg_type == 'cmvs-pmvs':
        mvs_cpmvs(options)
    elif options.alg_type == 'mve':
        mvs_mve(options)
    else:
        mvs_colmap(options)
