# -*- coding: UTF-8 -*-

import argparse
import os


def get_common_options_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sfm_global', type=bool, default=False, help='use global method rather than incremental')
    parser.add_argument('--num_gpu', type=int, default=1, help='how many gpu to use')
    parser.add_argument('--num_cpu', type=int, default=os.cpu_count(), help='how many gpu to use')
    return parser
