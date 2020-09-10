# -*- coding: UTF-8 -*-

import argparse
import os

class GlobalOptionsStorage:
    def __init__(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--sfm_global', type=bool, default=False, help='use global method rather than incremental')
        parser.add_argument('--num_gpu', type=int, default=1, help='how many gpu to use')
        parser.add_argument('--num_cpu', type=int, default=os.cpu_count(), help='how many gpu to use')
        parser.add_argument('--mvs_max_w', type=int, default=None, help='max width to mvs input')
        parser.add_argument('--mvs_max_h', type=int, default=None, help='max height to mvs input')
        parser.add_argument('--mvs_max_d', type=int, default=None, help='max depth to mvs input')
        self.parser = parser
        self.options = None

    def __getattr__(self, item):
        return getattr(self.options, item)

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self):
        self.options = self.parser.parse_args()

GLOBAL_OPTIONS = GlobalOptionsStorage()