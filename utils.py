# -*- coding: UTF-8 -*-

import subprocess
from csv import DictReader
import os
import logging


def SetupOneGpu():
    gpu_status_str = subprocess.run(('nvidia-smi', '--query-gpu=index,memory.free', '--format=csv'),
                                    stdout=subprocess.PIPE, check=True)

    gpu_status_dict = DictReader(gpu_status_str.stdout.decode().split('\n'))
    index_field_name = gpu_status_dict.fieldnames[0]
    memory_field_name = gpu_status_dict.fieldnames[1]
    gpu_status = list(gpu_status_dict)
    for gpu_info in gpu_status:
        gpu_info[index_field_name] = int(gpu_info[index_field_name].strip())
        gpu_info[memory_field_name] = int(gpu_info[memory_field_name].strip().split(' ')[0])

    gpu_status.sort(key=lambda x: x[memory_field_name], reverse=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_status[0][index_field_name])
    return gpu_status[0][index_field_name], gpu_status[0][memory_field_name]

def InitLogging(debug=False):
    # create logger
    logger = logging.getLogger('module_logger')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    if debug:
        ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
