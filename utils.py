# -*- coding: UTF-8 -*-

import subprocess
from csv import DictReader
import os
import logging
import pathlib


def SetupFreeGpu(n = 1):
    LogThanExitIfFailed(n > 0, 'gpu number must greater than 0, got %d', n)
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

    gpus = []
    for i in range(n):
        gpus.append(str(gpu_status[i][index_field_name]))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
    return gpus

def InitLogging(debug=False):
    # create logger
    if debug:
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def LogThanExitIfFailed(condition:bool, msg:str, *args):
    if not condition:
        logging.critical(msg, *args, stack_info=True)
        exit(1)

def GetFileFromBuildId(file_dir:str, pattern:str, build_id:int = None):
    file_candiates = list(pathlib.Path(file_dir).glob(pattern))
    if build_id is None:
        LogThanExitIfFailed(len(file_candiates) == 1,
                            'there are many matches in %s with %s, you must special a build_id: %s',
                            file_dir, pattern, file_candiates)
        return str(file_candiates[0].absolute().as_posix())
    else:
        file_selected = list(filter(lambda x: str(x.as_posix())[-1] == str(build_id), file_candiates))
        LogThanExitIfFailed(len(file_selected) == 1,
                            'there are many matches in %s with %s, build_id %d: %s',
                            file_dir, pattern, build_id, file_selected)
        return str(file_selected[0].absolute().as_posix())