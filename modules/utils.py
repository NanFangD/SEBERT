"""
 -*- coding: utf-8 -*-
@Time    : 2022/9/26 21:59
@Author  : nanfang
@File    : utils.py
"""
import sys
import time

import os
import torch
import pickle

from loguru import logger

from config import configs


def narrow_setup(interval=2):
    """
    抢占单GPU程序
    :param interval:间隔时间
    :return:
    """
    while True:
        unused_gpus = []
        datas = gpu_info()
        for data in datas:
            gpu_id = data['gpu_id']
            gpu_memory = data['gpu_memory']
            gpu_power = data['gpu_power']
            if gpu_memory > 1000:  # set waiting condition
                gpu_id_str = 'gpu id:%d W |' % gpu_id
                gpu_power_str = 'gpu power:%d W |' % gpu_power
                gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
                # sys.stdout.write('\t' + gpu_id_str + '\t' + gpu_memory_str + '\t' + gpu_power_str + '\n')
                # sys.stdout.flush()
            else:
                unused_gpus.append(gpu_id)
        if not unused_gpus:
            sys.stdout.write('正在抢占GPU，请等待....\r')
            sys.stdout.flush()
            time.sleep(interval)
        else:
            print('\n--------抢占成功，开始执行程序--------')
            return unused_gpus


def gpu_info():
    """
    得到当前所有GPU的信息
    :return:[{'gpu_id': i, 'gpu_memory': memory, 'gpu_power': power}]
    """
    datas = []
    memorys = []
    powers = []
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    for line in gpu_status:
        if 'MiB' in line:
            memorys.append(int(line.split('MiB')[0].strip()))
        if 'Default' in line:
            powers.append(int(line.split('%')[0].strip()))
    for i, (memory, power) in enumerate(zip(memorys, powers)):
        datas.append({'gpu_id': i, 'gpu_memory': memory, 'gpu_power': power})
    return datas


def clone_dic(dic):
    """
    复制一个新的dict()
    :param dic:
    :return:
    """
    _dic = dict()
    for k, v in dic.items():
        _dic[k] = v
    return _dic


def to_GPU(tensor):
    """
    将数据放到单GPU上
    :param tensor:
    :return:
    """
    if torch.cuda.is_available():
        if configs.device == '-1':
            device = torch.device(f"cuda:{configs.device[0]}")
        else:
            device = torch.device(f"cuda")
        tensor = tensor.to(device)
    return tensor


def to_GPUS(model):
    """
    将模型加载到多GPU上
    :param model:
    :return:
    """
    if torch.cuda.is_available():
        return torch.nn.DataParallel(model).cuda()
    else:
        return model


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_pickle(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
