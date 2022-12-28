"""
 -*- coding: utf-8 -*-
@Time    : 2022/8/1 15:24
@Author  : nanfang
@File    : json_op.py
"""
import json


def read_json(file):
    """
    加载列表类型存储的json数据
    :param file: str
    :return: list[json]
    """
    with open(file, encoding='utf-8')as f:
        res = []
        for line in f.readlines():
            if line == '\n':
                continue
            res.append(json.loads(line))
    return res


def write_json(res, write_path, _type='w'):
    """
    处理数据，并写入新的文件
    :param res: list[json]
    :param write_path: str
    :param _type : str   (默认是覆盖写w，添加的话可以改成a）
    :return: bool
    """
    if not res:
        return False
    try:
        with open(write_path, _type, newline='', encoding='utf-8') as f:  # 1. 创建文件对象
            for json_v in res:
                f.write(json.dumps(json_v, ensure_ascii=False) + '\n')  # 2. 基于文件对象构建 csv写入对象
        return True
    except IOError:
        return False
