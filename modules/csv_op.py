"""
-*- coding: utf-8 -*-
@Time :  2022-03-25 17:02
@Author : nanfang
"""
import csv
import pandas as pd

def write_csv(res, write_path, _type='w'):
    """
    处理数据，并写入新的文件
    :param res: [list[str]]
    :param write_path: str
    :param _type : str   (默认是覆盖写w，添加的话可以改成a）
    :return: bool
    """
    if not res:
        return False
    try:
        with open(write_path, _type, newline='', encoding='utf-8') as csvfile:  # 1. 创建文件对象
            writer = csv.writer(csvfile)  # 2. 基于文件对象构建 csv写入对象
            if type(res[0]) == list:
                for line in res:
                    writer.writerow(line)  # 3. 写入csv文件内容
            else:
                writer.writerow(res)  # 3. 写入csv文件内容
        return True
    except IOError:
        return False


def read_csv(file_path):
    """
    读取csv文件并以列表形式输出
    :param file_path: str
    :return: [list[str]]
    """
    res = []
    with open(file_path, 'r', encoding='utf-8')as f:
        read = csv.reader(f)
        for line in read:
            res.append(line)
    return res


def read_csv_complex(file_path, index=0):
    """
    读取复杂csv文件并以列表形式输出
    :param index: int    前几行是字符串
    :param file_path: str
    :return: [list[list..]]
    """
    person_msg = []
    with open(file_path, 'r', encoding='utf-8')as f:
        read = csv.reader(f)
        for lines in read:
            res = lines[:index]
            for line in lines[index:]:
                if line:
                    res.append(eval(line))
            person_msg.append(res)
    return person_msg


def del_col_row_csv(file_path, write_path=None, index=None, axis=0):
    """
    删除行或者列
    :param file_path:  str 读入文件地址
    :param write_path: str  输出文件地址
    :param index: str/int   删除行或者列的索引（删除列时可以是字符串或者序号，删除行时必须时序号）
    删除行时，不算title，序号从0开始
    删除列时，序号直接从0开始
    :param axis: int    axis=1 是列， axis=0 是行
    :return:DataFrame   删除后的结果
    """
    try:
        if index is None:
            return False
        data = pd.read_csv(file_path)
        if axis == 0:
            if type(index) != str:
                title = index
            else:
                title = data.columns[index]
        else:
            if type(index) == str:
                title = index
            else:
                title = data.columns[index]
        data_new = data.drop([title], axis=axis)  # 删除表头为Name的这列数据 ,axis=1表示列, axis=0表示删除行，默认值为0
        if write_path is not None:
            data_new.to_csv(write_path, index=0)
        return True
    except IOError:
        return False


def get_column_con(csv, column):
    """
    获取某一列的所有内容
    :param csv:  list[list[]]  csv文件内容
    :param column:  int   列号，从0开始
    """
    return [res[column] for res in csv]
