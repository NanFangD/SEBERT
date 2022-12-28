"""
 -*- coding: utf-8 -*-
@Time    : 2022/8/13 10:49
@Author  : nanfang
@File    : txt_op.py
"""


def read_txt_data(file):
    """
    读取打上【】标签的数据
    :param file:str
    :return:list[list[]]
    """
    lines = []
    with open(file, 'r', encoding='utf-8')as f:
        data = f.readlines()
    line = []
    for text in data:
        if text == '\n' and line != []:
            lines.append(line)
            line = []
        else:
            line.append(text.strip())
    if line:
        lines.append(line)
    return lines


def save_txt_data(file, contents, _type='contents'):
    """
    保存需要标签的数据为txt格式
    :param _type:
    :param file:str    _type=contents  则 contents=list[list[str]]    _type=content 则 contents=list[str]
    :param contents:
    :return:
    """
    with open(file, 'w', encoding='utf-8')as f:
        if _type == "contents":  # list[list[str]]
            for lines in contents:
                for line in lines:
                    f.write(line + '\n')
                f.write('\n')
        elif _type == 'content':  # list[str]
            for line in contents:
                f.write(line + '\n')


if __name__ == '__main__':
    train_data = read_txt_data('../../dataset/CoNLL-2003/train.txt')
    dev_data = read_txt_data('../../dataset/CoNLL-2003/dev.txt')
    test_data = read_txt_data('../../dataset/CoNLL-2003/test.txt')
