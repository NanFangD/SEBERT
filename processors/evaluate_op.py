"""
 -*- coding: utf-8 -*-
@Time    : 2022/7/22 11:01
@Author  : nanfang
@File    : evaluate_op.py
"""
import logging
import torch
from config import configs


def flatten_lists(lists, seq_lengths):
    """
    将二维列表的列表拼成一纬列表，根据seq_lengths清除扩充的0
    :param lists: 二维列表
    :param seq_lengths: labels 对应长度
    :return:
    """
    if type(lists) == torch.Tensor:
        lists = lists.tolist()
    flatten_list = []
    for length, line in zip(seq_lengths, lists):
        flatten_list += line[:length]
    return flatten_list


class MetricsWord(object):
    """
    单字级别的评价
    用于评价模型，计算每个标签的精确率，召回率，F1分数
    """

    def __init__(self, tag_to_ix, _type='word'):
        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.dic_pred_labels = dict()  # 预测标签各个标签的个数   ，实体级 统计预测各实体数
        self.dic_labels = dict()  # 源文件标签各个标签的个数      ，实体级 统计源文件中各实体数
        self.dic_pred_correct = dict()  # 预测正确的标签个数     ，实体级 统计预测正确的的实体数
        self.ix_to_tag = {v: k for k, v in tag_to_ix.items()}
        self.tag_to_ix = {k: v for k, v in tag_to_ix.items()}
        self._type = _type
        self.labels_sum = 0

        # 删除多余的填充符
        # if self.tag_to_ix.get("[PAD]", -100) != -100:
        #     del self.ix_to_tag[self.tag_to_ix['[PAD]']]
        #     del self.tag_to_ix['[PAD]']
        if _type == 'word':
            self.__init__word()
        else:
            self.__init__entity()

    def __init__word(self):
        self.ix_to_label = {v: k for k, v in self.tag_to_ix.items()}
        self.label_to_ix = {k: v for k, v in self.tag_to_ix.items()}

    def __init__entity(self):
        self.label_to_ix = dict()
        for label, ix in self.tag_to_ix.items():
            if label == "[PAD]":
                continue
            if label != "O" and label[2:] not in self.label_to_ix:
                self.label_to_ix[label[2:]] = ix
            elif label == "O" and label not in self.label_to_ix:
                self.label_to_ix[label] = ix
        self.ix_to_label = {v: k for k, v in self.label_to_ix.items()}

    def evaluate_res(self, pred, labels, seq_lengths):
        """
        根据预测值pred，和准确标签labels预测结果
        :param pred: 未被压缩的tensor
        :param labels: 未被压缩的tensor
        :param seq_lengths: tensor
        :return:
        """
        if type(pred) == torch.Tensor and len(pred.shape) > 2:
            pred = pred.max(dim=2)[1]
        pred_list = flatten_lists(pred, seq_lengths)
        labels_list = flatten_lists(labels, seq_lengths)
        if self._type == 'word':
            # 统计预测标签各个标签的个数
            self.list_to_dict(pred_list, self.dic_pred_labels)
            # 统计源文件标签各个标签的个数
            self.list_to_dict(labels_list, self.dic_labels)
            # 统计预测正确的标签个数
            self.count_correct_tags(pred_list, labels_list)
        else:
            # 统计预测标签各个实体的个数
            self.get_entity_nums(pred_list, self.dic_pred_labels)
            # 统计源文件标签各个实体的个数
            self.get_entity_nums(labels_list, self.dic_labels)
            # 统计预测正确的实体个数
            self.get_correct_entity_nums(pred_list, labels_list, self.dic_pred_correct)

        self.labels_sum = sum(self.dic_labels.values())
        # 计算精确率
        precision_scores = self.cal_precision()
        # 计算召回率
        recall_scores = self.cal_recall()
        # 计算F1分数
        f1_scores = self.cal_f1(precision_scores, recall_scores)
        avg_metrics = self.cal_weighted_average(precision_scores, recall_scores, f1_scores)
        return avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1_score']

    def count_correct_tags(self, pred_list, labels_list):
        """计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        for pred, label in zip(pred_list, labels_list):  # zip()用于将数据打包成元组
            if pred == label:  # dic_pred_correct[pred]加一
                pred = self.ix_to_tag[pred]
                if self.dic_pred_correct.get(pred, 0) == 0:  # 若该标签不在已知标签字典中，将其加入
                    self.dic_pred_correct[pred] = 1
                else:
                    self.dic_pred_correct[pred] += 1

    def get_entity_nums(self, labels_list, dic_labels):
        """
        计算每个实体对应的个数，注意BME、BE、S结构才为实体，其余均不计入 B-company E-company算实体
        :param tag_list:
        :return:
        """
        if configs.markup == 'bios':
            for ix in labels_list:
                tag = self.ix_to_tag[ix]
                if 'B-' in tag or 'S-' in tag:  # 当B-出现时，说明出现一个实体
                    tag = tag[2:]
                    dic_labels[tag] = dic_labels.get(tag, 0) + 1
                elif tag == 'O':
                    dic_labels[tag] = dic_labels.get(tag, 0) + 1
        else:  # 默认为bio标注
            for ix in labels_list:
                tag = self.ix_to_tag[ix]
                if 'B-' in tag:  # 当B-出现时，说明出现一个实体
                    tag = tag[2:]
                    dic_labels[tag] = dic_labels.get(tag, 0) + 1
                elif tag == 'O':
                    dic_labels[tag] = dic_labels.get(tag, 0) + 1

    def get_correct_entity_nums(self, pred, labels, dic_pred_correct):
        """
        计算每种实体被正确预测的个数
        address、book、company、game、government、movie、name、organization、position、scene
        :return:
        """
        length = len(labels)
        if configs.markup == 'bios':
            flag = 0  # 初始状态，表示等待下一个开始状态
            for i, (label, pred_label) in enumerate(zip(labels, pred)):  # zip()用于将数据打包成元组
                label = self.ix_to_tag[label]
                tag = label[2:]
                pred_label = self.ix_to_tag[pred_label]
                if 'B-' in label and label == pred_label and flag == 0:  # 当以B-开头且标签相等时
                    if i + 1 == length or (f"I-{tag}" != self.ix_to_tag[labels[i + 1]] and f"I-{tag}" != self.ix_to_tag[pred[i + 1]]):
                        flag = 0
                        dic_pred_correct[label[2:]] = dic_pred_correct.get(label[2:], 0) + 1
                    elif f"I-{tag}" == self.ix_to_tag[labels[i + 1]] and f"I-{tag}" == self.ix_to_tag[pred[i + 1]]:
                        flag = 1
                    else:
                        flag = 0
                elif 'I-' in label and label == pred_label and flag == 1:  # I前已经有过B
                    if i + 1 == length or (f"I-{tag}" != self.ix_to_tag[labels[i + 1]] and f"I-{tag}" != self.ix_to_tag[pred[i + 1]]):
                        flag = 0
                        dic_pred_correct[label[2:]] = dic_pred_correct.get(label[2:], 0) + 1
                    elif f"I-{tag}" == self.ix_to_tag[labels[i + 1]] and f"I-{tag}" == self.ix_to_tag[pred[i + 1]]:
                        flag = 1
                    else:
                        flag = 0
                elif 'S-' in label and label == pred_label:
                    flag = 0  # 状态重新调整回初始值
                    dic_pred_correct[label[2:]] = dic_pred_correct.get(label[2:], 0) + 1
                elif 'O' == label == pred_label:
                    flag = 0  # 状态重新调整回初始值
                    dic_pred_correct['O'] = dic_pred_correct.get('O', 0) + 1
                else:
                    flag = 0
        else:
            flag = 0  # 初始状态，表示等待下一个开始状态
            for i, (label, pred_label) in enumerate(zip(labels, pred)):  # zip()用于将数据打包成元组
                label = self.ix_to_tag[label]
                tag = label[2:]
                pred_label = self.ix_to_tag[pred_label]
                if 'B-' in label and label == pred_label and flag == 0:  # 当以B-开头且标签相等时
                    if i + 1 == length or (f"I-{tag}" != self.ix_to_tag[labels[i + 1]] and f"I-{tag}" != self.ix_to_tag[pred[i + 1]]):
                        flag = 0
                        dic_pred_correct[label[2:]] = dic_pred_correct.get(label[2:], 0) + 1
                    elif f"I-{tag}" == self.ix_to_tag[labels[i + 1]] and f"I-{tag}" == self.ix_to_tag[pred[i + 1]]:
                        flag = 1
                    else:
                        flag = 0
                elif 'I-' in label and label == pred_label and flag == 1:  # I前已经有过B
                    if i + 1 == length or (f"I-{tag}" != self.ix_to_tag[labels[i + 1]] and f"I-{tag}" != self.ix_to_tag[pred[i + 1]]):
                        flag = 0
                        dic_pred_correct[label[2:]] = dic_pred_correct.get(label[2:], 0) + 1
                    elif f"I-{tag}" == self.ix_to_tag[labels[i + 1]] and f"I-{tag}" == self.ix_to_tag[pred[i + 1]]:
                        flag = 1
                    else:
                        flag = 0
                elif 'O' == label == pred_label:
                    flag = 0  # 状态重新调整回初始值
                    dic_pred_correct['O'] = dic_pred_correct.get('O', 0) + 1
                else:
                    flag = 0

    def cal_precision(self):
        # 计算每个tag的准确率吧
        precision_scores = {}
        for ix, tag in self.ix_to_label.items():
            # pred_correct为tag计算正确的次数     pred_labels_num所有预测标签的个数
            precision_scores[tag] = self.dic_pred_correct.get(tag, 0) / max(1e-10, self.dic_pred_labels.get(tag, 0))
        return precision_scores

    def cal_recall(self):
        # 计算每个tag的召回率
        recall_scores = {}
        for ix, tag in self.ix_to_label.items():
            # correct_tags_number为tag计算正确的次数     golden_tags_counter为tag出现的总次数
            recall_scores[tag] = self.dic_pred_correct.get(tag, 0) / max(1e-10, self.dic_labels.get(tag, 0))
        return recall_scores

    def cal_f1(self, precision_scores, recall_scores):
        f1_scores = {}
        for ix, tag in self.ix_to_label.items():
            p, r = precision_scores.get(tag, 0), recall_scores.get(tag, 0)
            f1_scores[tag] = 2 * p * r / (p + r + 1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def show_evaluate(self):

        self.labels_sum = sum(self.dic_labels.values())
        # 计算精确率
        precision_scores = self.cal_precision()
        # 计算召回率
        recall_scores = self.cal_recall()
        # 计算F1分数
        f1_scores = self.cal_f1(precision_scores, recall_scores)
        # 输出结果
        self.report_scores(precision_scores, recall_scores, f1_scores)
        # 初始化参数
        self.__init__(self.tag_to_ix, _type=self._type)

    def report_scores(self, precision_scores, recall_scores, f1_scores):
        """将结果用表格的形式打印出来，像这个样子：
                      precision    recall  f1-score   pred_nums  pred_correct  labels_nums
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634
          avg/total      0.779     0.764     0.770      6178
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'pred_nums', 'pred_correct', 'labels_nums']
        print(header_format.format('', *header))
        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9} {:>9} {:>9}'
        # 打印每个标签的 精确率、召回率、f1分数
        for ix, tag in self.ix_to_label.items():
            print(row_format.format(tag, precision_scores.get(tag, 0), recall_scores.get(tag, 0), f1_scores.get(tag, 0),
                                    self.dic_pred_labels.get(tag, 0),
                                    self.dic_pred_correct.get(tag, 0),
                                    self.dic_labels.get(tag, 0)
                                    ))

        # 计算并打印平均值
        avg_metrics = self.cal_weighted_average(precision_scores, recall_scores, f1_scores)
        logging.info(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            sum(self.dic_pred_labels.values()),
            sum(self.dic_pred_correct.values()),
            self.labels_sum,
        ))

    def cal_weighted_average(self, precision_scores, recall_scores, f1_scores):

        weighted_average = {'precision': 0., 'recall': 0., 'f1_score': 0.}
        # 计算weighted precisions:
        for ix, tag in self.ix_to_label.items():
            if tag == "O":  # 平均值计算不包括O
                continue
            size = self.dic_labels.get(tag, 0)  # 标准文件各个标签的个数
            weighted_average['precision'] += precision_scores.get(tag, 0) * size
            weighted_average['recall'] += recall_scores.get(tag, 0) * size
            weighted_average['f1_score'] += f1_scores.get(tag, 0) * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= max(1e-10, self.labels_sum - self.dic_labels.get('O', 0))
        return weighted_average

    def list_to_dict(self, _list, dic):
        """
        将list添加到对应的dict
        :param _list: list[]
        :param dic: dict()
        :return:
        """
        for v in _list:
            v = self.ix_to_tag[v]
            if dic.get(v, 0) == 0:
                dic[v] = 1
            else:
                dic[v] += 1


if __name__ == '__main__':
    pass
    # # 需要更改的参数
    # data_path = '../../dataset/NER/suzhou'
    # labels_types = ['B', 'I']
    # metrics_type = "entity"  # 或者word
    # BIDIRETIONAL = True
    # SHUFFLE_DATA = True
    # use_bert = False
    # tokenizer_model = configs.chinese_model
    # processData, [trainloader, devloader, testloader] = get_dataloaders(batch_size=configs.batch_size,
    #                                                                     shuffle_data=SHUFFLE_DATA,
    #                                                                     file_path=data_path,
    #                                                                     labels_types=labels_types,
    #                                                                     bert_path='../pretrained_bert_models/bert-base-chinese/',
    #                                                                     use_bert=use_bert)
    # if devloader is None:
    #     devloader = testloader
    # # 评价指标类
    # Metrics = MetricsWord(processData.ix_to_tag, _type='entity')
