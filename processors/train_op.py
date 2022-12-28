"""
 -*- coding: utf-8 -*-
@Time    : 2022/7/27 11:10
@Author  : nanfang
@File    : train_op.py
"""
import torch
from loguru import logger
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from modules.print_op import *
# 模型训练模板
from modules.utils import to_GPU
from config import configs


def train_epoch(epoch, dataloader, model, metric, optimizer, loss_function=None):
    model.train()
    return training(epoch, dataloader, model, metric, optimizer, loss_function=loss_function, _type='Train')


def dev_epoch(epoch, dataloader, model, metric, loss_function=None):
    model.eval()
    with torch.no_grad():
        return training(epoch, dataloader, model, metric, loss_function=loss_function, _type='Dev')


def test_epoch(epoch, dataloader, model, metric, loss_function=None):
    model.eval()
    with torch.no_grad():
        return training(epoch, dataloader, model, metric, loss_function=loss_function, _type='Test')


def training(epoch, dataloader, model, metric, optimizer=None, loss_function=None, _type='Train'):
    # if _type == 'Train':
    #     testloader = dataloader[1]
    #     dataloader = dataloader[0]
    # acc_steps = 4  # 每四次更新一次参数
    print_GREEN("\n")
    loss_sum = 0
    i = 0
    data_bar = tqdm(dataloader, mininterval=0.3)
    for idx, items in enumerate(data_bar):
        i += 1  # 用于统计循环次数
        # if i % 50 == 1 and _type == 'Train':
        #     metric.__init__(metric.id2label, markup=configs.markup)  # 初始化评价指标类
        #     with torch.no_grad():
        #         training(0, testloader, model, metric, optimizer=optimizer, loss_function=loss_function, _type='Test')
        model.zero_grad()  # 模型梯度清零
        if 'lebert' in configs.model_name:
            # 字符数据，标签，字符mask，字符对应单词，单词mask，句子长度   ，#toGPU 将数据放到GPU上
            datas, labels, mask, words, word_mask, seq_lengths = to_GPU(items[0]), to_GPU(items[1]), to_GPU(
                items[2]), to_GPU(items[3]), to_GPU(
                items[4]), items[5]
            pred = model(input_ids=datas, attention_mask=mask, word_ids=words, word_mask=word_mask,
                         ignore_index=configs.ignore_index,
                         labels=labels, seq_lengths=seq_lengths, loss_function=loss_function)
        elif 'bert' in configs.model_name or 'lstm' in configs.model_name:
            datas, labels, mask, seq_lengths = to_GPU(items[0]), to_GPU(items[1]), to_GPU(items[2]), items[-1]
            pred = model(input_ids=datas, attention_mask=mask, seq_lengths=seq_lengths, loss_function=loss_function,
                         labels=labels)
        else:
            print("模型选择错误！，返回！")
            return
        if len(pred) == 2:
            logits, loss = pred
            loss = loss.mean()  # 对多卡loss求平均
            loss_sum += loss.item()
        else:
            logits = pred
        if _type == 'Train':  # 如果是训练集则更新模型参数
            loss.backward()  # 更新loss参数
            # if (idx + 1) % acc_steps == 0:  # 使用梯度累加更新，为了使用更大的batch
            for opti in optimizer:
                opti.step()
            optimizer[0].zero_grad()
        # 优化损失求和
        # 首先解除 batch_loss 张量的 GPU 占用，将张量中的数据取出再进行累加。
        label_ids = labels[:, 1:-1].tolist()  # 减去padding的[CLS]与[SEP]
        input_lens = (torch.sum(datas != 0, dim=-1) - 2).tolist()  # 减去padding的[CLS]与[SEP]
        if len(configs.device) > 1:  # 判断是否使用多GPU
            module = model.module
        else:
            module = model
        if 'crf' in configs.model_name:  # 如果用到了CRF模型，则需要用CRF函数计算一下结果
            preds = module.crf.decode(logits, mask).squeeze(0)
            preds = preds[:, 1:-1].tolist()  # 减去padding的[CLS]与[SEP]
        else:
            preds = torch.argmax(logits, dim=2)[:, 1:-1].tolist()  # 减去padding的[CLS]与[SEP]
        # 显示在 tqdm上
        data_bar.desc = '{}-epoch {} ,loss: {:.4f}'.format(epoch, _type, loss_sum / i)
        for j in range(len(label_ids)):
            input_len = input_lens[j]
            pred = preds[j][:input_len]
            label = label_ids[j][:input_len]
            metric.update(pred_paths=[pred], label_paths=[label])
    eval_loss = loss_sum / i  # 平均loos
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    print_BLUE(f"***** {_type} results *****")
    info = "-".join([f' {key}: {value:.5f} ' for key, value in results.items()])
    print_BLUE(info)
    print_BLUE("***** Entity results *****")
    for key in sorted(entity_info.keys()):
        print_BLUE("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.5f} ' for key, value in entity_info[key].items()])
        print_BLUE(info)
    return results


def distillation(student_pred, labels, teacher_pred, seq_lengths, temp, alpha):
    """
    关键，定义kd的loss
    """
    student_pred = pack_padded_sequence(student_pred, seq_lengths, batch_first=True)[0]
    teacher_pred = pack_padded_sequence(teacher_pred, seq_lengths, batch_first=True)[0]
    pack_labels = pack_padded_sequence(labels, seq_lengths, batch_first=True)[0]
    return nn.KLDivLoss()(F.log_softmax(student_pred / temp, dim=1), F.softmax(teacher_pred / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(student_pred, pack_labels) * (1. - alpha)


# 模型训练模板
def KD_train_epoch(epoch, trainloader, teacher_model, student_model, optimizer, Metrics, loss_function=None):
    # set module to training mode
    student_model.train()
    teacher_model.eval()  # teacher_model 不需要更新
    precision_scores = 0
    recall_scores = 0
    f1_scores = 0
    loss_sum = 0
    i = 0
    # step number in one epoch: 336
    train_bar = tqdm(trainloader)

    for idx, batch_samples in enumerate(train_bar):
        datas, labels, seq_lengths, masks = to_GPU(batch_samples[0]), to_GPU(batch_samples[1]), batch_samples[
            2], to_GPU(
            batch_samples[3])
        # compute module pred and loss
        # 求loss
        # teacher_pred = teacher_model(input_data=datas, attention_mask=masks, seq_lengths=seq_lengths)
        teacher_pred, teacher_loss = teacher_model(input_data=datas, attention_mask=masks, seq_lengths=seq_lengths,
                                                   loss_function=loss_function,
                                                   labels=labels)
        teacher_pred = teacher_pred.detach()  # 切断老师网络的反向传播
        # student_pred = student_model(input_data=datas, attention_mask=masks, seq_lengths=seq_lengths)
        student_pred, student_loss = student_model(input_data=datas, attention_mask=masks, seq_lengths=seq_lengths,
                                                   loss_function=loss_function,
                                                   labels=labels)
        r = 0.2
        loss = r * teacher_loss + (1 - r) * student_loss
        # loss = distillation(student_pred, labels, teacher_pred, seq_lengths, temp=5.0, alpha=0.7)
        precision_scores, recall_scores, f1_scores = Metrics.evaluate_res(student_pred, labels, seq_lengths)
        optimizer[0].zero_grad()
        if type(loss.tolist()) == list:  # 判断是不是放在多个GPU上了
            loss = loss.sum()  # 放在多个GPU上会得到多个loss，loss相加即可
        loss.backward()  # 更新loss参数
        for opti in optimizer:
            opti.step()
        loss_sum += loss.item()
        i += 1
        # 显示在 tqdm上
        train_bar.desc = '{}-epoch, precision :{:.5f},recall :{:.5f},f1 :{:.5f}, loss: {:.5f}'.format(epoch,
                                                                                                      precision_scores,
                                                                                                      recall_scores,
                                                                                                      f1_scores,
                                                                                                      loss_sum / i)
    if i != 0:
        print('{}-epoch Train, precision :{:.5f}, recall :{:.5f}, f1 :{:.5f}, loss Mean: {:.5f}'.format(epoch,
                                                                                                        precision_scores,
                                                                                                        recall_scores,
                                                                                                        f1_scores,
                                                                                                        loss_sum / i))
    return f1_scores
