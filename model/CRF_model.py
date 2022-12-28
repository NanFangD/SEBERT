"""
 -*- coding: utf-8 -*-
@Time    : 2022/9/11 10:57
@Author  : nanfang
@File    : CRF_model.py
"""
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torchcrf import CRF
import torch.nn as nn

from modules.utils import to_GPU


class CRF_model(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.crf = CRF(num_labels, batch_first=True)  # CEF层

    def forward(self, pred, attention_mask, labels=None, seq_lengths=None, max_length=None):
        # 为了防止多卡运行，所以labels和 attention_mask都要压缩一下
        if seq_lengths is not None and max_length is not None:
            mask = pack_padded_sequence(attention_mask, seq_lengths, batch_first=True)
            mask = pad_packed_sequence(mask, batch_first=True)[0]
            outputs = self.crf.decode(pred, mask)
            outputs[0] = outputs[0] + [0] * (max_length - len(outputs[0]))
            outputs = to_GPU(pad_sequence([torch.tensor(v) for v in outputs], batch_first=True))
            if labels is not None:
                label = pack_padded_sequence(labels, seq_lengths, batch_first=True)
                label = pad_packed_sequence(label, batch_first=True)[0]
                loss = - self.crf(pred, label, mask, reduction='mean')
                outputs = (outputs, loss)
            return outputs
        # 使用单GPU时，可直接使用以下代码
        else:
            outputs = self.crf.decode(pred, attention_mask)
            if labels is not None:
                loss = -self.crf.forward(pred, labels, attention_mask, reduction='mean')
                outputs = (outputs, loss)
            return outputs
