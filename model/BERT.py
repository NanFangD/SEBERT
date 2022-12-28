"""
 -*- coding: utf-8 -*-
@Time    : 2022/10/30 19:21
@Author  : nanfang
@File    : BERT.py
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from config import configs
from model.CLN import LayerNorm
import torch.nn as nn


class BertSoftmax(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmax, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True  # 使参数可更新
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.layerNorm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, head_mask=None, seq_lengths=None,
                loss_function=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=None,
                            position_ids=None,
                            inputs_embeds=None)
        sequence_output = outputs[0]

        if configs.add_weights:  # 是否加入全局信息
            sequence_output = [v[:] + v[0] * 0.5 for v in sequence_output]
            sequence_output = pad_sequence(sequence_output, batch_first=True)

        sequence_output = self.dropout(sequence_output)
        # 得到判别值
        logits = self.classifier(sequence_output)
        if labels is not None:
            active_loss = attention_mask.contiguous().view(-1) == 1
            active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
            active_labels = labels.contiguous().view(-1)[active_loss]
            loss = loss_function(active_logits, active_labels)
            return logits, loss
        else:
            return logits  # (loss), scores
