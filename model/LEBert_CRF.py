"""
 -*- coding: utf-8 -*-
@Time    : 2022/9/29 10:39
@Author  : nanfang
@File    : LEBert_CRF.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertPreTrainedModel
from transformers import logging

from config import configs
from model.CLN import LayerNorm
from model.LEBERT import LEBertModel
from model.crf import CRF

logging.set_verbosity_warning()


class LEBertCrf(BertPreTrainedModel):
    def __init__(self, config):
        super(LEBertCrf, self).__init__(config)
        self.num_labels = config.num_labels
        self.word_embeddings = nn.Embedding(config.word_vocab_size, config.word_embed_dim)
        self.bert = LEBertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True  # 使参数可更新
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.cln = LayerNorm(config.hidden_size, config.hidden_size, conditional=False)
        # self.w1 = nn.Parameter(torch.ones(configs.batch_size, 1, 1))
        # self.w2 = nn.Parameter(torch.ones(configs.batch_size, 1, 1))
        # self.w_loss = nn.Parameter(torch.ones(2))
        self.init_weights()

    def forward(self, input_ids, attention_mask, word_ids, word_mask, ignore_index=None, seq_lengths=None, labels=None,
                loss_function=None):
        word_embeddings = self.word_embeddings(word_ids)
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None,
            word_embeddings=word_embeddings, word_mask=word_mask
        )
        sequence_output = outputs[0]

        if configs.add_weights:  # 是否加入全局信息
            sequence_output = [v[:] + v[0] * 0.5 for v in sequence_output]
            sequence_output = pad_sequence(sequence_output, batch_first=True)
        sequence_output = self.cln(sequence_output)
        # if configs.add_weights:  # 是否加入全局信息
        #     weight = sequence_output[:, 0]
        #     weight = weight.reshape(weight.shape[0], 1, -1)
        #     sequence_output = sequence_output * self.w1
        #     weight = weight * self.w2
        #     sequence_output += weight

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss2 = - self.crf(emissions=logits, tags=labels, mask=attention_mask)
            if configs.add_mix_loss and loss_function is not None and attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss1 = loss_function(active_logits, active_labels)
                # w1 = torch.exp(self.w_loss[0]) / torch.sum(torch.exp(self.w_loss))
                # w2 = torch.exp(self.w_loss[1]) / torch.sum(torch.exp(self.w_loss))
                # loss = w1 * loss1 + w2 * loss2
                # loss = loss1 + loss2  # 1、
                loss = loss1 / loss1.detach() + loss2 / loss2.detach()  # 2、
                return logits, loss
            else:
                return logits, loss2
        else:
            return logits  # (loss), scores
