"""
 -*- coding: utf-8 -*-
@Time    : 2022/10/8 20:13
@Author  : nanfang
@File    : BILSTM_CRF.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from model.SelfAttention import SelfAttention
from model.crf import CRF
from config import configs
from model.CLN import LayerNorm


class LSTMCRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_labels, bidirectional=True):
        super(LSTMCRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = 2 if bidirectional else 1
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.5)
        self.num_labels = num_labels
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=4, batch_first=True, bidirectional=bidirectional)
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        self.layerNorm = nn.LayerNorm(hidden_dim * self.bidirectional)
        self.linear = nn.Linear(hidden_dim * self.bidirectional, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None, seq_lengths=None, loss_function=None):
        embeds = self.word_embeddings(input_ids)
        # max_length = embeds.shape[1]
        embeds = pack_padded_sequence(embeds, seq_lengths, batch_first=True)
        embeds, _ = self.lstm(embeds)
        embeds = pad_packed_sequence(embeds, batch_first=True)[0]

        # 加入语义信息
        if configs.add_weights:  # 是否加入语义信息
            weight = torch.cat(_, dim=2)[-1]
            embeds = embeds.reshape(embeds.shape[1], embeds.shape[0], -1)
            embeds += weight * 1  # 目前0.5效果是最好的，达到0.9784
            embeds = embeds.reshape(embeds.shape[1], embeds.shape[0], -1)
        embeds = self.dropout(embeds)
        embeds = self.layerNorm(embeds)

        # 得到判别值
        logits = self.linear(embeds)
        # 为了防止多卡运行，所以labels和 attention_mask都要压缩一下
        if labels is not None:
            loss2 = - self.crf(emissions=logits, tags=labels, mask=attention_mask)
            if configs.add_mix_loss and loss_function is not None and attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss1 = loss_function(active_logits, active_labels)
                loss = loss1 / loss1.detach() + loss2 / loss2.detach()
                return logits, loss
            else:
                return logits, loss2
        else:
            return logits  # (loss), scores
