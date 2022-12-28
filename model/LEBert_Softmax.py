"""
 -*- coding: utf-8 -*-
@Time    : 2022/9/28 15:09
@Author  : nanfang
@File    : LEBERT_Softmax.py
"""
import torch
import torch.nn as nn
from config import configs
from torch.nn.utils.rnn import pad_sequence
from transformers import BertPreTrainedModel
from model.LEBERT import LEBertModel
from transformers import logging

logging.set_verbosity_warning()


class LEBertSoftmax(BertPreTrainedModel):
    def __init__(self, config):
        super(LEBertSoftmax, self).__init__(config)
        self.word_embeddings = nn.Embedding(config.word_vocab_size, config.word_embed_dim)
        self.num_labels = config.num_labels
        self.bert = LEBertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True  # 使参数可更新
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.w1 = nn.Parameter(torch.ones(configs.batch_size, 1, 1))
        self.w2 = nn.Parameter(torch.ones(configs.batch_size, 1, 1))
        self.init_weights()

    def forward(self, input_ids, attention_mask, word_ids, word_mask, ignore_index=None, labels=None, seq_lengths=None,
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

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        outputs = logits
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (outputs, loss)
        return outputs
