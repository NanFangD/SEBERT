"""
 -*- coding: utf-8 -*-
@Time    : 2022/8/28 15:00
@Author  : nanfang
@File    : SelfAttention.py
"""
import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    input : batch_size * seq_len * input_dim
        q : batch_size * input_dim * dim_k
        k : batch_size * input_dim * dim_k
        v : batch_size * input_dim * dim_v
    """

    def __init__(self, input_dim, dim_k, dim_v, num_heads):
        super().__init__()
        self.dim_k = dim_k
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self.mulTiHeadAttention = nn.MultiheadAttention(input_dim, num_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.q(x)  # Q: batch_size * seq_len * dim_k
        K = self.k(x)  # K: batch_size * seq_len * dim_k
        V = self.v(x)  # V: batch_size * seq_len * dim_v
        attn_output, attn_output_weights = self.mulTiHeadAttention(Q, K, V)

        # attention = torch.bmm(self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(self.dim_k)), V)

        return attn_output
