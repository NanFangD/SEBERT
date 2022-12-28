import os
import json


class Vocabulary(object):
    """
    构建词表
    """

    def __init__(self, tokens, vocab_type='label'):
        """

        :param tokens:
        :param vocab_type:
        """
        assert vocab_type in ['label', 'word', '']
        self.token2idx = {}
        self.idx2token = []
        self.size = 0
        if vocab_type == 'word':
            tokens = ['[PAD]', '[UNK]'] + tokens
        elif vocab_type == 'label':
            tokens = ['[PAD]'] + tokens

        for token in tokens:
            self.token2idx[token] = self.size
            self.idx2token.append(token)
            self.size += 1

    def get_size(self):
        return self.size

    def convert_token_to_id(self, token):
        if token in self.token2idx:
            return self.token2idx[token]
        else:
            return self.token2idx['[UNK]']

    def convert_tokens_to_ids(self, tokens):
        if type(tokens) is list:
            return [self.convert_token_to_id(token) for token in tokens]
        else:
            return self.convert_token_to_id(tokens)

    def convert_id_to_token(self, idx):
        return self.idx2token[idx]

    def convert_ids_to_tokens(self, ids):
        if type(ids) is list:
            return [self.convert_id_to_token(id) for id in ids]
        else:
            return self.convert_id_to_token(ids)
