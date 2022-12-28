"""
 -*- coding: utf-8 -*-
@Time    : 2022/9/26 21:58
@Author  : nanfang
@File    : dataProcessor.py
"""
import os

import torch
from loguru import logger

from modules.json_op import read_json
from modules.txt_op import read_txt_data
from transformers import AutoTokenizer
from config import configs
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from modules.utils import load_pickle, write_pickle
from processors.vocab import Vocabulary


def labelsdict(labels):
    # 将label转换成字典对应形式
    dic = {"PAD": 0}
    for i, v in enumerate(labels, 1):
        dic[v] = i
    return dic


class Processor(object):
    # 数据处理
    def __init__(self, model_word_embedding, word_vocab, embed_dim, trie_tree):
        self.labe_vocab = Vocabulary(read_txt_data(configs.label_file)[0], vocab_type='label')
        self.tokenizer = AutoTokenizer.from_pretrained(configs.pretrain_model_path)
        self.word_vocab_num = self.tokenizer.__len__()
        self.labelPAD = self.labe_vocab.convert_tokens_to_ids('[PAD]')
        self.wordPAD = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.model_word_embedding = model_word_embedding
        self.word_vocab = word_vocab
        self.embed_dim = embed_dim
        self.trie_tree = trie_tree
        self.max_word_num = configs.max_word_num
        self.dataset = {
            'train': self.data2ix(read_json(configs.train_file), 'train'),
            'dev': self.data2ix(read_json(configs.dev_file), 'dev'),
            'test': self.data2ix(read_json(configs.test_file), 'test')
        }

    def data2ix(self, data, _type):
        """
        将数据转换为ix形式,并加载对应的词语
        :param data: list[dict()]
        :param _type: 传入的数据类型，例如'train','dev','test'
        :return: list[dict()]
        """
        json_data = []
        for line in tqdm(data, desc=f"加载{_type}集数据"):
            text = line['text'][:configs.max_seq_len]
            label = line['label'][:configs.max_seq_len]
            dic = dict()
            # 加载字向量
            if 'bert' in configs.model_name:
                # 使用Bert模型，则在句子前后加上 CLS 和 SEP
                dic['text'] = [self.tokenizer.convert_tokens_to_ids('[CLS]')] + \
                              [self.tokenizer.convert_tokens_to_ids(word.lower()) for word in text] + \
                              [self.tokenizer.convert_tokens_to_ids('[SEP]')]
                dic['label'] = [self.labe_vocab.token2idx['O']] + \
                               [self.labe_vocab.token2idx[tag] for tag in label] + \
                               [self.labe_vocab.token2idx['O']]
            else:
                dic['text'] = [self.tokenizer.convert_tokens_to_ids(word.lower()) for word in text]
                dic['label'] = [self.labe_vocab.token2idx[tag] for tag in label]
            # 加载词向量
            if configs.load_word_embed:
                char_index2words = self.get_char2words(text)
                word_ids_list = []  # 单词对应位置
                word_mask_list = []  # 单词对应mark标志
                for words in char_index2words:
                    # 去掉长度为1的字
                    words = [word for word in words if len(word) > 1]
                    # 获取前max_word_num 个单词
                    words = words[:self.max_word_num]
                    word_ids = self.word_vocab.convert_tokens_to_ids(words)
                    word_pad_num = self.max_word_num - len(words)
                    word_mask_list.append(len(word_ids) * [1] + [self.labelPAD] * word_pad_num)
                    word_ids = word_ids + [self.wordPAD] * word_pad_num
                    word_ids_list.append(word_ids)
                # 开头和结尾进行padding
                word_ids_list = [[self.wordPAD] * self.max_word_num] + word_ids_list + [
                    [self.wordPAD] * self.max_word_num]
                word_mask_list = [[self.wordPAD] * self.max_word_num] + word_mask_list + [
                    [self.wordPAD] * self.max_word_num]
                dic['words'] = word_ids_list
                dic['words_mask'] = word_mask_list
            json_data.append(dic)
        return json_data

    def collate(self, batch):
        # 将一个batch中的句子处理成相同长度
        # dataLoader加载时 先进入这个函数处理
        WORD_PAD_ID = self.wordPAD  # 获取PAD的id
        LABEL_PAD_ID = self.labelPAD
        batch.sort(key=lambda x: len(x['text']), reverse=True)  # 按照长度进行排序
        max_len = len(batch[0]['text'])  # 最长的
        input = []  # 字符
        target = []  # 标签
        words = []  # 单词
        words_mask = []  # 单词对应掩码
        seq_lengths = []  # 每个句子原始长度
        mask = []  # 字符对应掩码
        if configs.load_word_embed:
            for item in batch:
                pad_len = max_len - len(item['text'])
                input.append(item['text'] + [WORD_PAD_ID] * pad_len)
                target.append(item['label'] + [LABEL_PAD_ID] * pad_len)
                seq_lengths.append(len(item['label']))
                mask.append([1] * len(item['text']) + [0] * pad_len)
                words.append(item['words'] + [[WORD_PAD_ID] * self.max_word_num] * pad_len)
                words_mask.append(item['words_mask'] + [[0] * self.max_word_num] * pad_len)
            return torch.tensor(input), torch.tensor(target), torch.tensor(mask).bool(), torch.tensor(
                words), torch.tensor(
                words_mask).bool(), seq_lengths
        else:
            for item in batch:
                pad_len = max_len - len(item['text'])
                input.append(item['text'] + [WORD_PAD_ID] * pad_len)
                target.append(item['label'] + [LABEL_PAD_ID] * pad_len)
                seq_lengths.append(len(item['label']))
                mask.append([1] * len(item['text']) + [0] * pad_len)
            return torch.tensor(input), torch.tensor(target), torch.tensor(mask).bool(), seq_lengths

    def get_char2words(self, text):
        """
        获取每个汉字，对应的单词列表
        :param text:
        :return:
        """
        text_len = len(text)
        char_index2words = [[] for _ in range(text_len)]

        for idx in range(text_len):
            sub_sent = text[idx:idx + self.trie_tree.max_depth]  # speed using max depth
            words = self.trie_tree.enumerateMatch(sub_sent)  # 找到以text[idx]开头的所有单词
            for word in words:
                start_pos = idx
                end_pos = idx + len(word)
                for i in range(start_pos, end_pos):
                    char_index2words[i].append(word)
        # todo 截断
        # for i, words in enumerate(char_index2words):
        #     char_index2words[i] = char_index2words[i][:self.max_word_num]
        return char_index2words


class DiabetesDataset(Dataset):
    def __init__(self, data):
        self.xy = data
        self.len = len(data)

    def __getitem__(self, index):
        return self.xy[index]

    def __len__(self):
        return self.len


def get_dataloaders(model_word_embedding=None, word_vocab=None, embed_dim=None, trie_tree=None):
    """
    根据processor数据初始化Dataloadel
    :return:
    """
    # train_loader_path = configs.dataset_path + 'train_loader.pkl'
    # dev_loader_path = configs.dataset_path + 'dev_loader.pkl'
    # test_loader_path = configs.dataset_path + 'test_loader.pkl'
    dataloaders = []
    # # 下面是保存数据的，这里注释掉了
    # if configs.overwrite or (not os.path.exists(train_loader_path) or
    #                          not os.path.exists(dev_loader_path) or
    #                          not os.path.exists(test_loader_path)):
    batch_size = configs.batch_size
    shuffle_data = configs.shuffle_data
    drop_last = configs.drop_last
    num_workers = configs.num_workers
    # 获取数据
    processor = Processor(model_word_embedding, word_vocab, embed_dim, trie_tree)

    for _type in ['train', 'dev', 'test']:
        dataloader = DataLoader(DiabetesDataset(processor.dataset[_type]), batch_size=batch_size, shuffle=shuffle_data,
                                drop_last=drop_last,
                                num_workers=num_workers, pin_memory=True, collate_fn=processor.collate)
        dataloaders.append(dataloader)
        # logger.info('write train_loader.pkl...')
        # write_pickle(dataloaders[0], train_loader_path)  # train_loader
        # logger.info('write dev_loader.pkl...')
        # write_pickle(dataloaders[1], dev_loader_path)  # dev_loader
        # logger.info('write test_loader.pkl...')
        # write_pickle(dataloaders[2], test_loader_path)  # test_loader
    # else:
    #     # 直接加载存好的pkl
    #     logger.info('read train_loader.pkl...')
    #     train_loader = load_pickle(train_loader_path)
    #     logger.info('read dev_loader.pkl...')
    #     dev_loader = load_pickle(dev_loader_path)
    #     logger.info('read test_loader.pkl...')
    #     test_loader = load_pickle(test_loader_path)
    #     dataloaders = [train_loader, dev_loader, test_loader]
    return dataloaders, processor.word_vocab_num


def load_model(file, _type):
    """
    读取保存的模型
    :param file: 模型地址
    :param _type: 加载类型，CPU or GPU号
    :return:
    """
    if '.pt' not in file:
        file += '.pt'
    if type(_type) == str and _type.lower() == 'cpu':
        ckpt = torch.load(file, map_location='cpu')
    else:
        try:
            device = torch.device("cuda")
            ckpt = torch.load(file, map_location=device)
        except:
            print("请选择加载的机器类型，CPU or GPU型号 ！！")
            return None
    if list(ckpt.keys())[0][:6] == 'module':
        # 使用多GPU时加载model
        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        return new_state_dict
    else:
        # 使用单GPU训练时加载model
        return ckpt
