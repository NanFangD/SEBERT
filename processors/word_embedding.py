"""
 -*- coding: utf-8 -*-
@Time    : 2022/9/27 19:25
@Author  : nanfang
@File    : word_embedding.py
"""
import json
import os

import numpy as np
from loguru import logger
from tqdm import tqdm, trange
from config import configs
from modules.json_op import read_json
from modules.utils import write_pickle, load_pickle
from processors.trie_tree import Trie
from processors.vocab import Vocabulary


def wordsdict(words):
    # 将单词转换成字典对应形式
    dic = {"PAD": 0, "UNK": 1}
    for i, v in enumerate(words, 2):
        dic[v] = i
    return dic


def load_word_embedding(word_embed_path, max_scan_num):
    """
    todo 存在许多单字的，考虑是否去掉
    加载前max_scan_num个词向量, 并且返回词表
    :return:
    """
    logger.info('loading word embedding from pretrain')
    word_embed_dict = dict()
    word_list = list()
    with open(word_embed_path, 'r', encoding='utf8') as f:
        for idx, line in tqdm(enumerate(f), desc='加载词向量'):
            # 只扫描前max_scan_num个词向量
            if idx > max_scan_num:
                break
            items = line.strip().split()
            if idx == 0:
                assert len(items) == 2
                num_embed, word_embed_dim = items
                num_embed, word_embed_dim = int(num_embed), int(word_embed_dim)
            else:
                assert len(items) == word_embed_dim + 1
                word = items[0]
                embedding = np.empty([1, word_embed_dim])
                embedding[:] = items[1:]
                word_embed_dict[word] = embedding
                word_list.append(word)
    logger.info('word_embed_dim:{}'.format(word_embed_dim))
    logger.info('size of word_embed_dict:{}'.format(len(word_embed_dict)))
    logger.info('size of word_list:{}'.format(len(word_list)))
    return word_embed_dict, word_list, word_embed_dim


def get_words_from_text(text, trie_tree):
    """
    找出text中所有的单词
    :param text:
    :param trie_tree:
    :return:
    """
    length = len(text)
    matched_words_set = set()  # 存储匹配到的单词
    for idx in range(length):
        sub_text = text[idx:idx + trie_tree.max_depth]
        words = trie_tree.enumerateMatch(sub_text)

        for word in words:
            matched_words_set.add(word)
    matched_words_set = list(matched_words_set)
    matched_words_set = sorted(matched_words_set)
    return matched_words_set


def get_words_from_corpus(trie_tree):
    """
    找出文件中所有匹配的单词
    :param trie_tree:
    :return:
    """
    logger.info('getting words from corpus')
    dataset = {
        'train': read_json(configs.train_file),
        'dev': read_json(configs.dev_file),
        'test': read_json(configs.test_file)
    }
    all_matched_words = set()
    for _type, datas in dataset.items():
        for v in tqdm(datas, desc=f'匹配{_type}数据集词向量'):
            matched_words = get_words_from_text(list(v['text']), trie_tree)
            for word in matched_words:
                all_matched_words.add(word)
    all_matched_words = list(all_matched_words)
    all_matched_words = sorted(all_matched_words)
    return all_matched_words


def build_trie_tree(word_list):
    """
    # todo 是否不将单字加入字典树中
    构建字典树
    :return:
    """
    logger.info('initializing trie tree')
    trie_tree = Trie()
    for word in tqdm(word_list, desc='构建字典树'):
        trie_tree.insert(word)
    return trie_tree


def init_model_word_embedding(corpus_words, word_embed_dict):
    logger.info('initializing model word embedding')
    # 构建单词和id的映射
    word_vocab = Vocabulary(corpus_words, vocab_type='word')
    # embed_dim = len(word_embed_dict.items()[1].size)
    embed_dim = next(iter(word_embed_dict.values())).size

    scale = np.sqrt(3.0 / embed_dim)
    model_word_embedding = np.empty([word_vocab.size, embed_dim])

    matched = 0
    not_matched = 0
    for idx, word in enumerate(word_vocab.idx2token):
        if word in word_embed_dict:
            model_word_embedding[idx, :] = word_embed_dict[word]
            matched += 1
        else:
            model_word_embedding[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_matched += 1
    logger.info('num of match:{}, num of not_match:{}'.format(matched, not_matched))
    return model_word_embedding, word_vocab, embed_dim


def get_word_embedding():
    """
    获取词向量
    :return:
    """
    tire_tree_path = configs.dataset_path + 'trie_tree.pkl'
    model_word_embedding_path = configs.dataset_path + 'model_word_embedding.pkl'
    word_vocab_path = configs.dataset_path + 'word_vocab.pkl'
    if configs.overwrite or (not os.path.exists(tire_tree_path) or
                             not os.path.exists(model_word_embedding_path) or
                             not os.path.exists(word_vocab_path)):
        # 第一次加载或者想要重写，需要读取一遍数据
        word_embed_dict, word_list, word_embed_dim = load_word_embedding(configs.word_embed_path, configs.max_scan_num)
        # 构建字典树
        trie_tree = build_trie_tree(word_list)
        # 找到数据集中的所有单词
        corpus_words = get_words_from_corpus(trie_tree)
        # 初始化模型的词向量
        model_word_embedding, word_vocab, embed_dim = init_model_word_embedding(corpus_words, word_embed_dict)

        logger.info('write trie_tree.pkl...')
        write_pickle(trie_tree, tire_tree_path)
        logger.info('write model_word_embedding_path.pkl...')
        write_pickle(model_word_embedding, model_word_embedding_path)
        logger.info('write wordsdict.pkl...')
        write_pickle(word_vocab, word_vocab_path)
    else:
        # 直接加载存好的pkl
        logger.info('read trie_tree.pkl...')
        trie_tree = load_pickle(tire_tree_path)
        logger.info('read wordsdict.pkl...')
        word_vocab = load_pickle(word_vocab_path)
        logger.info('read model_word_embedding_path.pkl...')
        model_word_embedding = load_pickle(model_word_embedding_path)
        embed_dim = model_word_embedding.shape[1]
    return model_word_embedding, word_vocab, embed_dim, trie_tree
