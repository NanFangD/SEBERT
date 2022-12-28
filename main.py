"""
 -*- coding: utf-8 -*-
@Time    : 2022/9/26 21:28
@Author  : nanfang
@File    : main.py
"""
import os
import torch
import transformers
from loguru import logger
from torch import optim
from transformers import BertConfig
# from model.BERT_CRF import BertCRF
from model.BERT import BertSoftmax
from model.BERT_CRF import BertCRF
from model.LEBert_CRF import LEBertCrf
from metrics.ner_metrics import SeqEntityScore
from model.LEBert_Softmax import LEBertSoftmax
from model.BILSTM_CRF import LSTMCRF
from modules.txt_op import read_txt_data
from modules.utils import to_GPU, to_GPUS, narrow_setup
from processors.process_data_op import get_dataloaders
from config import configs, set_seed, set_logger
from processors.train_op import train_epoch, dev_epoch, test_epoch
from processors.vocab import Vocabulary
from processors.word_embedding import get_word_embedding


def get_optimizer(model):
    t_total = len(train_loader) // configs.grad_acc_step * configs.epochs
    warmup_steps = int(t_total * configs.warmup_proportion)
    # if len(configs.device) > 1:  # 判断是否使用多GPU
    #     module = model.module
    # else:
    module = model
    # todo 检查
    no_bert = ["word_embedding_adapter", "word_embeddings", 'crf', 'classifier', 'layerNorm']
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # lr不需要衰减的
    optimizer_grouped_parameters = [
        # bert no_decay
        {"params": [p for n, p in module.named_parameters()
                    # 如果no_bert 里面有 module的参数就返回False
                    # 如果no_decay里面有module的参数就返回True
                    if (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and any(
                nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': configs.lr},
        # bert decay
        {"params": [p for n, p in module.named_parameters()
                    if
                    (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and not any(
                        nd in n for nd in no_decay)],
         "weight_decay": configs.weight_decay, 'lr': configs.lr},
        # other no_decay
        {"params": [p for n, p in module.named_parameters()
                    if any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and any(
                nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": configs.adapter_lr},
        # other decay
        {"params": [p for n, p in module.named_parameters() if
                    any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and not any(
                        nd in n for nd in no_decay)],
         "weight_decay": configs.weight_decay, "lr": configs.adapter_lr},
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=configs.lr, eps=configs.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler


MODEL_CLASS = {
    'lebert-softmax': LEBertSoftmax,
    'lebert-crf': LEBertCrf,
    'bert-crf': BertCRF,
    'bilstm-crf': LSTMCRF,
    'bert': BertSoftmax
}


def train():
    if 'lebert' in configs.model_name:
        config = BertConfig.from_pretrained(configs.pretrain_model_path)
        config.num_labels = label_vocab.size  # 标签数量
        config.add_layer = configs.add_layer  # 将词汇信息添加到BERT的第几层
        config.word_vocab_size = model_word_embedding.shape[0]  # 加载的词向量大小
        config.word_embed_dim = model_word_embedding.shape[1]  # 加载的单个词向量长度
        model = MODEL_CLASS[configs.model_name].from_pretrained(configs.pretrain_model_path,
                                                                config=config)  # 根据configs.model_name选择model
        model.word_embeddings.weight.data.copy_(torch.from_numpy(model_word_embedding))  # 将自定义的向量词典放入nn.embeddings
        optimizer, scheduler = get_optimizer(model)
        optimizers = [optimizer, scheduler]
    elif 'bert' in configs.model_name:
        model = MODEL_CLASS[configs.model_name].from_pretrained(configs.pretrain_model_path,
                                                                num_labels=label_vocab.size)
        optimizer, scheduler = get_optimizer(model)
        optimizers = [optimizer, scheduler]
    elif 'lstm' in configs.model_name:
        model = MODEL_CLASS[configs.model_name](embedding_dim=configs.embedding_dim, hidden_dim=configs.hidden_dim,
                                                vocab_size=word_vocab_num,
                                                num_labels=label_vocab.size,
                                                bidirectional=configs.bidiretional)
        optimizer = optim.Adam(model.parameters(), lr=configs.lstm_lr)
        optimizers = [optimizer]
    else:
        print("没有选择模型！")
        return
    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=configs.ignore_index)
    model.zero_grad()

    if configs.device == '-1':
        # 抢占GPU
        gpu_id = narrow_setup()
        # 获取可以使用的的GPU_id
        configs.device = f'{gpu_id[0]}'
    if torch.cuda.is_available():
        logger.info(f"GPU可用，挂载{configs.device}号GPU！")
    else:
        logger.info("GPU不可用，使用CPU！")
    if len(configs.device) > 1 and configs != '-1':  # 判断是否使用多GPU
        model = to_GPUS(model)
    else:
        model = to_GPU(model)

    max_devf1_score = 0  # 记录最大的验证集f1分数
    devacc_score = 0
    devrecall_score = 0
    max_testf1_score = 0  # 记录最大的测试集f1分数
    testacc_score = 0
    testrecall_score = 0
    for epoch in range(1, configs.epochs + 1):
        metric.__init__(label_vocab.idx2token, markup=configs.markup)  # 初始化评价指标类
        train_epoch(epoch=epoch, dataloader=train_loader, model=model, metric=metric, optimizer=optimizers,
                    loss_function=loss_function)
        metric.__init__(label_vocab.idx2token, markup=configs.markup)
        dev_results = dev_epoch(epoch=epoch, dataloader=dev_loader, model=model, metric=metric, loss_function=loss_function)
        # 保存验证集中f1值最大的模型
        if dev_results['f1'] > max_devf1_score:
            max_devf1_score = dev_results['f1']
            devacc_score = dev_results['acc']
            devrecall_score = dev_results['recall']
        metric.__init__(label_vocab.idx2token, markup=configs.markup)
        test_results = test_epoch(epoch=0, dataloader=test_loader, model=model, metric=metric, loss_function=loss_function)
        if test_results['f1'] > max_testf1_score:
            max_testf1_score = test_results['f1']
            testacc_score = test_results['acc']
            testrecall_score = test_results['recall']
            # save_model(_model=model, file=configs.model_save_file)
        logger.info('{}-epoch Max Dev F1_Score is {:.3f}'.format(epoch, max_devf1_score * 100))
        logger.info('Dev acc: {:.3f},recall:{:.3f}'.format(devacc_score * 100, devrecall_score * 100))
        logger.info('{}-epoch Max Test F1_Score is {:.3f}'.format(epoch, max_testf1_score * 100))
        logger.info('Test acc: {:.3f},recall:{:.3f}'.format(testacc_score * 100, testrecall_score * 100))


if __name__ == '__main__':

    # 加载基本日志信息，设置GPU
    set_logger()
    # 设置随机种子
    # set_seed(configs.seed)

    if configs.load_word_embed:
        # 获取词向量
        model_word_embedding, word2vocab, embed_dim, trie_tree = get_word_embedding()
        # 加载数据
        [train_loader, dev_loader, test_loader], word_vocab_num = get_dataloaders(model_word_embedding, word2vocab,
                                                                                  embed_dim, trie_tree)
    else:
        [train_loader, dev_loader, test_loader], word_vocab_num = get_dataloaders()
    train_size = train_loader.dataset.len  # 训练集长度，用于微调BERT
    # 算出label的个数
    label_vocab = Vocabulary(read_txt_data(configs.label_file)[0])
    # 评价指标
    metric = SeqEntityScore(label_vocab.idx2token, markup=configs.markup)
    configs.ignore_index = label_vocab.convert_tokens_to_ids('[PAD]')
    # # 加载模型
    logger.info(f'加载预训练模型{configs.model_name}...')
    # 开始训练
    train()
    # 输出日志名称
    logger.info(f'日志名称：{configs.time} {configs.model_name}.log')


