"""
 -*- coding: utf-8 -*-
@Time    : 2022/9/26 21:28
@Author  : nanfang
@File    : config.py
"""
import argparse
import random
import os
import time
from os.path import join
import torch
import numpy as np
from loguru import logger


def get_configs():
    parser = argparse.ArgumentParser()
    ## 需要随时更改的
    parser.add_argument("--device", type=str, default='6', help="选择训练的卡号,-1代表开启抢占GPU！")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--epochs", type=int, default=30)
    # parser.add_argument("--batch_size", type=int, default=20)

    parser.add_argument("--dataset_path", type=str, default='dataset/chinese/CVOID/',
                        choices=['resume', "weibo", 'ontonote4', 'msra', 'CVOID'
                                                                         'CoNLL-2000', 'CoNLL-2003'], help='数据集存放路径')
    parser.add_argument('--markup', default='bio', type=str, choices=['bios', 'bio'], help='数据集的标注方式')
    parser.add_argument("--model_name", type=str, default="bilstm-crf",
                        choices=['bert-crf', 'lebert-softmax', 'lebert-crf', 'bilstm-crf', 'bert'], help='运行模型名称')
    parser.add_argument("--add_weights", type=bool, default=True, help='是否在模型中加入全局信息')
    parser.add_argument("--add_mix_loss", type=bool, default=True, help='是否在模型中混合loss')
    parser.add_argument("--load_word_embed", default=False, help='是否加载预训练的词向量')
    parser.add_argument("--overwrite", action='store_true', default=False, help="覆盖数据处理的结果")
    # parser.add_argument("--pretrain_model_path", type=str, default="pretrained_models/BERT/chinese_roberta_wwm_large_ext")
    parser.add_argument("--pretrain_model_path", type=str, default="pretrained_models/BERT/bert-base-chinese")
    # parser.add_argument("--pretrain_model_path", type=str, default="pretrained_models/BERT/bert-base-cased")
    parser.add_argument("--max_word_num", type=int, default=3, help="每个汉字最多融合多少个词汇信息")
    parser.add_argument("--max_scan_num", type=int, default=1000000, help="取预训练词向量的前max_scan_num个构造字典树")
    parser.add_argument("--max_seq_len", type=int, default=200, help="输入的最大长度")
    # LSTM 参数
    parser.add_argument('--embedding_dim', default=200, type=float, help='embedding_dim')
    parser.add_argument('--hidden_dim', default=256, type=float, help='hidden_dim')
    parser.add_argument('--bidiretional', default=True, type=bool, help='bidiretional')
    parser.add_argument("--lstm_lr", default=5e-4, type=float, help="lstm learning rate")
    if 'resume' in parser.parse_args().dataset_path:
        #  resume 学习率
        parser.add_argument("--adapter_lr", default=5e-6, type=float, help="其他模型参数的学习率")
        parser.add_argument("--lr", type=float, default=5e-6, help='Bert的学习率')
    elif 'weibo' in parser.parse_args().dataset_path:
        #  weibo学习率
        parser.add_argument("--adapter_lr", default=1e-4, type=float, help="其他模型参数的学习率")
        parser.add_argument("--lr", type=float, default=1e-5, help='Bert的学习率')
        # parser.add_argument("--adapter_lr", default=5e-5, type=float, help="其他模型参数的学习率")
        # parser.add_argument("--lr", type=float, default=5e-6, help='Bert的学习率')
    elif 'ontonote4' in parser.parse_args().dataset_path:
        #  ontonote4学习率
        parser.add_argument("--adapter_lr", default=1e-4, type=float, help="其他模型参数的学习率")
        parser.add_argument("--lr", type=float, default=1e-5, help='Bert的学习率')
    elif 'msra' in parser.parse_args().dataset_path:
        #  msra学习率
        parser.add_argument("--adapter_lr", default=1e-5, type=float, help="其他模型参数的学习率")
        parser.add_argument("--lr", type=float, default=3e-5, help='Bert的学习率')
    elif 'COVID' in parser.parse_args().dataset_path:
        #  COVID学习率
        parser.add_argument("--adapter_lr", default=5e-4, type=float, help="其他模型参数的学习率")
        parser.add_argument("--lr", type=float, default=1e-5, help='Bert的学习率')
        # parser.add_argument("--adapter_lr", default=1e-4, type=float, help="其他模型参数的学习率")
        # parser.add_argument("--lr", type=float, default=3e-5, help='Bert的学习率')
    else:
        parser.add_argument("--adapter_lr", default=1e-4, type=float, help="其他模型参数的学习率")
        parser.add_argument("--lr", type=float, default=1e-5, help='Bert的学习率')
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--eps', default=1.0e-08, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument('--hidden_dropout_prob', default=0.2, type=float, help='Dropout')
    # 不需要调整的
    parser.add_argument('--dataset_language', default='chinese', type=str, choices=['chinese', 'english'],
                        help='数据集的语言')
    parser.add_argument("--ignore_index", type=int, default=0, help='残差网络忽略的标签')
    parser.add_argument("--num_labels", type=int, default=0, help='标签数量,无需修改，自动设置的')
    parser.add_argument("--drop_last", type=bool, default=False, help='是否删除最后不足一个batch的数据')
    parser.add_argument("--shuffle_data", type=bool, default=True, help='batch数据是否随机分配')
    parser.add_argument("--train_file", type=str, default="dataset/chinese/ontonote4/train.json")
    parser.add_argument("--dev_file", type=str, default="dataset/chinese/ontonote4/dev.json")
    parser.add_argument("--test_file", type=str, default="dataset/chinese/ontonote4/test.json")
    parser.add_argument("--label_file", type=str, default="dataset/chinese/ontonote4/labels.txt")
    parser.add_argument("--output_path", type=str, default='output/', help='输出模型与预处理数据的存放位置')
    parser.add_argument("--word_embed_path", type=str,
                        default='pretrained_models/emb/tencent/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt',
                        help='腾讯预训练词向量路径')
    parser.add_argument('--add_layer', default=1, type=str, help='在bert的第几层后面融入词汇信息')
    parser.add_argument('--grad_acc_step', default=1, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--full_fine_tuning', default=True, type=bool, help='BERT是否微调')

    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='梯度裁剪阈值')
    parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--time', type=str, default=time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()), help="当前时间")
    # parser.add_argument('--time', type=str, default=time.strftime("%Y-%m-%d", time.localtime()), help="当前时间")
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.')
    print("初始化configs:")
    return parser.parse_args()


configs = get_configs()


def set_logger():
    # 新建日志文件
    logger.add(join(configs.output_path + '/log', f'{configs.time} {configs.model_name}.log'))
    # 设置GPU号
    if configs.device != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{configs.device}'
    if 'chinese' in configs.dataset_path:
        configs.dataset_langulage = 'chinese'
    elif 'english' in configs.dataset_path:
        configs.dataset_langulage = 'english'
    else:
        print("数据集语言选择错误，请修改后重新运行代码！")
        exit()
    logger.info("start training")
    logger.info(f"batch_size:{configs.batch_size}")
    logger.info(f"epochs:{configs.epochs}")
    logger.info(f"数据集存放路径:{configs.dataset_path}", )
    logger.info(f"预训练模型路径:{configs.pretrain_model_path}", )
    logger.info(f"词向量路径:{configs.word_embed_path}")
    logger.info(f"输出位置:{configs.output_path}", )
    logger.info(f"输入最大长度:{configs.max_seq_len}", )
    logger.info(f"数据集标注方式:{configs.markup}", )
    logger.info(f"是否在模型中加入全局信息:{configs.add_weights}", )
    logger.info(f"是否在模型中混合loss:{configs.add_mix_loss}", )
    if 'lstm' in configs.model_name:
        logger.info(f"embedding_dim:{configs.embedding_dim}", )
        logger.info(f"hidden_dim:{configs.hidden_dim}", )
        logger.info(f"bidiretional:{configs.bidiretional}", )
        logger.info(f"lstm_lr:{configs.lstm_lr}", )
    elif 'bert' in configs.model_name:
        logger.info(f"其他模型参数的学习率:{configs.adapter_lr}", )
        logger.info(f"Bert的学习率:{configs.lr}", )
    else:
        print("模型尚未选择清楚！")
        exit()
    configs.train_file = configs.dataset_path + 'train.json'
    if 'COVID' in configs.dataset_path or 'CoNLL-2000' in configs.dataset_path:
        configs.dev_file = configs.dataset_path + 'test.json'
    else:
        configs.dev_file = configs.dataset_path + 'dev.json'
    configs.test_file = configs.dataset_path + 'test.json'
    configs.label_file = configs.dataset_path + 'labels.txt'


def set_seed(seed=106524):
    """
    设置整个开发环境的seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    logger.info(f"随机种子设置为:{seed}")


if __name__ == '__main__':
    configs = configs
