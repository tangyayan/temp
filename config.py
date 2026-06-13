import torch
import numpy as np


NUM_TOKEN      = "<NUM>"
PAD_TOKEN      = "<PAD>"
PAD_IDX        = 0
UNK_TOKEN      = "<UNK>"
DATASET_SEED   = 1234
DATASET_SPLIT    = 0.9

class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding, dataset_method):
        # init
        self.seed = 1234
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        # model
        self.model_name = 'BiLSTM'                                  # !模型名称
        self.embedding_pretrained = torch.load(f'{dataset}/{embedding}.pt', weights_only=True)\
            if embedding != 'random' else None                          # 预训练词向量
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 64

        # output
        self.save_dir = f'./result/{self.model_name}_{dataset_method}_{embedding}'    # 模型训练结果保存路径                  

        # train
        self.dropout = 0.5                                              # 随机失活
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 5e-4                                       # 学习率
        self.weight_decay = 5e-4                                        # 权重衰减（L2正则化）
        self.scheduler = {                                              # 学习率调度策略及参数
            "name": "cosine", 
            # "step_size": 5, "gamma": 0.5
        }   
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.patience = 1000                                            # 若超过 1000 batch F1还没提升，则提前结束训练
        self.grad_clip = 5.0                                            # 梯度裁剪阈值
        self.ce_weights = False                                          # 是否使用类别权重（针对类别不平衡问题）

        # train_print
        self.print_step = 10                                            # 每多少步打印一次训练状态

        #dataset
        self.n_vocab = 0                                                # 词表大小，在运行时赋值!
        self.pad_size = 128                                             # 每句话处理成的长度(长切)
        self.dataset_method = dataset_method                            # 数据集预处理方法，jieba 或 sentencepiece
        self.num_classes = 15                                           # 类别数
        if self.dataset_method == "jieba" or self.dataset_method == "char":
            self.train_path = dataset + '/train.jsonl'                                # 训练集
            self.dev_path = dataset + '/dev.jsonl'                                    # 验证集
            self.test_path = dataset + '/test.jsonl'                                  # 测试集
            self.vocab_path = dataset + '/vocab.json'                                 # 词表
