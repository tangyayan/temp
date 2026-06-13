"""
main.py — 主程序入口

用法示例：
    python main.py
"""

import os
import time
import json
import torch
from torch.utils.data import DataLoader, Subset

from dataset import TNEWSDataset, collate_fn
from model.bilstm_att import BiLSTM_att
from model.bilstm import BiLSTM 
from train import train, final_test
from config import Config
from utils import save_hyperparameters

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    dataset = "./mydatasets/char"  
    embedding = "sogou"  # "sogou" 或 "random"
    datamethod = "char"  # "jieba" 或 "char" 
    conf = Config(dataset=dataset, embedding=embedding, dataset_method=datamethod)

    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with open(conf.vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        conf.n_vocab = data["vocab_size"]

    train_path = conf.train_path
    dev_path   = conf.dev_path
    test_path  = conf.test_path

    train_dataset = TNEWSDataset(train_path, conf)
    dev_dataset   = TNEWSDataset(dev_path, conf)
    test_dataset  = TNEWSDataset(test_path, conf)

    # 小样本测试
    # train_dataset = Subset(train_dataset, range(min(len(train_dataset), 10000)))
    # dev_dataset   = Subset(dev_dataset,   range(min(len(dev_dataset), 1000)))
    # test_dataset  = Subset(test_dataset,  range(min(len(test_dataset), 1000)))

    print(f"\n数据集大小: train={len(train_dataset)}  "
          f"dev={len(dev_dataset)}  test={len(test_dataset)}")

    # 按每个批次最长的填充
    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_dataset,   batch_size=conf.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=conf.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    if conf.model_name == "BiLSTM_att":
        model = BiLSTM_att(conf).to(device)
    elif conf.model_name == "BiLSTM":
        model = BiLSTM(conf).to(device)
    else:
        raise NotImplementedError(f"模型 {conf.model_name} 尚未实现")

    total_params = count_parameters(model)
    print(f"\n模型: {conf.model_name}  |  可训练参数量: {total_params:,}")

    # 创建文件夹
    os.makedirs(conf.save_dir, exist_ok=True)

    start_time = time.time()
    history = train(model, train_loader, dev_loader, conf)
    elapsed_time = time.time() - start_time

    best_model_path = os.path.join(conf.save_dir, "best_model.pt")
    final_test(model, best_model_path, test_loader, conf, save_dir=conf.save_dir)

    # 写入超参数
    save_hyperparameters(conf, embedding, elapsed_time)


if __name__ == "__main__":
    main()