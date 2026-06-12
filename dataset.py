"""
dataset.py — TNEWS PyTorch Dataset
将预处理好的 .jsonl 文件封装为 Dataset，供 DataLoader 使用。
"""

import json
import torch
from torch.utils.data import Dataset
from config import Config, UNK_TOKEN, PAD_IDX

# TNEWS 官方 15 个标签，按 int 排序后建立连续映射
VALID_LABELS = ["100","101","102","103","104","106","107",
                "108","109","110","112","113","114","115","116"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(VALID_LABELS)}
IDX2LABEL = {i: lbl for lbl, i in LABEL2IDX.items()}
LABELSTR = ['story', 'culture', 'entertainment', 'sports', 'finance', 
            'house', 'car', 'edu', 'tech', 'military', 'travel', 'world', 
            'stock', 'agriculture', 'game']

class TNEWSDataset(Dataset):
    """
    读取数据集
    """
    def __init__(self, data_path: str, config: Config):
        self.samples = []
        if config.dataset_method == "jieba":
            with open(config.vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.vocab = data["word2idx"]

            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    label_str = str(r["label"])
                    assert(label_str in LABEL2IDX), f"标签 {label_str} 不在有效标签列表中。"
                    tokens = r["tokens"]
                    if len(tokens) > config.pad_size: # 截断
                        tokens = tokens[:config.pad_size]
                    input_ids = [self.vocab.get(t, self.vocab[UNK_TOKEN]) for t in tokens]
                    self.samples.append({
                        "input_ids":  torch.tensor(input_ids, dtype=torch.long), # (pad_size,)
                        "label":      torch.tensor(LABEL2IDX[label_str], dtype=torch.long), # 映射为 0~14 的 idx
                        "label_desc": r.get("label_desc", ""),
                        "sentence":   r.get("sentence", ""),
                        "tokens":     r.get("tokens", []),
                        "seq_len":    len(tokens),
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """将 list of dict 合并为 batch tensors。"""
    input_ids = torch.stack([x["input_ids"] for x in batch])   # (B, L)
    labels    = torch.stack([x["label"]     for x in batch])   # (B,)
    seq_lens   = torch.tensor([x["seq_len"]    for x in batch]) # (B,)
    return {"input_ids": input_ids, "labels": labels, "seq_lens": seq_lens}

from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    seq_lens = torch.tensor([len(seq) for seq in input_ids], dtype=torch.long)

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_IDX)  # (B, max_seq_len)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long) # (B,)

    return {"input_ids": input_ids, "labels": labels, "seq_lens": seq_lens}

if __name__ == "__main__":
    # 简单测试
    cfg = Config(dataset="./mydatasets/jieba", embedding="random", dataset_method="jieba")
    dataset = TNEWSDataset(cfg.test_path, cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    print(f"数据集大小: {len(dataset)}")
    print(f"第一个样例: {dataset[3]}")
    for batch in dataloader:
        print(batch["input_ids"])
        print(f"批次 input_ids 形状: {batch['input_ids'].shape}")
        print(f"批次 labels 形状: {batch['labels'].shape}")
        print(f"批次 seq_lens: {batch['seq_lens']}")
        break