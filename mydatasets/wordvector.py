import json
import numpy as np
import torch
from config import PAD_TOKEN, NUM_TOKEN, PAD_IDX, UNK_TOKEN

DATA_DIR = "./mydatasets/char"

# 读取词表
with open(f"{DATA_DIR}/vocab.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)["word2idx"]

# 读取预训练向量
embedding_dict = {}

with open("./mydatasets/sgns.sogou.char", "r", encoding="utf-8") as f:
    vocab_size, embed_dim = map(int, f.readline().split())

    for line in f:
        items = line.rstrip().split()

        token = items[0]
        vector = np.asarray(items[1:], dtype=np.float32)

        embedding_dict[token] = vector

# 构造Embedding矩阵
embedding_matrix = np.random.normal(loc=0, scale=0.1, size=(len(word2idx), embed_dim)).astype(np.float32)

# 特殊token
embedding_matrix[PAD_IDX] = np.zeros(embed_dim)
embedding_matrix[word2idx[NUM_TOKEN]] = embedding_dict["1"]

invector_cnt = 0
for token, idx in word2idx.items():
    if token in [PAD_TOKEN, NUM_TOKEN, UNK_TOKEN]:
        continue
    if token in embedding_dict:
        embedding_matrix[idx] = embedding_dict[token]
    else:
        invector_cnt += 1
        if invector_cnt <= 10:
            print(f"未找到预训练向量的token: {token}")

torch.save(torch.tensor(embedding_matrix), f"{DATA_DIR}/sogou.pt")

print(embedding_matrix.shape)
print(f"未找到预训练向量的token数量: {invector_cnt}/{len(word2idx)}")