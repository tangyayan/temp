import json
import numpy as np
import torch

DATA_DIR = "./mydatasets/jieba"

# 读取词表
with open(f"{DATA_DIR}/vocab.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)["word2idx"]

# 读取预训练向量
embedding_dict = {}

with open("./mydatasets/sgns.renmin.char", "r", encoding="utf-8") as f:
    vocab_size, embed_dim = map(int, f.readline().split())

    for line in f:
        items = line.rstrip().split()

        token = items[0]
        vector = np.asarray(items[1:], dtype=np.float32)

        embedding_dict[token] = vector

# 构造Embedding矩阵
embedding_matrix = np.random.normal(loc=0, scale=0.1, size=(len(word2idx), embed_dim)).astype(np.float32)

# 特殊token
embedding_matrix[word2idx["<PAD>"]] = np.zeros(embed_dim)

invector_cnt = 0
for token, idx in word2idx.items():

    if token in embedding_dict:
        embedding_matrix[idx] = embedding_dict[token]
    else:
        invector_cnt += 1

torch.save(torch.tensor(embedding_matrix), f"{DATA_DIR}/embedding.pt")

print(embedding_matrix.shape)
print(f"未找到预训练向量的token数量: {invector_cnt}/{len(word2idx)}")