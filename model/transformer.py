import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import Config, PAD_IDX

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.tensor([[pos / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)] 
                            for pos in range(max_len)], dtype=torch.float)
        self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = torch.cos(self.pe[:, 1::2])
        self.pe = self.pe.unsqueeze(0)  # [1, max_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, L, _ = x.size()
        x = x + self.pe[:, :L, :].to(x.device)  # 添加位置编码
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, seq_lens):
        B, L, _ = x.size()
        mask = torch.arange(L, device=x.device).expand(B, L)
        mask = mask < seq_lens.unsqueeze(1) # [B, L]
        
        q = self.W_q(x).view(B, -1, self.num_heads, self.d_head).transpose(1, 2)  # [B, H, L, D_head]
        k = self.W_k(x).view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, -1, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.d_head ** -0.5  # [B, H, L, L]
        scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))  # 填充无效位置
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, L, L]
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, D_head]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, D_model]
        attn_output = self.out_linear(attn_output) # [B, L, D_model]
        
        out = self.dropout(attn_output) + x  # 残差连接
        out = self.layer_norm(out)
        return out

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        # 不需要mask，因为前馈网络是逐位置独立的

    def forward(self, x):
        out = self.fc1(x) # [B, L, hidden]
        out = F.relu(out)
        out = self.fc2(out)   # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(config.pad_size, config.embed, config.dropout) # [cls]
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.fc_out = nn.Linear(config.embed, config.num_classes)

        if config.pooling == 'cls':
            self.cls_emb = nn.Parameter(torch.randn(1, 1, config.embed))  # [1, 1, D_model]
        elif config.pooling == 'mean':
            self.mean_pooling = MeanPooling()
        else:
            self.attention_pooling = AttentionPooling(config.embed)
        self.pooling = config.pooling

    def forward(self, x, seq_lens):
        x = self.embedding(x)  # [B, L, D]

        # cls
        if self.pooling == 'cls': # 已经在dataset中截断了，所以这里直接拼接cls
            cls = self.cls_emb.expand(x.size(0), -1, -1)  # [B, 1, D_model]
            x = torch.cat([cls, x], dim=1)  # [B, L+1, D]
            seq_lens = seq_lens + 1  # cls占一个位置

        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, seq_lens) # [B, L, D]
        
        # cls
        if self.pooling == 'cls':
            x = x[:, 0, :]  # [B, D_model]

        # attention pooling
        if self.pooling == 'attention':
            x = self.attention_pooling(x, seq_lens)  # [B, D_model]

        # mean pooling
        if self.pooling == 'mean':
            x = self.mean_pooling(x, seq_lens)  # [B, D_model]
        
        x = self.fc_out(x)  # [B, num_classes]
        return x

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.w = nn.Linear(d_model, 1)

    def forward(self, x, seq_lens):
        scores = self.w(x).squeeze(-1)  # [B, L]
        B, L, _ = x.size()
        mask = torch.arange(L, device=x.device).expand(B, L)
        mask = mask < seq_lens.unsqueeze(1) # [B, L]
        scores = scores.masked_fill(~mask, float('-inf'))  # 填充无效位置
        attn_weights = torch.softmax(scores, dim=-1)  # [B, L]
        pooled = attn_weights.unsqueeze(-1) * x  # [B, L, D]
        pooled = pooled.sum(dim=1)  # [B, D]
        return pooled

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x, seq_lens):
        B, L, _ = x.size()
        mask = torch.arange(L, device=x.device).expand(B, L)
        mask = mask < seq_lens.unsqueeze(1) # [B, L]
        masked_x = x.masked_fill(~mask.unsqueeze(-1), 0.0)  # [B, L, D]
        sum_x = masked_x.sum(dim=1)  # [B, D]
        mean_x = sum_x / seq_lens.unsqueeze(1)  # [B, D]
        return mean_x

class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super(TransformerBlock, self).__init__()
        
        self.mha = MultiHeadAttention(config.embed, config.num_heads, config.dropout)
        self.ffn = Position_wise_Feed_Forward(config.embed, config.hidden_size, config.dropout)
    def forward(self, x, seq_lens):
        x = self.mha(x, seq_lens)
        x = self.ffn(x) # [B, L, D_model]
        return x


if __name__ == "__main__":
    # 测试模型
    conf = Config(dataset="./mydatasets/char", embedding="random", dataset_method="char")
    import json
    with open(conf.vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        conf.n_vocab = data["vocab_size"]
    model = Transformer(conf)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 0, 0]])  # [B, L]
    seq_lens = torch.tensor([5, 3])  # [B]
    output = model(input_ids, seq_lens)
    print(output.shape)  # 应该是 [B, num_classes]