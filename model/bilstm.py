import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from config import Config, PAD_IDX

class BiLSTM(nn.Module):
    def __init__(self, config: Config):
        super(BiLSTM, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Linear(config.hidden_size * 2, 1, bias=False)
        # self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x, seq_lens=None):
        emb = self.embedding(x)  # [batch_size, L, embeding]

        # 计算mask
        B, L, _ = emb.size()
        mask = torch.arange(L, device=emb.device).expand(B, L)
        mask = mask < seq_lens.unsqueeze(1) # [batch_size, L]

        packed_emb = pack_padded_sequence(emb, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        H, _ = self.lstm(packed_emb)
        H, _ = pad_packed_sequence(H, batch_first=True, padding_value=PAD_IDX, total_length=L)
        # [batch_size, L, hidden_size * 2]

        out = torch.sum(H, dim=1) / seq_lens.unsqueeze(-1)  # [batch_size, hidden_size * 2]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [batch_size, num_classes]
        return out
    
if __name__ == "__main__":
    cfg = Config(dataset="./mydatasets/jieba", embedding="random", dataset_method="jieba")
    import json
    with open(cfg.vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        cfg.n_vocab = data["vocab_size"]
    model = BiLSTM(cfg)
    input_ids = torch.tensor([[1,2,3,4,0], [5,6,7,0,0]], dtype=torch.long) # (B=2, L=5)
    seq_lens = torch.tensor([4, 3], dtype=torch.long) # (B,)
    output = model(input_ids, seq_lens)
    print(f"输出形状: {output.shape}") # 应为 (B=2, num_classes=15)