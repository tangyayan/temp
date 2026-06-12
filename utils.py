import json

def load_jsonl(path):
    """读取 JSONL 文件，返回列表"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

def analysis_attention(model_path):
    """分析 BiLSTM_att 模型的注意力权重分布"""
    import torch
    from model.bilstm_att import BiLSTM_att
    from config import Config

    cfg = Config(dataset="./mydatasets/jieba", embedding="random", dataset_method="jieba")
    with open(cfg.vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        cfg.n_vocab = data["vocab_size"]
    from dataset import TNEWSDataset
    dataset = TNEWSDataset(cfg.test_path, cfg)
    test_data = dataset[53] # 取测试集的第一个样本

    model = BiLSTM_att(cfg)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    _, alpha = model(test_data["input_ids"].unsqueeze(0), torch.tensor([test_data["seq_len"]]))
    alpha = alpha.squeeze(0).squeeze(-1).tolist() # (L,)
    avg_distribution = 1 / test_data["seq_len"]
    print(test_data["tokens"])
    print(test_data["input_ids"])
    print(f"平均注意力权重分布: {avg_distribution:.4f}")
    print(f"注意力权重分布: {alpha}")

if __name__ == "__main__":
    analysis_attention("./result/BiLSTM_att/best_model.pt")