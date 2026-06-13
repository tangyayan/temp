import json
from config import Config, DATASET_SEED, DATASET_SPLIT
import os

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

def save_hyperparameters(conf: Config, embedding):
    """将超参数保存到 JSON 文件"""
    
    hyperparams = {
        # 全局常量
        "DATASET_SEED": DATASET_SEED,
        "DATASET_SPLIT": DATASET_SPLIT,
        
        # 基础配置
        "seed": conf.seed,
        
        # 模型参数
        "model": {
            "model_name": conf.model_name,
            "hidden_size": conf.hidden_size,
            "num_layers": conf.num_layers,
            "hidden_size2": conf.hidden_size2,
        },
        
        # 训练参数
        "train": {
            "dropout": conf.dropout,
            "num_epochs": conf.num_epochs,
            "batch_size": conf.batch_size,
            "learning_rate": conf.learning_rate,
            "weight_decay": conf.weight_decay,
            "scheduler": conf.scheduler,
            "embed": conf.embed,
            "patience": conf.patience,
            "grad_clip": conf.grad_clip,
            "print_step": conf.print_step,
        },
        
        # 数据集参数
        "dataset": {
            "pad_size": conf.pad_size,
            "dataset_method": conf.dataset_method,
            "embedding_pretrained": embedding if embedding != 'random' else None,
        }
    }
    
    os.makedirs(conf.save_dir, exist_ok=True)
    save_path = os.path.join(conf.save_dir, "hyperparameters.json")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)
    
    print(f"超参数已保存至: {save_path}")

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
    # analysis_attention("./result/BiLSTM_att/best_model.pt")
    config = Config(dataset="./mydatasets/jieba", embedding="sogou", dataset_method="jieba")
    save_hyperparameters(config, embedding="sogou")