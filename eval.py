from config import Config
from model.transformer import Transformer
import torch
from train import final_test
from dataset import TNEWSDataset, collate_fn
import os

def eval():
    dataset = "./mydatasets/jieba"  
    embedding = "sogou"  # "sogou" 或 "random"
    datamethod = "jieba"  # "jieba" 或 "char" 
    model_name = "Transformer"  # "BiLSTM_att", "BiLSTM" 或 "Transformer"
    conf = Config(dataset=dataset, embedding=embedding, dataset_method=datamethod, model_name=model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # model
    model = Transformer(conf).to(device)
    model_path = os.path.join(os.path.dirname(__file__), "result", "Transformer_jieba_sogou_cls", "best_model.pt")

    # dataset
    test_dataset = TNEWSDataset(conf.test_path, conf)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size,
                               shuffle=False, collate_fn=collate_fn)
    
    test_metrics = final_test(model, model_path, test_loader, conf, conf.save_dir)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}  "
          f"Macro_F1: {test_metrics['macro_f1']:.4f}  ")
    
if __name__ == "__main__":
    eval()