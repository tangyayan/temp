"""
train.py — 训练与评估核心函数

提供：
  train_one_epoch(model, loader, optimizer, criterion, device)
      -> dict  包含 loss、accuracy

  evaluate(model, loader, criterion, device, split_name)
      -> dict  包含 loss、accuracy、macro_f1、report、confusion_matrix

  train(model, train_loader, dev_loader, cfg, save_dir)
      -> history dict  含各 epoch 指标，执行早停，保存最优权重

  final_test(model_path, test_loader, cfg, save_dir)
      -> 载入最优权重后在测试集上评估，打印混淆矩阵与分类报告
"""

import os
import time
import json
import copy
import torch
import torch.nn as nn
import numpy as np
from config import Config, PAD_IDX
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from dataset import LABELSTR

def evaluate(model, loader, criterion, device, is_report: bool = True):
    """
    在给定数据集上评估模型。
    返回：{
      "loss": float, "accuracy": float, "macro_f1": float,
      "report": str, "confusion_matrix": np.ndarray
    }
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    # debug
    min_loss = float("inf")
    max_loss = float("-inf")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            seq_lens   = batch["seq_lens"].to(device)
            logits    = model(input_ids, seq_lens)
            loss      = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if loss.item() < min_loss:
                min_loss = loss.item()
            if loss.item() > max_loss:
                max_loss = loss.item()

    n        = len(all_labels)
    avg_loss = total_loss / n
    acc      = accuracy_score(all_labels, all_preds)

    # debug
    # print(f"\nmin_loss={min_loss:.4f}  max_loss={max_loss:.4f}")

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    if not is_report:
        return {"loss": avg_loss, "accuracy": acc, "macro_f1": macro_f1}

    report = classification_report(
        all_labels, all_preds,
        labels=range(len(LABELSTR)),
        target_names=LABELSTR,
        zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    # print(f"\n[{split_name}] loss={avg_loss:.4f}  acc={acc:.4f}  macro-F1={macro_f1:.4f}")
    return {
        "loss": avg_loss, "accuracy": acc, "macro_f1": macro_f1,
        "report": report, "confusion_matrix": cm,
        "preds": all_preds, "labels": all_labels,
    }


def train(model, train_loader, dev_loader, cfg: Config):
    """
    完整训练流程
    保存训练历史 history.json
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = cfg.device
    model  = model.to(device)

    # 增加类别权重
    if cfg.ce_weights:
        train_labels = []
        for batch in train_loader:
            labels = batch["labels"].cpu().numpy().tolist()
            train_labels.extend(labels)
        weight = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # 学习率调度
    sched_name = cfg.scheduler.get("name", "step")
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs)
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.scheduler.get("step_size", 5),
            gamma=cfg.scheduler.get("gamma", 0.5))
    else:
        scheduler = None

    history = {
        # epoch
        "train_loss":    [], "train_acc":    [],
        "dev_loss":      [], "dev_acc":      [], "dev_macro_f1": [],
        "lr":            [],
        "epoch_steps":   [],   # 每个 epoch 结束时对应的 global_step
        # step
        "step_steps":    [],   # global_step 编号
        "step_win_loss": [],   # 窗口平均 train loss
        "step_win_acc":  [],   # 窗口 train acc
        "step_dev_acc":  [],   # 对应 dev acc
        "step_dev_loss": [],   # 对应 dev loss
        "step_dev_f1":   [],   # 对应 dev macro-F1
    }

    best_f1      = -1.0
    best_state   = None
    patience_step= 0
    patience     = cfg.patience 
    global_step  = 0

    print(f"\n{'='*60}")
    print(f"模型: {cfg.model_name}  |  设备: {device}")
    print(f"Epochs: {cfg.num_epochs}  lr: {cfg.learning_rate} "
          f"patience: {patience}  scheduler: {sched_name}")
    print(f"{'='*60}")

    for epoch in range(1, cfg.num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
 
        model.train()
        epoch_loss   = 0.0          # 整个 epoch 的累计 loss（按样本数加权）
        epoch_preds  = []           # 整个 epoch 的预测
        epoch_labels = []           # 整个 epoch 的真实标签
 
        win_loss   = 0.0
        win_preds  = []
        win_labels = []
        t0         = time.time()
 
        for batch in train_loader:
            model.train()
            global_step += 1
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            seq_lens   = batch["seq_lens"].to(device)
            batch_size = input_ids.size(0)
 
            optimizer.zero_grad()
            logits = model(input_ids, seq_lens)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
 
            # 累计到 epoch 统计
            epoch_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            epoch_preds.extend(preds)
            epoch_labels.extend(labels.cpu().numpy().tolist())
 
            # 累计到 print_step 窗口统计
            win_loss += loss.item() * batch_size
            win_preds.extend(preds)
            win_labels.extend(labels.cpu().numpy().tolist())
 
            if global_step % cfg.print_step == 0:
                win_n       = len(win_labels)
                win_avg_loss = win_loss / win_n
                win_acc      = accuracy_score(win_labels, win_preds)
 
                pre_dev_metrics = evaluate(model, dev_loader, criterion, device, is_report=False)
                dev_acc_tmp = pre_dev_metrics["accuracy"]

                if best_f1 < pre_dev_metrics["macro_f1"]:
                    best_f1 = pre_dev_metrics["macro_f1"]
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save(best_state, os.path.join(cfg.save_dir, "best_model.pt"))
                    patience_step = global_step
                else:
                    if global_step - patience_step >= patience:
                        break
 
                print(f"  [Step {global_step:5d} | Epoch {epoch}/{cfg.num_epochs}]  "
                      f"win_loss={win_avg_loss:.4f}  "
                      f"win_acc={win_acc:.4f}  "
                      f"dev_acc={dev_acc_tmp:.4f}  "
                      f"lr={current_lr:.2e}")
 
                history["step_steps"].append(global_step)
                history["step_win_loss"].append(win_avg_loss)
                history["step_win_acc"].append(win_acc)
                history["step_dev_acc"].append(dev_acc_tmp)
                history["step_dev_loss"].append(pre_dev_metrics["loss"])
                history["step_dev_f1"].append(pre_dev_metrics["macro_f1"])
 
                win_loss   = 0.0
                win_preds  = []
                win_labels = []
        
        if global_step - patience_step >= patience:
            print(f"\nEarly stopping triggered at global step {global_step} "
                  f"after {patience} steps without F1 improvement.")
            break

        if scheduler is not None:
            scheduler.step()
 
        epoch_n       = len(epoch_labels)
        epoch_avg_loss = epoch_loss / epoch_n
        epoch_acc      = accuracy_score(epoch_labels, epoch_preds)
        elapsed        = time.time() - t0
 
        dev_metrics = evaluate(model, dev_loader, criterion, device, "dev") # 在每个 epoch 结束时完整评估
 
        # 记录 history
        history["train_loss"].append(epoch_avg_loss)
        history["train_acc"].append(epoch_acc)
        history["dev_loss"].append(dev_metrics["loss"])
        history["dev_acc"].append(dev_metrics["accuracy"])
        history["dev_macro_f1"].append(dev_metrics["macro_f1"])
        history["epoch_steps"].append(global_step)   # epoch 对应的 step 位置
 
        print(f"Epoch {epoch:3d}/{cfg.num_epochs}  "
              f"train_loss={epoch_avg_loss:.4f}  train_acc={epoch_acc:.4f}  "
              f"dev_loss={dev_metrics['loss']:.4f}  dev_acc={dev_metrics['accuracy']:.4f}  "
              f"dev_F1={dev_metrics['macro_f1']:.4f}  "
              f"lr={current_lr:.2e}  t={elapsed:.1f}s")

    # 训练结束汇总
    print(f"\n{'='*60}")
    print(f"训练结束  最优验证集 macro-F1: {best_f1:.4f}")
    print(f"最优模型已保存至: {os.path.join(cfg.save_dir, 'best_model.pt')}")

    # 保存 history
    hist_path = os.path.join(cfg.save_dir, "history.json")
    _history_json = {k: (v.tolist() if hasattr(v, "tolist") else v)
                     for k, v in history.items()}
    with open(hist_path, "w") as f:
        json.dump(_history_json, f, indent=2)
    print(f"训练曲线已保存至: {hist_path}")

    # 恢复最优权重
    model.load_state_dict(best_state)
    return history


def final_test(model, model_path: str, test_loader, cfg: Config, save_dir: str):
    """
    载入最优权重，在测试集上做最终评估。
    输出：accuracy、macro-F1、分类报告、混淆矩阵。
    结果保存至 save_dir/test_results.json，混淆矩阵图保存至 save_dir/confusion_matrix.png
    """
    device = cfg.device
    print(f"\n{'='*60}")
    print("加载最优权重进行测试集评估 ...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device, "test")

    print("\n─── 分类报告 ───")
    print(test_metrics["report"])

    # disp = ConfusionMatrixDisplay(confusion_matrix=test_metrics["confusion_matrix"],
    #                               display_labels=LABELSTR)
    # disp.plot()
    # from matplotlib import pyplot as plt
    # plt.title("Confusion Matrix (Test Set)")
    # plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))

    # 保存结果
    result = {
        "accuracy":  test_metrics["accuracy"],
        "macro_f1":  test_metrics["macro_f1"],
        "report":    test_metrics["report"],
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
    }
    out_path = os.path.join(save_dir, "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n测试结果已保存至: {out_path}")
    print(f"{'='*60}")

    return test_metrics