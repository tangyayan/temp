"""
plot.py — 训练曲线可视化
用法：
  python plot.py --model_name BiLSTM --result_dir ./result
  python plot.py                                      # 使用默认路径
"""

import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")           # 无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dataset import LABELSTR

C_TRAIN_STEP  = "#A8C7E8"   # 浅蓝：step 级 train 折线
C_TRAIN_EPOCH = "#1A6CB0"   # 深蓝：epoch 级 train 标注
C_DEV_STEP    = "#F4A97E"   # 浅橙：step 级 dev 折线
C_DEV_EPOCH   = "#D94F00"   # 深橙：epoch 级 dev 标注
C_F1          = "#2CA02C"   # 绿：dev macro-F1
C_F1_STEP     = "#98DF8A"   # 浅绿：step 级 dev macro-F1
C_BEST        = "#9467BD"   # 紫：最优 epoch 竖线


def _annotate_epoch_points(ax, x_list, y_list, color, fmt=".4f",
                            fontsize=7, offset=(0, 6)):
    """在每个 epoch 点上标注数值。"""
    for x, y in zip(x_list, y_list):
        ax.annotate(
            f"{y:{fmt}}",
            xy=(x, y), xytext=(offset[0], offset[1]),
            textcoords="offset points",
            fontsize=fontsize, color=color,
            ha="center", va="bottom",
        )


def _draw_epoch_vlines(ax, epoch_steps, ymin, ymax, alpha=0.15):
    """每个 epoch 结束处画一条淡灰竖线。"""
    for s in epoch_steps:
        ax.axvline(x=s, color="gray", linewidth=0.6, linestyle="--", alpha=alpha)


def plot_training_curves(history: dict, save_dir: str, model_name: str = ""):
    """
    history 需含：
      step 级: step_steps, step_win_loss, step_win_acc, step_dev_acc
      epoch 级: epoch_steps, train_loss, train_acc,
                dev_loss, dev_acc, dev_macro_f1
    """
    # 数据提取
    # step 级
    ss        = history.get("step_steps",    [])
    s_loss    = history.get("step_win_loss", [])
    s_tacc    = history.get("step_win_acc",  [])
    s_dloss   = history.get("step_dev_loss", [])
    s_dacc    = history.get("step_dev_acc",  [])
    s_df1     = history.get("step_dev_f1",   [])

    # epoch 级（x 轴对齐到 global_step）
    es        = history.get("epoch_steps",   [])
    e_tloss   = history.get("train_loss",    [])
    e_tacc    = history.get("train_acc",     [])
    e_dloss   = history.get("dev_loss",      [])
    e_dacc    = history.get("dev_acc",       [])
    e_f1      = history.get("dev_macro_f1",  [])

    # 最优 epoch（dev_macro_f1 最大处）
    # best_epoch_idx = int(np.argmax(e_f1)) if e_f1 else None
    # best_step      = es[best_epoch_idx] if best_epoch_idx is not None else None
    best_step_idx = int(np.argmax(s_df1)) if s_df1 else None
    best_step     = ss[best_step_idx] if best_step_idx is not None else None

    have_steps = len(ss) > 0

    # 画布
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Training Curves{'  —  ' + model_name if model_name else ''}",
        fontsize=14, fontweight="bold", y=0.98
    )
    ax_loss, ax_tacc, ax_dacc, ax_f1 = axes.flatten()

    # 公共设置
    xlabel = "Global Step"
    for ax in axes.flatten():
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if best_step is not None:
            ax.axvline(x=best_step, color=C_BEST, linewidth=1.2,
                       linestyle=":", alpha=0.7, label="Best step")

    def _epoch_x(ax):
        """在 x 轴下方加 epoch 刻度标签。"""
        if es:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(es)
            ax2.set_xticklabels([f"E{i+1}" for i in range(len(es))],
                                fontsize=7, color="gray")
            ax2.tick_params(axis="x", length=0, pad=1)
            ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Loss
    ax = ax_loss
    ax.set_title("Loss", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=9)

    if have_steps:
        ax.plot(ss, s_loss, color=C_TRAIN_STEP, linewidth=1.2,
                alpha=0.8, label="Train loss (step)")
        ax.plot(ss, s_dloss, color=C_DEV_STEP, linewidth=1.2,
                alpha=0.8, label="Dev loss (step)")
    if es:
        ax.plot(es, e_tloss, color=C_TRAIN_EPOCH, linewidth=2.0,
                marker="o", markersize=5, label="Train loss (epoch)")
        ax.plot(es, e_dloss, color=C_DEV_EPOCH, linewidth=2.0,
                marker="s", markersize=5, linestyle="--", label="Dev loss (epoch)")
        _draw_epoch_vlines(ax, es,
                           min(e_tloss + e_dloss + s_loss) if s_loss else min(e_tloss + e_dloss),
                           max(e_tloss + e_dloss + s_loss) if s_loss else max(e_tloss + e_dloss))
        _annotate_epoch_points(ax, es, e_tloss, C_TRAIN_EPOCH)
        _annotate_epoch_points(ax, es, e_dloss, C_DEV_EPOCH, offset=(0, -12))

    ax.legend(fontsize=8, loc="upper right")
    _epoch_x(ax)

    # Train Acc
    ax = ax_tacc
    ax.set_title("Train Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_ylim(-0.02, 1.08)

    if have_steps:
        ax.plot(ss, s_tacc, color=C_TRAIN_STEP, linewidth=1.2,
                alpha=0.8, label="Train acc (step)")
    if es:
        ax.plot(es, e_tacc, color=C_TRAIN_EPOCH, linewidth=2.0,
                marker="o", markersize=5, label="Train acc (epoch)")
        _draw_epoch_vlines(ax, es, 0, 1)
        _annotate_epoch_points(ax, es, e_tacc, C_TRAIN_EPOCH)

    ax.legend(fontsize=8, loc="lower right")
    _epoch_x(ax)

    # Dev Acc
    ax = ax_dacc
    ax.set_title("Dev Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_ylim(-0.02, 1.08)

    if have_steps:
        ax.plot(ss, s_dacc, color=C_DEV_STEP, linewidth=1.2,
                alpha=0.8, label="Dev acc (step)")
    if es:
        ax.plot(es, e_dacc, color=C_DEV_EPOCH, linewidth=2.0,
                marker="s", markersize=5, linestyle="--", label="Dev acc (epoch)")
        _draw_epoch_vlines(ax, es, 0, 1)
        _annotate_epoch_points(ax, es, e_dacc, C_DEV_EPOCH)

    ax.legend(fontsize=8, loc="lower right")
    _epoch_x(ax)

    # Dev Macro-F1
    ax = ax_f1
    ax.set_title("Dev Macro-F1  (epoch)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Macro-F1", fontsize=9)
    ax.set_ylim(-0.02, 1.08)

    if have_steps:
        ax.plot(ss, s_df1, color=C_F1_STEP, linewidth=1.2,
                alpha=0.8, label="Dev macro-F1 (step)")
    if es:
        ax.plot(es, e_f1, color=C_F1, linewidth=2.0,
                marker="D", markersize=5, label="Dev macro-F1")
        _draw_epoch_vlines(ax, es, 0, 1)
        _annotate_epoch_points(ax, es, e_f1, C_F1)

        # 最优点特别标注
        # if best_epoch_idx is not None:
        #     bx, by = es[best_epoch_idx], e_f1[best_epoch_idx]
        #     ax.scatter([bx], [by], color=C_BEST, s=80, zorder=5,
        #                label=f"Best F1={by:.4f} (E{best_epoch_idx+1})")
        #     ax.annotate(
        #         f"Best\nF1={by:.4f}",
        #         xy=(bx, by), xytext=(12, -20),
        #         textcoords="offset points",
        #         fontsize=8, color=C_BEST,
        #         arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.0),
        #     )
        if best_step_idx is not None:
            bx, by = ss[best_step_idx], s_df1[best_step_idx]
            ax.scatter([bx], [by], color=C_BEST, s=80, zorder=5,
                       label=f"Best F1={by:.4f} (S{best_step_idx+1})")
            ax.annotate(
                f"Best\nF1={by:.4f}",
                xy=(bx, by), xytext=(12, -20),
                textcoords="offset points",
                fontsize=8, color=C_BEST,
                arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.0),
            )

    ax.legend(fontsize=8, loc="lower right")
    _epoch_x(ax)

    # 图例补充说明
    legend_patches = [
        mpatches.Patch(color=C_BEST,  alpha=0.7, label="Best epoch (dotted line)"),
        mpatches.Patch(color="gray",  alpha=0.4, label="Epoch boundary (dashed)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=2, fontsize=8, framealpha=0.6,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = os.path.join(save_dir, f"training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"训练曲线已保存至: {out_path}")


def draw_confusion_matrix(test_result_path: str, save_dir: str):
    if not os.path.exists(test_result_path):
        print(f"找不到测试结果文件: {test_result_path}")
        return
    with open(test_result_path, "r") as f:
        test_metrics = json.load(f)
    cm = np.array(test_metrics["confusion_matrix"])

    # 归一化
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm = cm / np.where(cm_sum == 0, 1, cm_sum)  # 避免除以零

    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELSTR)
    fig, ax = plt.subplots(figsize=(12, 10)) 
    disp.plot(ax=ax, xticks_rotation=45) 
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    print(f"混淆矩阵已保存至: {os.path.join(save_dir, 'confusion_matrix.png')}")

def main():
    parser = argparse.ArgumentParser(description="绘制训练曲线")
    parser.add_argument("--model_name", default="BiLSTM", help="模型文件名")
    parser.add_argument("--result_dir",  default="./result", help="输出目录")

    args = parser.parse_args()

    out_dir = os.path.join(args.result_dir, args.model_name)      # 输出目录：result/模型名称/
    history_path = os.path.join(out_dir, "history.json")          # history 文件路径：result/模型名称/history.json
    test_result_path = os.path.join(out_dir, "test_results.json") # 测试结果文件路径：result/模型名称/test_results.json

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(history_path):
        print(f"找不到 history 文件: {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    plot_training_curves(history, save_dir=out_dir, model_name=args.model_name)
    draw_confusion_matrix(test_result_path, save_dir=out_dir)


if __name__ == "__main__":
    main()