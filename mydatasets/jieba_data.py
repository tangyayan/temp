import json
import random
import os
from collections import Counter
import jieba
import jieba.posseg as pseg
from utils import load_jsonl
from config import NUM_TOKEN, PAD_TOKEN, UNK_TOKEN, DATASET_SEED, DATASET_SPLIT

CLEAN_DATA_DIR = "./mydatasets/tnews_cleared"
OUTPUT_DIR     = "./mydatasets/jieba"
MIN_FREQ       = 2          # 低频词阈值（训练集词频 < MIN_FREQ 则为低频词）

os.makedirs(OUTPUT_DIR, exist_ok=True)
jieba.add_word(NUM_TOKEN)

def tokenize(text: str) -> list[str]:
    """
    使用 jieba 精确模式分词，返回非空 token 列表。
    """
    parts = text.split(NUM_TOKEN)
    tokens = []
    for i, part in enumerate(parts):
        tokens.extend(jieba.lcut(part))
        if i < len(parts) - 1:
            tokens.append(NUM_TOKEN)

    tokens = [t for t in tokens if t.strip()] # 过滤掉纯空格 token
    return tokens


def build_vocab(train_tokens: list[list[str]], min_freq: int = MIN_FREQ):
    """
    基于训练集构建词表
    """
    counter = Counter(tok for tokens in train_tokens for tok in tokens)

    # 过滤低频词
    vocab = [w for w, cnt in counter.items() if cnt >= min_freq and w not in {PAD_TOKEN, UNK_TOKEN, NUM_TOKEN}]
    vocab = sorted(vocab)

    # 特殊 token 放最前
    special = [PAD_TOKEN, UNK_TOKEN, NUM_TOKEN]
    vocab = special + vocab

    word2idx = {w: i for i, w in enumerate(vocab)}

    print(f"\n[词表] 低频词阈值: freq < {min_freq}")
    print(f"  总词数（含低频）: {len(counter)}")
    print(f"  词表大小（过滤后）: {len(vocab)}（含 {len(special)} 个特殊 token）")

    return word2idx, counter


def encode(tokens: list[str], word2idx: dict,
           max_len: int) -> list[int]:
    """
    将 token 列表编码为 idx 序列
    """
    unk_idx = word2idx[UNK_TOKEN]
    pad_idx = word2idx[PAD_TOKEN]
    ids = [word2idx.get(t, unk_idx) for t in tokens]

    # 截断
    if len(ids) > max_len:
        ids = ids[:max_len]
    # 填充
    ids += [pad_idx] * (max_len - len(ids))

    return ids

def main():
    clean_splits = {}
    random.seed(DATASET_SEED)
    path = os.path.join(CLEAN_DATA_DIR, f"train.jsonl")
    if os.path.exists(path):
        data = load_jsonl(path)
        random.shuffle(data)
        n = len(data)
        split_idx = int(n * DATASET_SPLIT)
        clean_splits["train"] = data[:split_idx]
        clean_splits["dev"] = data[split_idx:]
    else:
        print(f"error: {path} does not exist.")

    path = os.path.join(CLEAN_DATA_DIR, f"dev.jsonl")
    if os.path.exists(path):
        clean_splits["test"] = load_jsonl(path)
    else:
        print(f"error: {path} does not exist.")

    print("\n[jieba 分词] 开始…")
    jieba.setLogLevel("WARN")  # 关闭 jieba 初始化日志
    for split, records in clean_splits.items():
        for r in records:
            r["tokens"] = tokenize(r["cleaned"])

    # 打印一个样例
    if clean_splits["train"]:
        eg = clean_splits["train"][0]
        print(f"\n  样例(train[0]):")
        print(f"    原文  : {eg['sentence']}")
        print(f"    清洗后: {eg['cleaned']}")
        print(f"    分词  : {eg['tokens']}")

    train_tokens = [r["tokens"] for r in clean_splits.get("train", [])]
    word2idx, freq_counter = build_vocab(train_tokens)

    # 保存各 split 的处理结果
    for split, records in clean_splits.items():
        out_path = os.path.join(OUTPUT_DIR, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                out = {
                    "label":      r["label"],
                    "label_desc": r["label_desc"],
                    "sentence":   r["sentence"],
                    "cleaned":    r["cleaned"],
                    "tokens":     r["tokens"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(records)} 条 -> {out_path}")

    # 保存词表
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    vocab_info = {
        "special_tokens": {PAD_TOKEN: 0, UNK_TOKEN: 1, NUM_TOKEN: 2},
        "vocab_size": len(word2idx),
        "min_freq": MIN_FREQ,
        "word2idx": word2idx,
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, ensure_ascii=False, indent=2)
    print(f"  词表: {len(word2idx)} 词 -> {vocab_path}")

    # 保存词频统计
    freq_path = os.path.join(OUTPUT_DIR, "word_freq.json")
    with open(freq_path, "w", encoding="utf-8") as f:
        json.dump(dict(freq_counter.most_common()), f, ensure_ascii=False, indent=2)
    print(f"  词频: {freq_path}")

    print("\n" + "=" * 60)
    print("摘要统计")
    print("=" * 60)
    if "train" in clean_splits:
        all_lens = [len(r["tokens"]) for r in clean_splits["train"]]
        oov_counts = sum(
            sum(1 for t in r["tokens"] if t not in word2idx or word2idx[t] == word2idx[UNK_TOKEN])
            for r in clean_splits["dev"]
        )
        bytes_num = sum(len(r["sentence"].encode("utf-8")) for r in clean_splits["train"])
        token_num = sum(len(r["tokens"]) for r in clean_splits["train"])
        compression_ratio = bytes_num / token_num if token_num > 0 else 0
        total_tokens = sum(all_lens)
        test_tokens = sum([len(r["tokens"]) for r in clean_splits.get("test", [])])
        low_freq = sum(1 for w, c in freq_counter.items() if c < MIN_FREQ)
        print(f"  训练集句子数    : {len(all_lens)}, 测试集句子数: {len(clean_splits.get('test', []))}, "
              f"验证集句子数: {len(clean_splits.get('dev', []))}")
        print(f"  平均 token 数   : {sum(all_lens)/len(all_lens):.1f}")
        print(f"  最大 token 数   : {max(all_lens)}")
        print(f"  最小 token 数   : {min(all_lens)}")
        print(f"  词表大小        : {len(word2idx)}（含特殊 token）")
        print(f"  数据压缩率      : {compression_ratio:.2f} 字节/token")
        print(f"  低频词数        : {low_freq}（freq < {MIN_FREQ}，映射为 UNK）")
        print(f"  训练集 token 总数: {total_tokens}")
        print(f"  测试集 OOV token 数     : {oov_counts}")
        print(f"  测试集 token 总数: {test_tokens}")


if __name__ == "__main__":
    main()