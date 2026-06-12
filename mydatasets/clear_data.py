import json
import os
import re
from collections import Counter
import unicodedata
from utils import load_jsonl
from config import NUM_TOKEN

DATA_DIR       = "./mydatasets/tnews_public"
IMAGE_DIR      = "./image/data_analysis"
OUTPUT_DIR     = "./mydatasets/tnews_cleared"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_LABELS = {
    "100", "101", "102", "103", "104",
    "106", "107", "108", "109", "110",
    "112", "113", "114", "115", "116"
}


def filter_and_deduplicate(records, split="train"):
    """
    去除：
      1. sentence 纯空白或空的样本
      2. 完全重复的 (sentence, label) 对
      3. label 不在合法集合中的样本
    """
    total_raw = len(records)
    removed_empty   = 0
    removed_illegal = 0
    removed_dup     = 0

    seen = set()
    cleaned = []
    for r in records:
        sentence = r.get("sentence", "")
        label    = str(r.get("label", "")).strip()

        # 空文本
        if not sentence or not sentence.strip():
            removed_empty += 1
            continue

        # 非法标签
        if label not in VALID_LABELS:
            removed_illegal += 1
            continue

        # 重复样本（以原始文本+标签为 key）
        key = (sentence.strip(), label)
        if key in seen:
            removed_dup += 1
            continue
        seen.add(key)

        cleaned.append({"sentence": sentence.strip(), "label": label,
                         "label_desc": r.get("label_desc", "")})

    print(f"\n[{split}] 原始样本: {total_raw}")
    print(f"  去除空文本: {removed_empty}")
    print(f"  去除非法标签: {removed_illegal}")
    print(f"  去除重复样本: {removed_dup}")
    print(f"  有效样本: {len(cleaned)}")

    # 各类别统计
    import matplotlib.pyplot as plt
    label_counter = Counter(r["label"] for r in cleaned)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_counter.keys(), label_counter.values(), color="skyblue", edgecolor="black")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, yval, ha='center', va='bottom')
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title(f"Label Distribution in {split} Set")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(IMAGE_DIR, f"{split}_label_distribution.png"))

    return cleaned


# 全角转为半角
_FULL2HALF_TABLE = str.maketrans(
    "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９："
    "；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴ"
    "ＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎ"
    "ｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～",
    "!\"#$%&'()*+,-./"
    "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRST"
    "UVWXYZ[\\]^_`abcdefghijklmn"
    "opqrstuvwxyz{|}~"
)


def clean_text(text: str) -> str:
    """
    文本清洗
    """
    # 去除不可见字符（零宽空格、控制字符等）
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf", "Cs")
           or ch in ("\n", "\t", " ")
    )

    # 全角转半角
    text = text.translate(_FULL2HALF_TABLE)

    # 合并连续空白为单个空格
    text = re.sub(r"\s+", " ", text).strip()

    # 只保留中文字符、英文字母、数字、空格
    text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf\u4e00-\u9fa5a-zA-Z0-9 ]", "", text)
    text = re.sub(r"\d+", NUM_TOKEN, text)  # 将连续数字替换为 <NUM>

    # 压缩空格
    text = re.sub(r" +", " ", text).strip()

    return text

def main():
    splits = {}
    for split in ["train", "dev"]: # 跳过test集
        path = os.path.join(DATA_DIR, f"{split}.json")
        if os.path.exists(path):
            splits[split] = load_jsonl(path)
        else:
            print(f"error: {path} does not exist.")

    clean_splits = {}
    for split, records in splits.items():
        clean_splits[split] = filter_and_deduplicate(records, split)

    print("\n[文本清洗] 开始…")
    for split, records in clean_splits.items():
        for r in records:
            r["cleaned"] = clean_text(r["sentence"])

    for split, records in clean_splits.items():
        out_path = os.path.join(OUTPUT_DIR, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                out = {
                    "label":      r["label"],
                    "label_desc": r["label_desc"],
                    "sentence":   r["sentence"],
                    "cleaned":    r["cleaned"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(records)} 条 -> {out_path}")

if __name__ == "__main__":
    main()