# -*- coding: utf-8 -*-
"""
1_preprocess.py
- 读取 H-1.csv + FASTA，阈值=700
- 生成负样本，过滤到均有序列的样本
- 保存 xgb_processed_pairs.csv 供后续 RF/富集脚本使用
"""

import os
import sys
import random
from pathlib import Path
from collections import Counter
import pandas as pd

# ================= 可配 =================
DATASET_CSV        = "H-1.csv"
FASTA_PRIMARY      = "matched_human_sequences.fasta"
FASTA_FALLBACK     = "protein_sequences.fasta"
COMBINED_THRESHOLD = 700
RESULT_DIR         = "results"
OUT_PROCESSED      = os.path.join(RESULT_DIR, "xgb_processed_pairs.csv")
SEED               = 42
# ======================================

def set_seed(seed=SEED):
    random.seed(seed)

# -------- FASTA 读取（尽量兼容 header 形式） --------
def load_fasta_sequences_robust(fasta_path: Path):
    seq_dict = {}
    if not fasta_path.exists():
        return seq_dict
    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        curr_id, chunks = None, []
        for line in f:
            if line.startswith(">"):
                if curr_id and chunks:
                    seq_dict[curr_id] = "".join(chunks)
                header = line[1:].strip()
                token = header.split()[0].split("|")[0]
                candidates = [token.split(".")[-1], token.split(".")[0], token]
                picked = next((c for c in candidates if "ENSP" in c), token)
                picked = picked.split(".")[0]
                curr_id, chunks = picked, []
            else:
                chunks.append(line.strip())
        if curr_id and chunks:
            seq_dict[curr_id] = "".join(chunks)
    return seq_dict

# -------- 读取 CSV 并打标签 --------
def load_string_csv_with_threshold(path, threshold):
    df = pd.read_csv(path)
    if "protein1" not in df.columns or "protein2" not in df.columns:
        raise ValueError("CSV 缺少 protein1 / protein2 列")
    if "combined_score" not in df.columns:
        raise ValueError("CSV 缺少 combined_score 列")
    p1 = df["protein1"].astype(str).apply(lambda x: x.split(".")[-1])
    p2 = df["protein2"].astype(str).apply(lambda x: x.split(".")[-1])
    labels = (df["combined_score"] >= threshold).astype(int).tolist()
    pairs = list(zip(p1.tolist(), p2.tolist()))
    proteins = set(p1).union(set(p2))
    return pairs, labels, list(proteins)

# -------- 负样本生成 --------
def generate_negative_pairs(protein_set, existing_pairs, num_negatives):
    proteins = list(protein_set)
    existing = set(tuple(sorted(x)) for x in existing_pairs)
    neg = set()
    trials = 0
    while len(neg) < num_negatives:
        a, b = random.sample(proteins, 2)
        key = tuple(sorted((a, b)))
        if key not in existing and key not in neg:
            neg.add(key)
        trials += 1
        if trials % 100000 == 0:
            print(f"[NEG] progress: {len(neg)}/{num_negatives}")
    return [(a, b) for a, b in neg]

def main():
    set_seed()
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1) CSV + 标签
    print(f"[INFO] 读取 {DATASET_CSV} 并按阈值 {COMBINED_THRESHOLD} 打标签 ...")
    pairs_pos, labels_pos, all_proteins = load_string_csv_with_threshold(DATASET_CSV, COMBINED_THRESHOLD)
    print(f"[INFO] 正样本: {sum(labels_pos)} / 总边数: {len(labels_pos)}")

    # 2) 负样本
    print("[INFO] 生成负样本 ...")
    neg_pairs = generate_negative_pairs(set(all_proteins), pairs_pos, len(pairs_pos))
    pairs_all = pairs_pos + neg_pairs
    labels_all = labels_pos + [0] * len(neg_pairs)
    print(f"[INFO] 合并后样本: {len(pairs_all)} (含负样本 {len(neg_pairs)})")

    # 3) FASTA
    fasta_path = Path(FASTA_PRIMARY) if Path(FASTA_PRIMARY).exists() else Path(FASTA_FALLBACK)
    seq_dict = load_fasta_sequences_robust(fasta_path)
    print(f"[INFO] 已加载序列: {len(seq_dict)} from {fasta_path.name}")

    # 4) 过滤
    flt = [(p, y) for p, y in zip(pairs_all, labels_all) if (p[0] in seq_dict and p[1] in seq_dict)]
    if flt:
        pairs_all, labels_all = zip(*flt)
        pairs_all, labels_all = list(pairs_all), list(labels_all)
    else:
        pairs_all, labels_all = [], []
    print(f"[INFO] 过滤后可用样本: {len(pairs_all)}")
    if len(pairs_all) == 0:
        print("❌ 无可用样本（请检查 FASTA 与 CSV 的 ID 是否一致）")
        sys.exit(1)

    # 5) 保存
    df_out = pd.DataFrame({
        "protein1": [a for (a, b) in pairs_all],
        "protein2": [b for (a, b) in pairs_all],
        "label": labels_all
    })
    df_out.to_csv(OUT_PROCESSED, index=False)
    print(f"[SAVE] 已保存过滤后的样本到 {OUT_PROCESSED}")

if __name__ == "__main__":
    main()
