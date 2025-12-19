# -*- coding: utf-8 -*-
"""
2_catboost_eval.py
- 从 xgb_processed_pairs.csv 生成 AAC 频率特征
- 训练 CatBoost 模型
- 保存指标、分类报告、ROC曲线
- 保存 Top-K 蛋白列表
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

# ================= 可配 =================
RESULT_DIR    = "results"
PROCESSED_CSV = os.path.join(RESULT_DIR, "catboost_processed_pairs.csv")
FASTA_PRIMARY = "matched_human_sequences.fasta"
FASTA_FALLBK  = "protein_sequences.fasta"
TOPK_PAIRS    = 300
SEED          = 42
# ======================================

# -------- FASTA 读取 --------
def load_fasta_sequences_robust(fpath: Path):
    seq = {}
    if not fpath.exists(): return seq
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        cur, buf = None, []
        for line in f:
            if line.startswith(">"):
                if cur and buf: seq[cur] = "".join(buf)
                token = line[1:].strip().split()[0].split("|")[0]
                cands = [token.split(".")[-1], token.split(".")[0], token]
                picked = next((c for c in cands if "ENSP" in c), token)
                picked = picked.split(".")[0]
                cur, buf = picked, []
            else:
                buf.append(line.strip())
        if cur and buf:
            seq[cur] = "".join(buf)
    return seq

# -------- AAC 特征 --------
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

def aac_vector(seq: str) -> np.ndarray:
    seq = seq.upper()
    seq = ''.join([c for c in seq if c in AA_SET])
    if not seq:
        return np.zeros(len(AA), dtype=np.float32)
    from collections import Counter
    cnt = Counter(seq)
    vec = np.array([cnt.get(a, 0) for a in AA], dtype=np.float32)
    return vec / (vec.sum() + 1e-8)

# -------- 主函数 --------
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not Path(PROCESSED_CSV).exists():
        raise SystemExit(f"没有找到 {PROCESSED_CSV}，请先运行 1_preprocess.py。")

    # 1) 读数据
    df = pd.read_csv(PROCESSED_CSV)
    print(f"[INFO] 读取样本: {len(df)}")

    # 2) FASTA
    fasta_path = Path(FASTA_PRIMARY) if Path(FASTA_PRIMARY).exists() else Path(FASTA_FALLBK)
    seq_dict = load_fasta_sequences_robust(fasta_path)
    print(f"[INFO] 已加载序列: {len(seq_dict)} from {fasta_path.name}")

    # 3) 特征
    def get_vec(pid):
        return aac_vector(seq_dict.get(pid, ""))
    X = np.vstack([np.concatenate([get_vec(a), get_vec(b)]) for a,b in zip(df["protein1"], df["protein2"])])
    y = df["label"].astype(int).to_numpy()

    # 4) 划分
    Xtr, Xte, ytr, yte, pairs_tr, pairs_te = train_test_split(
        X, y, list(zip(df["protein1"], df["protein2"])),
        test_size=0.2, random_state=SEED, stratify=y
    )

    # 5) CatBoost 训练
    cat = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        random_seed=SEED,
        verbose=100,
        loss_function="Logloss",
        eval_metric="AUC"
    )
    cat.fit(Xtr, ytr)

    # 6) 预测
    y_pred  = cat.predict(Xte)
    y_proba = cat.predict_proba(Xte)[:, 1]

    # 7) 主要指标
    acc  = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec  = recall_score(yte, y_pred, zero_division=0)
    f1   = f1_score(yte, y_pred, zero_division=0)

    print(f"[METRICS] Accuracy : {acc:.4f}")
    print(f"[METRICS] Precision: {prec:.4f}")
    print(f"[METRICS] Recall   : {rec:.4f}")
    print(f"[METRICS] F1-score : {f1:.4f}")

    metrics_df = pd.DataFrame([{
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }])
    metrics_df.to_csv(os.path.join(RESULT_DIR, "catboost_main_metrics.csv"), index=False)

    report = classification_report(yte, y_pred, digits=4, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(RESULT_DIR, "catboost_classification_report.csv"), index=True)

    # 8) ROC / AUC
    fpr, tpr, _ = roc_curve(yte, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CatBoost ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "catboost_auc_roc.png"), dpi=300)
    plt.close()

    # === 5.5 保存测试集预测结果（用于总 ROC 图） ===
    pred_df = pd.DataFrame({
        "protein1": [p[0] for p in pairs_te],
        "protein2": [p[1] for p in pairs_te],
        "y_true": yte,
        "y_score": y_proba
    })

    save_path = os.path.join(RESULT_DIR, "catboost_processed_pairs.csv")
    pred_df.to_csv(save_path, index=False)
    print(f"[SAVE] catboost test predictions saved to: {save_path}")

    # 9) Top-K 蛋白集合
    order = np.argsort(-y_proba)[:TOPK_PAIRS]
    top_pairs = [pairs_te[i] for i in order]
    top_proteins = sorted(set([p for ab in top_pairs for p in ab]))
    out_txt = os.path.join(RESULT_DIR, "top_proteins_catboost.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for pid in top_proteins:
            f.write(str(pid) + "\n")
    print(f"[SAVE] Top-K 概率蛋白清单 -> {out_txt}（{len(top_proteins)} 个ID）")

if __name__ == "__main__":
    main()
