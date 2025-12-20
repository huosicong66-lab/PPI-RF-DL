import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split

from simple_embed import SimpleFreqEmbed  # 20-dim frequency embedding

# ------------------ Configurable ------------------
RESULT_DIR    = "results"
PROCESSED_CSV = os.path.join(RESULT_DIR, "rf_dl_processed_pairs.csv")
FASTA_PRIMARY = "matched_human_sequences.fasta" # Primary FASTA file containing protein sequences (must exist)
FASTA_FALLBK  = "protein_sequences.fasta"  # Secondary FASTA containing protein sequences, used as a fallback option
TOPK_PAIRS    = 300       # Select Top-K protein pairs by RF probability (test set)
SEED          = 42
# -------------------------------------------------


def load_fasta_sequences_robust(fpath: Path):
    seq = {}
    if not fpath.exists():
        return seq

    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        cur, buf = None, []
        for line in f:
            if line.startswith(">"):
                if cur and buf:
                    seq[cur] = "".join(buf)

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


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    if not Path(PROCESSED_CSV).exists():
        raise SystemExit(
            f"{PROCESSED_CSV} not found. Please run the preprocessing script first."
        )

    # 1) Load data
    df = pd.read_csv(PROCESSED_CSV)
    print(f"[INFO] Loaded samples: {len(df)}")

    fasta_path = (
        Path(FASTA_PRIMARY)
        if Path(FASTA_PRIMARY).exists()
        else Path(FASTA_FALLBK)
    )
    seq_dict = load_fasta_sequences_robust(fasta_path)
    print(f"[INFO] Loaded sequences: {len(seq_dict)} from {fasta_path.name}")

    # 2) Generate features
    #    (concatenate two 20-dim frequency vectors → 40-dim feature)
    embed = SimpleFreqEmbed(device="cpu")

    def get_vec(pid):
        return embed.embed(seq_dict.get(pid, "")).numpy()

    X = np.vstack([
        np.concatenate([get_vec(a), get_vec(b)])
        for a, b in zip(df["protein1"], df["protein2"])
    ])
    y = df["label"].astype(int).to_numpy()

    # 3) Train / test split
    Xtr, Xte, ytr, yte, pairs_tr, pairs_te = train_test_split(
        X,
        y,
        list(zip(df["protein1"], df["protein2"])),
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    # 4) Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(Xtr, ytr)

    # 5) Prediction + probabilities
    y_pred  = rf.predict(Xte)
    y_proba = rf.predict_proba(Xte)[:, 1]

    # 6) Core metrics
    acc  = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec  = recall_score(yte, y_pred, zero_division=0)
    f1   = f1_score(yte, y_pred, zero_division=0)

    print(f"[METRICS] Accuracy : {acc:.4f}")
    print(f"[METRICS] Precision: {prec:.4f}")
    print(f"[METRICS] Recall   : {rec:.4f}")
    print(f"[METRICS] F1-score : {f1:.4f}")

    # Save main metrics
    metrics_df = pd.DataFrame([{
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }])
    metrics_path = os.path.join(RESULT_DIR, "rf_dl_main_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVE] Main metrics -> {metrics_path}")

    # Save classification report
    report = classification_report(
        yte,
        y_pred,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    rep_df = pd.DataFrame(report).transpose()
    rep_path = os.path.join(RESULT_DIR, "rf_classification_report.csv")
    rep_df.to_csv(rep_path, index=True)
    print(f"[SAVE] Classification report -> {rep_path}")

    # 7) ROC / AUC
    fpr, tpr, _ = roc_curve(yte, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RF-DL ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = os.path.join(RESULT_DIR, "rf_dl_auc_roc.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"[SAVE] ROC curve -> {roc_path} (AUC={roc_auc:.4f})")

    # 8) Save test-set predictions
    pred_df = pd.DataFrame({
        "protein1": [p[0] for p in pairs_te],
        "protein2": [p[1] for p in pairs_te],
        "y_true": yte,
        "y_score": y_proba
    })

    save_path = os.path.join(RESULT_DIR, "rf_dl_processed_pairs.csv")
    pred_df.to_csv(save_path, index=False)
    print(f"[SAVE] RF-DL test predictions saved to: {save_path}")

    # 9) Top-K protein set
    order = np.argsort(-y_proba)[:TOPK_PAIRS]
    top_pairs = [pairs_te[i] for i in order]
    top_proteins = sorted(set(p for ab in top_pairs for p in ab))

    out_txt = os.path.join(RESULT_DIR, "top_proteins_rf_dl.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for pid in top_proteins:
            f.write(str(pid) + "\n")

    print(f"[SAVE] Top-K proteins -> {out_txt} ({len(top_proteins)} IDs)")


if __name__ == "__main__":
    main()
