import os
from pathlib import Path

import numpy as np
import pandas as pd
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

from sklearn.model_selection import StratifiedKFold

from simple_embed import SimpleFreqEmbed



RESULT_DIR = "results"

PROCESSED_CSV = os.path.join(
    RESULT_DIR,
    "rf_dl_dl_scores.csv"
)

FASTA_PRIMARY = "matched_human_sequences.fasta"

FASTA_FALLBK = "protein_sequences.fasta"

TOPK_PAIRS = 300

SEED = 42

# =====================================================
# FASTA
# =====================================================

def load_fasta_sequences_robust(fpath: Path):

    seq = {}

    if not fpath.exists():
        return seq

    with open(
        fpath,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        cur, buf = None, []

        for line in f:

            if line.startswith(">"):

                if cur and buf:
                    seq[cur] = "".join(buf)

                token = line[1:].strip()

                token = token.split()[0]

                token = token.split("|")[0]

                cands = [

                    token.split(".")[-1],

                    token.split(".")[0],

                    token

                ]

                picked = next(
                    (c for c in cands if "ENSP" in c),
                    token
                )

                picked = picked.split(".")[0]

                cur, buf = picked, []

            else:

                buf.append(line.strip())

        if cur and buf:
            seq[cur] = "".join(buf)

    return seq

# =====================================================
# MAIN
# =====================================================

def main():

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    # =====================================================
    # LOAD DATA
    # =====================================================

    if not Path(PROCESSED_CSV).exists():

        raise SystemExit(
            f"{PROCESSED_CSV} not found."
        )

    df = pd.read_csv(PROCESSED_CSV)

    print("\n========== DATA INFO ==========")

    print(f"[INFO] Samples: {len(df)}")

    fasta_path = (

        Path(FASTA_PRIMARY)

        if Path(FASTA_PRIMARY).exists()

        else Path(FASTA_FALLBK)

    )

    seq_dict = load_fasta_sequences_robust(
        fasta_path
    )

    print(
        f"[INFO] Loaded sequences: "
        f"{len(seq_dict)} from {fasta_path.name}"
    )

    # =====================================================
    # AAC FEATURE
    # =====================================================

    embed = SimpleFreqEmbed(device="cpu")

    def get_vec(pid):

        return embed.embed(
            seq_dict.get(pid, "")
        ).numpy()

    # 40-dim feature

    X = np.vstack([

        np.concatenate([

            get_vec(a),

            get_vec(b)

        ])

        for a, b in zip(

            df["protein1"],

            df["protein2"]

        )

    ])

    y = df["label"].astype(int).to_numpy()

    pairs = list(zip(
        df["protein1"],
        df["protein2"]
    ))

    print(f"[INFO] Feature shape: {X.shape}")

    # =====================================================
    # CV-5
    # =====================================================

    skf = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=SEED

    )

    # =====================================================
    # METRICS
    # =====================================================

    accs  = []
    precs = []
    recs  = []
    f1s   = []
    aucs  = []

    fold_results = []

    all_predictions = []

    # =====================================================
    # ROC
    # =====================================================

    mean_fpr = np.linspace(0, 1, 100)

    tprs = []

    plt.figure(figsize=(6,6))

    # =====================================================
    # LOOP
    # =====================================================

    for fold, (tr_idx, te_idx) in enumerate(

        skf.split(X, y),

        1

    ):

        print(f"\n========== Fold {fold} ==========")

        # =====================================================
        # SPLIT
        # =====================================================

        Xtr = X[tr_idx]
        Xte = X[te_idx]

        ytr = y[tr_idx]
        yte = y[te_idx]

        pairs_te = [pairs[i] for i in te_idx]

        # =====================================================
        # RF
        # =====================================================

        rf = RandomForestClassifier(

            n_estimators=600,

            max_depth=None,

            min_samples_split=2,

            min_samples_leaf=1,

            max_features="sqrt",

            bootstrap=True,

            class_weight="balanced_subsample",

            random_state=SEED + fold,

            n_jobs=-1

        )

        # =====================================================
        # TRAIN
        # =====================================================

        rf.fit(Xtr, ytr)

        # =====================================================
        # PREDICT
        # =====================================================

        y_pred = rf.predict(Xte)

        y_proba = rf.predict_proba(Xte)[:,1]

        # =====================================================
        # METRICS
        # =====================================================

        acc = accuracy_score(yte, y_pred)

        prec = precision_score(
            yte,
            y_pred,
            zero_division=0
        )

        rec = recall_score(
            yte,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            yte,
            y_pred,
            zero_division=0
        )

        fpr, tpr, _ = roc_curve(
            yte,
            y_proba
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        # =====================================================
        # SAVE METRICS
        # =====================================================

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        aucs.append(roc_auc)

        fold_results.append({

            "Fold": fold,

            "Accuracy": acc,

            "Precision": prec,

            "Recall": rec,

            "F1": f1,

            "AUC": roc_auc

        })

        # =====================================================
        # SAVE PREDICTIONS
        # =====================================================

        pred_df = pd.DataFrame({

            "Fold": fold,

            "protein1": [p[0] for p in pairs_te],

            "protein2": [p[1] for p in pairs_te],

            "y_true": yte,

            "y_score": y_proba

        })

        all_predictions.append(pred_df)

        # =====================================================
        # ROC CURVE
        # =====================================================

        interp_tpr = np.interp(
            mean_fpr,
            fpr,
            tpr
        )

        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)

        plt.plot(

            fpr,

            tpr,

            lw=1,

            alpha=0.5,

            label=f"Fold {fold} AUC={roc_auc:.4f}"

        )

        # =====================================================
        # PRINT
        # =====================================================

        print(f"ACC : {acc:.4f}")

        print(f"PREC: {prec:.4f}")

        print(f"REC : {rec:.4f}")

        print(f"F1  : {f1:.4f}")

        print(f"AUC : {roc_auc:.4f}")

    # =====================================================
    # MEAN ROC
    # =====================================================

    mean_tpr = np.mean(
        tprs,
        axis=0
    )

    mean_tpr[-1] = 1.0

    mean_auc = auc(
        mean_fpr,
        mean_tpr
    )

    std_auc = np.std(aucs)

    plt.plot(

        mean_fpr,

        mean_tpr,

        lw=2,

        label=f"Mean ROC (AUC={mean_auc:.4f} ± {std_auc:.4f})"

    )

    plt.plot([0,1],[0,1],"k--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("RF-DL CV5 ROC Curve")

    plt.legend(loc="lower right")

    plt.tight_layout()

    roc_path = os.path.join(
        RESULT_DIR,
        "rf_dl_cv5_roc.png"
    )

    plt.savefig(
        roc_path,
        dpi=300
    )

    plt.close()

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print("\n========== FINAL CV RESULTS ==========")

    print(
        f"ACC : "
        f"{np.mean(accs):.4f} ± {np.std(accs):.4f}"
    )

    print(
        f"PREC: "
        f"{np.mean(precs):.4f} ± {np.std(precs):.4f}"
    )

    print(
        f"REC : "
        f"{np.mean(recs):.4f} ± {np.std(recs):.4f}"
    )

    print(
        f"F1  : "
        f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
    )

    print(
        f"AUC : "
        f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
    )

    # =====================================================
    # SAVE CV RESULTS
    # =====================================================

    cv_df = pd.DataFrame(fold_results)

    cv_df.loc["mean"] = [

        "mean",

        np.mean(accs),

        np.mean(precs),

        np.mean(recs),

        np.mean(f1s),

        np.mean(aucs)

    ]

    cv_df.loc["std"] = [

        "std",

        np.std(accs),

        np.std(precs),

        np.std(recs),

        np.std(f1s),

        np.std(aucs)

    ]

    cv_path = os.path.join(
        RESULT_DIR,
        "rf_dl_cv5_results.csv"
    )

    cv_df.to_csv(
        cv_path,
        index=False
    )

    print(f"[SAVE] CV results -> {cv_path}")

    # =====================================================
    # SAVE ALL PREDICTIONS
    # =====================================================

    pred_all_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    pred_path = os.path.join(
        RESULT_DIR,
        "rf_dl_predictions_all_folds.csv"
    )

    pred_all_df.to_csv(
        pred_path,
        index=False
    )

    print(f"[SAVE] Predictions -> {pred_path}")

    # =====================================================
    # TOP-K
    # =====================================================

    pred_all_df = pred_all_df.sort_values(
        "y_score",
        ascending=False
    )

    top_df = pred_all_df.head(TOPK_PAIRS)

    top_proteins = sorted(set(

        list(top_df["protein1"]) +

        list(top_df["protein2"])

    ))

    out_txt = os.path.join(
        RESULT_DIR,
        "top_proteins_rf_dl.txt"
    )

    with open(
        out_txt,
        "w",
        encoding="utf-8"
    ) as f:

        for pid in top_proteins:

            f.write(str(pid) + "\n")

    print(
        f"[SAVE] Top-K proteins -> "
        f"{out_txt} ({len(top_proteins)} IDs)"
    )

    print(f"[SAVE] ROC curve -> {roc_path}")



if __name__ == "__main__":

    main()