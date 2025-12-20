import os
import sys
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ================= Configurable =================
DATASET_CSV        = "H-1.csv"
FASTA_PRIMARY      = "matched_human_sequences.fasta" # Primary FASTA file containing protein sequences (must exist)
FASTA_FALLBACK     = "protein_sequences.fasta"  # Secondary FASTA containing protein sequences, used as a fallback option
COMBINED_THRESHOLD = 400
BATCH_SIZE         = 512
EPOCHS             = 200
LR                 = 1e-3
HIDDEN_DIM         = 64
DEVICE             = "cpu"
SEED               = 42
RESULT_DIR         = "DL-RF/results"
OUT_PROCESSED      = os.path.join(RESULT_DIR, "rf_dl_processed_pairs.csv")
# ===============================================


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------- FASTA reader --------
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


def load_string_csv_with_threshold(path, threshold):
    df = pd.read_csv(path)

    p1 = df["protein1"].astype(str).apply(lambda x: x.split(".")[-1])
    p2 = df["protein2"].astype(str).apply(lambda x: x.split(".")[-1])

    labels = (df["combined_score"] >= threshold).astype(int).tolist()
    pairs = list(zip(p1.tolist(), p2.tolist()))
    proteins = set(p1).union(set(p2))

    return pairs, labels, list(proteins)


def generate_negative_pairs(protein_set, existing_pairs, num_negatives):
    proteins = list(protein_set)
    existing = set(tuple(sorted(x)) for x in existing_pairs)

    neg = set()
    while len(neg) < num_negatives:
        a, b = random.sample(proteins, 2)
        key = tuple(sorted((a, b)))
        if key not in existing:
            neg.add(key)

    return list(neg)


# -------- Simple Frequency Embedding --------
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

class SimpleFreqEmbed:
    def embed(self, seq: str):
        if not seq:
            return torch.zeros(len(AA))

        seq = "".join([c for c in seq.upper() if c in AA_SET])
        if not seq:
            return torch.zeros(len(AA))

        cnt = Counter(seq)
        vec = torch.tensor([cnt.get(a, 0) for a in AA], dtype=torch.float32)
        return vec / (vec.sum() + 1e-8)


# -------- Dataset --------
class PPIDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, labels, seq_dict, embed_model):
        self.pairs = pairs
        self.labels = labels
        self.seq_dict = seq_dict
        self.embed_model = embed_model
        self.cache = {}

    def __len__(self):
        return len(self.pairs)

    def _emb(self, pid):
        if pid not in self.cache:
            self.cache[pid] = self.embed_model.embed(
                self.seq_dict.get(pid, "")
            )
        return self.cache[pid]

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        y = float(self.labels[idx])
        return self._emb(a), self._emb(b), torch.tensor(y)


# -------- DL Model --------
class DL_PPI(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        d = input_dim * 2

        self.fc = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, x2):
        return self.fc(torch.cat([x1, x2], dim=1))


# -------- Training --------
def train_dl(model, loader, optimizer, criterion):
    model.train()
    for ep in range(EPOCHS):
        loss_sum = 0.0
        for e1, e2, y in loader:
            optimizer.zero_grad()
            out = model(e1, e2).squeeze()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        print(f"Epoch {ep+1}/{EPOCHS} | Loss={loss_sum/len(loader):.4f}")


# -------- Inference (y_score) --------
@torch.no_grad()
def infer_dl(model, loader):
    model.eval()
    scores = []
    for e1, e2, _ in loader:
        logits = model(e1, e2).squeeze()
        probs = torch.sigmoid(logits)
        scores.extend(probs.numpy().tolist())
    return scores


# ===================== MAIN =====================
def main():
    set_seed()
    os.makedirs(RESULT_DIR, exist_ok=True)

    pairs_pos, labels_pos, proteins = load_string_csv_with_threshold(
        DATASET_CSV, COMBINED_THRESHOLD
    )

    neg_pairs = generate_negative_pairs(
        set(proteins), pairs_pos, len(pairs_pos)
    )

    pairs_all = pairs_pos + neg_pairs
    labels_all = labels_pos + [0] * len(neg_pairs)

    fasta_path = Path(FASTA_PRIMARY) if Path(FASTA_PRIMARY).exists() else Path(FASTA_FALLBACK)
    seq_dict = load_fasta_sequences_robust(fasta_path)

    filtered = [
        (p, y) for p, y in zip(pairs_all, labels_all)
        if p[0] in seq_dict and p[1] in seq_dict
    ]

    pairs_all, labels_all = zip(*filtered)

    embed = SimpleFreqEmbed()
    dataset = PPIDataset(pairs_all, labels_all, seq_dict, embed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DL_PPI(input_dim=20, hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    train_dl(model, loader, optimizer, criterion)

    print("[INFO] Running inference to get y_score...")
    infer_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    y_scores = infer_dl(model, infer_loader)

    df_out = pd.DataFrame({
        "protein1": [a for a, b in pairs_all],
        "protein2": [b for a, b in pairs_all],
        "label": labels_all,
        "y_score": y_scores
    })

    df_out.to_csv(OUT_PROCESSED, index=False)
    print(f"[SAVE] {OUT_PROCESSED}")


if __name__ == "__main__":
    main()
