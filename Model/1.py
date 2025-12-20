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
FASTA_PRIMARY      = "matched_human_sequences.fasta"
FASTA_FALLBACK     = "protein_sequences.fasta"
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

# -------- FASTA reader (robust to different header formats) --------
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

                # Prefer ENSP-containing tokens and remove version numbers
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

    if "protein1" not in df.columns or "protein2" not in df.columns:
        raise ValueError("CSV missing protein1 / protein2 columns")

    if "combined_score" not in df.columns:
        raise ValueError("CSV missing combined_score column")

    p1 = df["protein1"].astype(str).apply(lambda x: x.split(".")[-1])
    p2 = df["protein2"].astype(str).apply(lambda x: x.split(".")[-1])

    labels = (df["combined_score"] >= threshold).astype(int).tolist()
    pairs = list(zip(p1.tolist(), p2.tolist()))
    proteins = set(p1).union(set(p2))

    return pairs, labels, list(proteins)


def generate_negative_pairs(protein_set, existing_pairs, num_negatives, max_trials=10_000_000):
    proteins = list(protein_set)
    existing = set(tuple(sorted(x)) for x in existing_pairs)

    neg = set()
    trials = 0

    while len(neg) < num_negatives and trials < max_trials:
        a, b = random.sample(proteins, 2)
        key = tuple(sorted((a, b)))

        if key not in existing and key not in neg:
            neg.add(key)

        trials += 1
        if trials % 1_000_000 == 0:
            print(f"[NEG] progress: {len(neg)}/{num_negatives} (trials={trials})", flush=True)

    return [(a, b) for a, b in neg]


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

class SimpleFreqEmbed:
    def __init__(self, device="cpu"):
        self.device = device

    def embed(self, seq: str):
        if not seq:
            return torch.zeros(len(AA), dtype=torch.float32)

        seq = "".join([c for c in seq.upper() if c in AA_SET])
        if not seq:
            return torch.zeros(len(AA), dtype=torch.float32)

        cnt = Counter(seq)
        vec = torch.tensor([cnt.get(a, 0) for a in AA], dtype=torch.float32)
        vec = vec / (vec.sum() + 1e-8)

        return vec


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
        if pid in self.cache:
            return self.cache[pid]

        emb = self.embed_model.embed(self.seq_dict.get(pid, ""))
        self.cache[pid] = emb
        return emb

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        y = float(self.labels[idx])

        e1 = self._emb(a)
        e2 = self._emb(b)

        return e1, e2, torch.tensor(y, dtype=torch.float32)


class DL_PPI(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=1):
        super().__init__()
        d = input_dim * 2

        self.proj = nn.Linear(d, d)
        self.bn1  = nn.BatchNorm1d(d)

        self.attn = nn.Sequential(
            nn.Linear(d, d),
            nn.Sigmoid()
        )

        self.dropout1 = nn.Dropout(0.3)

        # Dimensionality reduction + BatchNorm + GELU
        self.fc1 = nn.Linear(d, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(0.3)

        # Output layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        # Projection + BatchNorm
        x_proj = self.bn1(self.proj(x))

        # Element-wise attention
        x_gate = self.attn(x_proj)
        x = x_proj * x_gate
        x = self.dropout1(x)

        # Dimensionality reduction
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout2(x)

        # Output logits
        return self.fc2(x)


def train_dl(model, dataloader, optimizer, criterion, epochs=EPOCHS, device=DEVICE):
    model.train()

    for ep in range(epochs):
        total_loss = 0.0

        for e1, e2, y in dataloader:
            e1, e2, y = e1.to(device), e2.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(e1, e2).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {ep + 1}, Loss: {total_loss / len(dataloader):.4f}", flush=True)


def main():
    set_seed()
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(
        f"[INFO] Loading {DATASET_CSV} and labeling using threshold {COMBINED_THRESHOLD} ...",
        flush=True
    )

    pairs_pos, labels_pos, all_proteins = load_string_csv_with_threshold(
        DATASET_CSV, COMBINED_THRESHOLD
    )

    print(
        f"[INFO] Positive samples: {sum(labels_pos)} / Total edges: {len(labels_pos)}",
        flush=True
    )

    # 2) Negative samples
    print("[INFO] Generating negative samples ...", flush=True)
    neg_pairs = generate_negative_pairs(
        set(all_proteins),
        pairs_pos,
        len(pairs_pos)
    )

    pairs_all = pairs_pos + neg_pairs
    labels_all = labels_pos + [0] * len(neg_pairs)

    print(
        f"[INFO] Total samples after merge: {len(pairs_all)} "
        f"(including {len(neg_pairs)} negatives)",
        flush=True
    )

    # 3) FASTA loading
    fasta_path = (
        Path(FASTA_PRIMARY)
        if Path(FASTA_PRIMARY).exists()
        else Path(FASTA_FALLBACK)
    )

    seq_dict = load_fasta_sequences_robust(fasta_path)
    print(
        f"[INFO] Loaded sequences: {len(seq_dict)} from {fasta_path.name}",
        flush=True
    )

    # 4) Filter pairs with available sequences (sync labels)
    filtered = [
        (p, y)
        for p, y in zip(pairs_all, labels_all)
        if (p[0] in seq_dict and p[1] in seq_dict)
    ]

    if filtered:
        pairs_all, labels_all = zip(*filtered)
        pairs_all, labels_all = list(pairs_all), list(labels_all)
    else:
        pairs_all, labels_all = [], []

    print(
        f"[INFO] Usable samples after filtering: {len(pairs_all)}",
        flush=True
    )

    if len(pairs_all) == 0:
        print(
            "No usable samples found (please check ID consistency between FASTA and CSV).",
            flush=True
        )
        sys.exit(1)

    # 5) Save processed data for downstream scripts
    df_out = pd.DataFrame({
        "protein1": [a for (a, b) in pairs_all],
        "protein2": [b for (a, b) in pairs_all],
        "label": labels_all
    })

    df_out.to_csv(OUT_PROCESSED, index=False)
    print(f"[SAVE] Filtered samples saved to {OUT_PROCESSED}", flush=True)

    # 6) Training (lightweight DL model)
    embed_model = SimpleFreqEmbed(device=DEVICE)
    dataset = PPIDataset(pairs_all, labels_all, seq_dict, embed_model)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DL_PPI(
        input_dim=20,
        hidden_dim=HIDDEN_DIM,
        output_dim=1
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    train_dl(
        model,
        dataloader,
        optimizer,
        criterion,
        epochs=EPOCHS,
        device=DEVICE
    )

    print("[SUCCESS] Training completed (script 1).", flush=True)


if __name__ == "__main__":
    main()
