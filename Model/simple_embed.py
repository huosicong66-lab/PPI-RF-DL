import torch
from collections import Counter

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

class SimpleFreqEmbed:
    def __init__(self, device='cpu'):
        self.device = device

    def embed(self, seq: str):
        if not seq:
            return torch.zeros(len(AA), dtype=torch.float32)
        seq = ''.join([c for c in seq.upper() if c in AA_SET])
        if not seq:
            return torch.zeros(len(AA), dtype=torch.float32)
        cnt = Counter(seq)
        vec = torch.tensor([cnt.get(a, 0) for a in AA], dtype=torch.float32)
        vec = vec / (vec.sum() + 1e-8)
        return vec
