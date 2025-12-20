import torch
from torch.utils.data import Dataset

class PPIDataset(Dataset):
    def __init__(self, pairs, labels, seq_dict, embed_model):
        self.pairs = pairs
        self.labels = labels
        self.seq_dict = seq_dict
        self.embed_model = embed_model
        self.embedding_cache = {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        label = self.labels[idx]

        try:
            if p1 not in self.embedding_cache:
                if p1 in self.seq_dict:
                    self.embedding_cache[p1] = self.embed_model.embed(self.seq_dict[p1])
                else:
                    print(f"[WARNING] 蛋白 {p1} 不在序列字典中，返回零向量")
                    self.embedding_cache[p1] = torch.zeros(1024)

            if p2 not in self.embedding_cache:
                if p2 in self.seq_dict:
                    self.embedding_cache[p2] = self.embed_model.embed(self.seq_dict[p2])
                else:
                    print(f"[WARNING] 蛋白 {p2} 不在序列字典中，返回零向量")
                    self.embedding_cache[p2] = torch.zeros(1024)

        except Exception as e:
            print(f"[ERROR] 获取嵌入失败，{p1}, {p2}，原因：{e}")
            return torch.zeros(1024), torch.zeros(1024), torch.tensor(0.0)

        return self.embedding_cache[p1], self.embedding_cache[p2], torch.tensor(label, dtype=torch.float)

