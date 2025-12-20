import torch
import torch.nn as nn

class FRN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FRN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, e1, e2):
        x = torch.cat([e1, e2], dim=-1)
        return self.fc(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_PPI(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=1):
        super(DL_PPI, self).__init__()
        concat_dim = input_dim * 2

        self.proj = nn.Linear(concat_dim, concat_dim)
        self.norm1 = nn.LayerNorm(concat_dim)
        self.dropout1 = nn.Dropout(0.3)

        self.attn = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.norm1(self.proj(x))
        x = x * self.attn(x)
        x = self.dropout1(x)

        x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        return x

