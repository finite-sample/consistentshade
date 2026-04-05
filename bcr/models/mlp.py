"""MLP model definitions."""

import torch.nn as nn


class DropMLP(nn.Module):
    """Simple MLP with dropout for regression or classification."""

    def __init__(self, d_in: int, hid: int = 64, p: float = 0.2, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hid),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
