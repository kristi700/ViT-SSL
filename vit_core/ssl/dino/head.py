import torch.nn.functional as F

from torch import nn
from torch.nn.utils import weight_norm


class DINOHead(nn.Module):
    def __init__(self, embed_dim, output_dim, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.fully_connected = weight_norm(nn.Linear(embed_dim, output_dim), name="weight")

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=1)
        x = self.fully_connected(x)
        return x
