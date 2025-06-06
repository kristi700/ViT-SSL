import torch

from torch import nn


class MLPHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear(x)
        return x
