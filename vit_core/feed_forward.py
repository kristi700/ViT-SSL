import torch

from torch import nn
import torch.nn.functional as F

class FeedForwardBlock(nn.Module):
    """
    Simple FeedForwardBlock for vanilla transformer.
    """
    def __init__(self, d_model: int = 512, d_ff: int= 2048, dropout: float = 0.1):
        super().__init__()
        self.linear_in = nn.Linear(d_model, d_ff)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FeedForwardBlock forward
        Args:
            - x: Context vector
        Output:
            - x: Linearly transformed context vector
        """
        x = F.gelu(self.linear_in(x))
        x = self.dropout(x)
        return self.linear_out(x)
