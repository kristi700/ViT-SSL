import torch
from torch import nn

from vit_core.feed_forward import FeedForwardBlock
from vit_core.attention import MultiHeadedAttention

class DecoderBlock(nn.Module):
    """
    Implements a Transformer Block for the MAE Decoder.
    NOTE: This is structurally identical to the EncoderBlock.
    It uses self-attention, not cross-attention.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = MultiHeadedAttention(d_model, num_heads)
        self.feed_forward = FeedForwardBlock(d_model, mlp_dim, dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn=False) -> torch.Tensor:
        """
        Forward pass using Pre-LN.
        Args:
            x: Input tensor (batch_size, seq_len, d_model).
               For the decoder, seq_len is the *full* number of patches.
        Output:
            Tensor of the same shape as input.
        """
        residual = x
        x = self.layer_norm1(x)
        x, attn_probs = self.self_attention(
            query=x, key=x, value=x, return_attn=return_attn
        )
        x = self.drop1(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.drop2(x)
        x = x + residual
        return x, attn_probs