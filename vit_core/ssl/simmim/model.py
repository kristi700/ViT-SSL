import torch
from torch import nn

from .masking import simple_masking
from vit_core.encoder_block import EncoderBlock


class ViT(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_shape,
        embed_dim: int,
        patch_size: int,
        num_heads: int = 8,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        mask_ratio: float =  0.6,
    ):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
                for i in range(num_blocks)
            ]
        )
        self.unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)

        self.mask_ratio = mask_ratio
        self.input_shape = input_shape

    def forward(self, x: torch.Tensor, return_attn=False) -> torch.Tensor:
        patches =  torch.permute(self.unfold(x), (0, 2, 1))
        patches, bool_mask, targets = simple_masking(patches, self.mask_ratio)

        for encoder_block in self.encoder_blocks:
            x, attn_probs = encoder_block(x, return_attn) # NOTE - here we always get the last one, might need some nicer implementation later
        cls_token_output = x[:, 0]

        """
        if return_attn:
            return logits, attn_probs
        else:
            return logits
        """
