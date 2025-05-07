import torch
from torch import nn

from vit_core.mlp_head import MLPHead
from vit_core.encoder_block import EncoderBlock
from vit_core.patch_embedding import ConvolutionalPatchEmbedding


class ViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_blocks: int,
        input_shape,
        embed_dim: int,
        patch_size: int,
        num_heads: int = 8,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
                for i in range(num_blocks)
            ]
        )
        self.patch_embedding = ConvolutionalPatchEmbedding(
            input_shape, embed_dim, patch_size
        )
        self.classification_head = MLPHead(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, return_attn=False) -> torch.Tensor:
        x = self.patch_embedding(x)
        for encoder_block in self.encoder_blocks:
            x, attn_probs = encoder_block(x, return_attn) # NOTE - here we always get the last one, might need some nicer implementation later
        cls_token_output = x[:, 0]
        logits = self.classification_head(cls_token_output)

        if return_attn:
            return logits, attn_probs
        else:
            return logits
