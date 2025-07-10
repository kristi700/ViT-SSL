import torch
from torch import nn

from .decoder import DecoderBlock
from .masking import get_and_apply_mask
from vit_core.encoder_block import EncoderBlock

# TODO - init needs to be separate for encoder and decoder as we are creating non symmetrical design

class MAEViT(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_shape,
        embed_dim: int,
        patch_size: int,
        num_heads: int = 8,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        mask_ratio: float = 0.6,
    ):
        super().__init__()
        # ENCODER
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.encoder_projection = nn.Linear(
            (input_shape[0] * patch_size * patch_size), embed_dim
        )
        self.encoder_positional_embedding = nn.Parameter(
            torch.rand(1, (input_shape[1] // patch_size) ** 2, embed_dim)
        )

        # DECODER
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)
        self.mask_ratio = mask_ratio

    def encoder_forward(self, x):
        patches =  torch.permute(self.unfold(x), (0, 2, 1)).to(x.device)
        patches = self.encoder_projection(patches)
        patches += self.encoder_positional_embedding
        visible_patches, masked_indicies, resolve_ids = get_and_apply_mask(patches, self.mask_ratio)
        
        for encoder_block in self.encoder_blocks:
            visible_patches, _ = encoder_block(visible_patches)

        return visible_patches, masked_indicies, resolve_ids

    def decoder_forward(self, x, masked_indicies, resolve_ids):
        ...

    def forward(self, x: torch.Tensor, return_bool_mask=False) -> torch.Tensor:
        x, masked_indicies, resolve_ids = self.encoder_forward(x)
        x = self.decoder_forward(x, masked_indicies, resolve_ids)
        return x, masked_indicies


    @torch.no_grad()
    def inference_forward(self, x: torch.Tensor, return_patch_features=False):
        ...