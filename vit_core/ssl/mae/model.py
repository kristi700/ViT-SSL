import torch
from torch import nn
from typing import Tuple

from .decoder import DecoderBlock
from .masking import get_and_apply_mask
from vit_core.encoder_block import EncoderBlock


class MAEViT(nn.Module):
    """
    Masked Autoencoder /w Vision Transformer.
        - Supports assymetrical setup in respect to the encoder-decoder
    """
    def __init__(
        self,
        encoder_depth: int,
        decoder_depth: int,
        input_shape: Tuple[int, int, int],
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        patch_size: int,
        encoder_num_heads: int,
        decoder_num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        mask_ratio: float = 0.75,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.decoder_embed_dim = decoder_embed_dim
        self.num_patches = (input_shape[1] // patch_size) ** 2

        # ENCODER
        self.unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(encoder_embed_dim, encoder_num_heads, mlp_dim, dropout)
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_projection = nn.Linear(
            (input_shape[0] * patch_size * patch_size), encoder_embed_dim
        )
        self.encoder_positional_embedding = nn.Parameter(
            torch.rand(1, self.num_patches, encoder_embed_dim)
        )

        # DECODER
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_dim, dropout)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_projection = nn.Linear(
            encoder_embed_dim, decoder_embed_dim
        )
        self.mask_token = nn.Parameter(torch.rand(1, 1, decoder_embed_dim))
        self.decoder_positional_embedding = nn.Parameter(
            torch.rand(1, self.num_patches, decoder_embed_dim)
        )
        self.prediction_head = nn.Linear(decoder_embed_dim, input_shape[0] * patch_size * patch_size)

    def encoder_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The MAE's encoder forward
            - Applies masking and forwards all non-masked patches
        Returns:
            - visible_patches: Processed patches
            - masked_indicies: Indicies of the masked patches for loss calculation
            - resolve_ids: the ids for deshuffling patches
        """
        patches =  torch.permute(self.unfold(x), (0, 2, 1)).to(x.device)
        patches = self.encoder_projection(patches)
        patches += self.encoder_positional_embedding
        visible_patches, masked_indicies, resolve_ids = get_and_apply_mask(patches, self.mask_ratio)
        
        for encoder_block in self.encoder_blocks:
            visible_patches, _ = encoder_block(visible_patches)

        return visible_patches, masked_indicies, resolve_ids

    def decoder_forward(self, x: torch.Tensor, resolve_ids: torch.Tensor) -> torch.Tensor:
        """
        MAE's decoder forward
            - Reconstructs the original image with the help of the encoders outputs.
        Returns:
            - reconstructed_image: the whole reconstructed image - (batch * num_patches * (channel *patch_size**2))
        """
        x = self.decoder_projection(x)
        
        num_masked = self.num_patches - x.shape[1]
        masked_tokens = self.mask_token.expand(x.shape[0], num_masked , -1)
        full_patches = torch.cat([x, masked_tokens], dim=1)

        resolve_ids_expanded = resolve_ids.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim)
        full_patches = torch.gather(full_patches, dim=1, index=resolve_ids_expanded)

        for decoder_block in self.decoder_blocks:
            full_patches, _ = decoder_block(full_patches)

        reconstructed_image = self.prediction_head(full_patches)
        return reconstructed_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward of the MAE class
        """
        x, masked_indicies, resolve_ids = self.encoder_forward(x)
        x = self.decoder_forward(x, resolve_ids)
        return x, masked_indicies

    # TODO - add CLS token based approach as well (https://arxiv.org/pdf/2111.06377 - A. Implementation Details, ViT architecture)
    @torch.no_grad()
    def inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference forward of MAE, using only the encoder on full image inputs
        """
        patches =  torch.permute(self.unfold(x), (0, 2, 1)).to(x.device)
        patches = self.encoder_projection(patches)
        patches += self.encoder_positional_embedding
        for encoder_block in self.encoder_blocks:
            patches, _ = encoder_block(patches)
        return patches.mean(dim=1)