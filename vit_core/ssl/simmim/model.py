import torch
from torch import nn

from .masking import simple_masking
from vit_core.encoder_block import EncoderBlock


class SimMIMViT(nn.Module):
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
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)
        self.projection = nn.Linear(
            (input_shape[0] * patch_size * patch_size), embed_dim
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(
            torch.rand(1, (input_shape[1] // patch_size) ** 2, embed_dim)
        )
        self.simmim_head = nn.Linear(
            embed_dim, input_shape[0] * patch_size * patch_size
        )

        self.mask_ratio = mask_ratio
        self.input_shape = input_shape

    def forward(self, x: torch.Tensor, return_bool_mask=False) -> torch.Tensor:
        patches = torch.permute(self.unfold(x), (0, 2, 1)).to(x.device)
        patches, bool_mask, targets = simple_masking(patches, self.mask_ratio)
        patches = self.projection(patches)
        # NOTE - for now we wont as CLS as pretraining doesnt require it (will be included in the later head)
        bool_mask = bool_mask.unsqueeze(-1)
        encoder_input_embeddings = torch.where(bool_mask, self.mask_token, patches)
        encoder_input_embeddings += self.positional_embedding
        x = encoder_input_embeddings

        for encoder_block in self.encoder_blocks:
            x, _ = encoder_block(
                x
            )  # NOTE - here we always get the last one, might need some nicer implementation later
        encoder_outputs_for_masked_patches = x[bool_mask.squeeze(-1)]
        predicted_pixels = self.simmim_head(encoder_outputs_for_masked_patches)

        if return_bool_mask:
            return predicted_pixels, targets, bool_mask
        else:
            return predicted_pixels, targets

@torch.no_grad()
def inference_forward(self, x: torch.Tensor, return_patch_features=False):
    """
    Clean inference forward pass for feature extraction.
    
    Args:
        x: Input image tensor [B, C, H, W]
        return_patch_features: If True, return all patch features; 
                             If False, return global average pooled features
    
    Returns:
        features: Either patch-level features [B, num_patches, embed_dim] 
                 or global features [B, embed_dim]
    """
    self.eval()
    
    patches = torch.permute(self.unfold(x), (0, 2, 1)).to(x.device)
    patches = self.projection(patches)
    
    encoder_input_embeddings = patches + self.positional_embedding
    x = encoder_input_embeddings
    
    for encoder_block in self.encoder_blocks:
        x, _ = encoder_block(x)
    
    if return_patch_features:
        return x
    else:
        return x.mean(dim=1)