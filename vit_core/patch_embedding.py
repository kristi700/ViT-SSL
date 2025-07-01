"""
Both Manual and convolutional patching is implemented in this file.
"""
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class DynamicPatchEmbedding(nn.Module):
    """
    Patch embedder that can handle variable input sizes by interpolating
    the positional embeddings.
    """
    def __init__(self, input_shape, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = (input_shape[1] // patch_size, input_shape[2] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(input_shape[0], embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, embed_dim))
        
    def interpolate_pos_encoding(self, x, w, h):
        """
        Helper function to interpolate positional encoding.
        x: The patch embedding tensor (without CLS token)
        w, h: The width and height of the patch grid
        """
        npatch = x.shape[1]
        
        if npatch == self.num_patches and w == h:
            return self.positional_embedding

        class_pos_embed = self.positional_embedding[:, 0]
        patch_pos_embed = self.positional_embedding[:, 1:]

        patch_pos_embed = patch_pos_embed.reshape(1, self.grid_size[0], self.grid_size[1], x.shape[-1]).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(w, h),
            mode='bicubic',
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, x.shape[-1])
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"Input image dimensions ({height}x{width}) must be divisible by patch size ({self.patch_size}).")
    
        x = self.proj(x)
        w, h = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2) 
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        pos_embed = self.interpolate_pos_encoding(x, w, h)
        x = torch.cat((cls_tokens, x), dim=1)

        return x + pos_embed

class ConvolutionalPatchEmbedding(nn.Module):
    """
    Conv2d based patch embedder
    """

    def __init__(self, input_shape, embedding_dimension, patch_size):
        super().__init__()

        if input_shape[1] % patch_size != 0 or input_shape[2] % patch_size != 0:
            raise ValueError(
                f"Image dimensions H={input_shape[1]}, W={input_shape[2]} must be divisible by patch_size={patch_size}"
            )

        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            input_shape[0],
            embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, embedding_dimension))
        self.positional_embedding = nn.Parameter(
            torch.rand(1, (input_shape[1] // patch_size) ** 2 + 1, embedding_dimension)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        x = torch.permute(x, (0, 2, 1))
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x += self.positional_embedding
        return x


class ManualPatchEmbedding(nn.Module):
    """
    Manual patch embedder
    """

    def __init__(self, input_shape, embedding_dimension, patch_size):
        super().__init__()

        if input_shape[1] % patch_size != 0 or input_shape[2] % patch_size != 0:
            raise ValueError(
                f"Image dimensions H={input_shape[1]}, W={input_shape[2]} must be divisible by patch_size={patch_size}"
            )

        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)
        self.linear = nn.Linear(
            input_shape[0] * patch_size * patch_size, embedding_dimension
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, embedding_dimension))
        self.positional_embedding = nn.Parameter(
            torch.rand(1, (input_shape[1] // patch_size) ** 2 + 1, embedding_dimension)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unfold(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.linear(x)
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x += self.positional_embedding
        return x
