"""
Both Manual and convolutional patching is implemented in this file.
"""
import torch
import torch.nn as nn

class ConvolutionalPatchEmbedding(nn.Module):
    """
    Conv2d based patch embedder
    """
    def __init__(self, input_shape, embedding_dimension, patch_size):
        super().__init__()

        if input_shape[1] % patch_size != 0 or input_shape[2] % patch_size != 0:
            raise ValueError(f"Image dimensions H={input_shape[1]}, W={input_shape[2]} must be divisible by patch_size={patch_size}")

        self.patch_size = patch_size
        self.conv = nn.Conv2d(input_shape[0], embedding_dimension, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.rand(1,1,embedding_dimension))
        self.positional_embedding = nn.Parameter(torch.rand(1, (input_shape[1] // patch_size) ** 2 + 1, embedding_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        x = torch.permute(x, (0, 2, 1))
        x = torch.cat([ self.cls_token.repeat(x.shape[0], 1 ,1), x], dim=1)
        x += self.positional_embedding
        return  x

class ManualPatchEmbedding(nn.Module):
    """
    Manual patch embedder
    """
    def __init__(self, input_shape, embedding_dimension, patch_size):
        super().__init__()

        if input_shape[1] % patch_size != 0 or input_shape[2] % patch_size != 0:
            raise ValueError(f"Image dimensions H={input_shape[1]}, W={input_shape[2]} must be divisible by patch_size={patch_size}")
        
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)
        self.linear = nn.Linear(input_shape[0] * patch_size * patch_size, embedding_dimension)
        self.cls_token = nn.Parameter(torch.rand(1,1,embedding_dimension))
        self.positional_embedding = nn.Parameter(torch.rand(1, (input_shape[1] // patch_size) ** 2 + 1, embedding_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unfold(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.linear(x)
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1 ,1), x], dim=1)
        x += self.positional_embedding
        return  x
