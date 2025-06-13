from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    patch_size: int
    in_channels: int
    embed_dim: int
    num_blocks: int
    num_heads: int
    mlp_dim: int
    dropout: float
    output_dim: int
    center_momentum: Optional[float]
    mask_ratio: Optional[float]
    num_classes: Optional[int]
