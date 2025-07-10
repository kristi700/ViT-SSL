from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    patch_size: int
    in_channels: int
    encoder_embed_dim: int
    encoder_depth: int
    encoder_num_heads: int
    mlp_dim: int
    dropout: float
    output_dim: int
    decoder_embed_dim: Optional[int]
    decoder_depth: Optional[int]
    decoder_num_heads: Optional[int]
    center_momentum: Optional[float]
    mask_ratio: Optional[float]
    num_classes: Optional[int]
