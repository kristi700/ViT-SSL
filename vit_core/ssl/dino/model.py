import copy
import torch

from torch import nn
from typing import List, Tuple

from .head import DINOHead
from vit_core.encoder_block import EncoderBlock
from vit_core.patch_embedding import DynamicPatchEmbedding

# NOTE - might be nicer to move to some generic place for later use (to use it in more places for refactor)
class ViTBackbone(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_shape: Tuple[int, int],
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
                for _ in range(num_blocks)
            ]
        )
        self.patch_embedding = DynamicPatchEmbedding(
            input_shape, embed_dim, patch_size
        )

    def forward(self, x: torch.Tensor, return_attn=False) -> torch.Tensor:
        x = self.patch_embedding(x)
        for encoder_block in self.encoder_blocks:
            x, attn_probs = encoder_block(
                x, return_attn
            )  # NOTE - here we always get the last one, might need some nicer implementation later
        cls_token_output = x[:, 0]

        if return_attn:
            return cls_token_output, attn_probs
        else:
            return cls_token_output


class DINOViT(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_shape,
        embed_dim: int,
        patch_size: int,
        num_heads: int = 8,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        output_dim: int = 65536,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.center_momentum = center_momentum

        self.teacher_backbone = ViTBackbone(
            num_blocks, input_shape, embed_dim, patch_size, num_heads, mlp_dim, dropout
        )
        self.student_backbone = copy.deepcopy(self.teacher_backbone)

        # NOTE- done without deepcopy due to pytorch bug - (https://github.com/pytorch/pytorch/issues/28594)
        self.teacher_head = DINOHead(embed_dim, output_dim)
        self.student_head = DINOHead(embed_dim, output_dim)

        self.student_head.load_state_dict(self.teacher_head.state_dict())

        for param in self.teacher_backbone.parameters():
            param.requires_grad = False

        for param in self.teacher_head.parameters():
            param.requires_grad = False

        self.register_buffer("center", torch.zeros(1, output_dim))

    def _student_forward(self, x: torch.Tensor):
        """
        Performs simple forwarding for student model
        """
        x = self.student_backbone(x)
        x = self.student_head(x)
        return x

    @torch.no_grad()
    def _update_center(self, teacher_output: torch.Tensor):
        """
        As described in: https://arxiv.org/pdf/2104.14294 - Eq 4.
        """
        batch_mean = torch.mean(teacher_output, dim=0)
        self.center = (
            self.center_momentum * self.center + (1 - self.center_momentum) * batch_mean
        )

    def _teacher_forward(self, x: torch.Tensor):
        """
        Performs the teacher model' forward step /w centering.
        """
        x = self.teacher_backbone(x)
        x = self.teacher_head(x)
        self._update_center(x.detach())
        return x

    def forward(self, multi_crop_views: List[torch.Tensor], num_global_views: int):
        """
        Performs both student and teacher models' forwarding methods.
        """
        global_crops = torch.cat(multi_crop_views[:num_global_views], dim=0)
        local_crops = torch.cat(multi_crop_views[num_global_views:], dim=0)

        student_output_global = self._student_forward(global_crops)
        student_output_local = self._student_forward(local_crops)
        student_output = torch.cat([student_output_global, student_output_local])

        with torch.no_grad():
            teacher_output = self._teacher_forward(global_crops)

        return teacher_output, student_output

    @torch.no_grad()
    def momentum_update_teacher(self, teacher_momentum):
        for param_student_bb, param_teacher_bb in zip(
            self.student_backbone.parameters(), self.teacher_backbone.parameters()
        ):
            param_teacher_bb.data.mul_(teacher_momentum).add_(
                (1 - teacher_momentum) * param_student_bb.detach().data
            )
        for param_student_h, param_teacher_h in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            param_teacher_h.data.mul_(teacher_momentum).add_(
                (1 - teacher_momentum) * param_student_h.detach().data
            )

    @torch.no_grad()
    def inference_forward(self, x: torch.Tensor, return_features=False):
        """
        Clean inference forward pass using teacher backbone.

        """
        self.eval()

        features = self.teacher_backbone(x)

        if return_features:
            return features

        output = self.teacher_head(features)
        return output
