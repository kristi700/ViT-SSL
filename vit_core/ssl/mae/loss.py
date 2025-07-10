import torch

from torch import nn

class MAELoss(nn.Module):
    """
    Implemented as given in the original paper (https://arxiv.org/pdf/2111.06377)
        - MSE on reconstructed pixels (masked during decoding)
        - pixel values are normalized with STD and Var coming from ALL pixel values
    """
    def __init__(self, patch_size: int):
        super().__init__()
        self.patchify = nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)

    def forward(self, x: torch.Tensor, masked_indices: torch.Tensor, y: torch.Tensor):
        """
            Args:
                - x: Reconstructed image, with shapes (batch * num_patches * input_channel * patch_size * patch_size)
                - masked_indices: masked_patches indicies (batch * num_patches - (num_patches * (1 - mask_ratio)))
                - y: targets
        """
        gt_patches = torch.permute(self.patchify(y), (0, 2, 1)).to(y.device)
        std = torch.std(gt_patches, dim=-1, keepdim=True)
        mean = torch.mean(gt_patches, dim=-1, keepdim=True)
        gt_patches = (gt_patches - mean) / std
        masked_indices_expanded = masked_indices.unsqueeze(-1).expand(-1, -1, gt_patches.shape[-1])
        gt_patches = torch.gather(gt_patches, dim=1, index=masked_indices_expanded)
        pred_patches = torch.gather(x, dim=1, index=masked_indices_expanded)

        loss = torch.mean((gt_patches - pred_patches) **2)
        return loss

