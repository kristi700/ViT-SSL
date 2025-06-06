import torch

from typing import Union


def simple_masking(
    patches: torch.Tensor, mask_ratio: float
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        - patches: pre-projected patches (batch * num_patches * (Channel * patch_size * patch_size))
        - mask_ratio: percent of patches to mask
    Output:
        - patches: In this implementation, the same as the input patches
        - bool_mask: mask representing which patches to mask (with boolean True) (batch * num_patches)
        - targets: target output for simmim head (masked_patches * (Channel * patch_size * patchsize))
    """
    device = patches.device
    batch_size, num_patches, _ = patches.shape

    num_masked = int(num_patches * mask_ratio)
    mask_indicies = [
        torch.randperm(num_patches, device=device)[:num_masked]
        for _ in range(batch_size)
    ]
    mask_indicies = torch.stack(mask_indicies, dim=0)
    bool_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
    rows = (
        torch.arange(mask_indicies.size(0), device=device)
        .unsqueeze(1)
        .expand_as(mask_indicies)
    )  # NOTE could be batch_size instead of mask_indicies.size(0)
    bool_mask[rows, mask_indicies] = True

    targets = patches[bool_mask]

    return patches, bool_mask, targets
