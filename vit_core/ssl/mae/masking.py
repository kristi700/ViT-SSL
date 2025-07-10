import torch

from typing import Union


def get_and_apply_mask(
    patches: torch.Tensor, mask_ratio: float
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates masking for mask_raito percent of the patches and applies it
    
    Returns:
        - visible_patches: Processed patches
        - masked_indicies: Indicies of the masked patches for loss calculation
        - resolve_ids: the ids for deshuffling patches
    """
    batch_num, patch_num, ddim = patches.shape
    num_visible = int(patch_num * (1 - mask_ratio))

    noise = torch.rand(batch_num, patch_num, device=patches.device)
    shuffle_ids = torch.argsort(noise, dim=1)
    resolve_ids = torch.argsort(shuffle_ids, dim=1)

    visible_ids = shuffle_ids[:, :num_visible]
    masked_ids = shuffle_ids[:, num_visible:]

    visible_ids_expanded = visible_ids.unsqueeze(-1).expand(-1, -1, ddim)
    visible_patches = torch.gather(patches, dim=1, index=visible_ids_expanded)

    return visible_patches, masked_ids, resolve_ids