import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split, Subset

from .datasets import (
    CIFAR10Dataset,
    STL10Dataset,
    STL10UnsupervisedDataset,
    STL10DINODataset,
)


def _get_dataset(config, mode, transforms):
    """
    Internal helper to instantiate the correct dataset.
    """

    data_cfg = config.get("eval", config.get("data", config))
    dataset_name = data_cfg.dataset_name.lower()
    data_dir = data_cfg.data_dir

    data_csv = config.get("eval", {}).get("data_csv", data_cfg.get("data_csv"))

    if mode in ["supervised", "finetune", "eval_knn", "eval_linear", "eval_umap"]:
        if dataset_name == "cifar10":
            return CIFAR10Dataset(
                data_csv, data_dir, transform=transforms["train"]
            ), CIFAR10Dataset(data_csv, data_dir, transform=transforms["val"])
        elif dataset_name == "stl10":
            return STL10Dataset(
                data_csv, data_dir, transform=transforms["train"]
            ), STL10Dataset(data_csv, data_dir, transform=transforms["val"])
        else:
            raise ValueError(f"Unknown supervised/labeled dataset: {dataset_name}")

    elif mode == "unsupervised":
        if dataset_name == "stl10":

            dataset = STL10UnsupervisedDataset(data_dir, transform=transforms["train"])
            return dataset, dataset
        else:
            raise ValueError(f"Unknown unsupervised dataset: {dataset_name}")

    elif mode in ["dino_unsupervised", "eval_dino"]:
        if dataset_name == "stl10":

            dataset = STL10DINODataset(
                data_dir,
                transforms=transforms,
                num_all_views=config.training.num_all_views,
                num_global_views=config.training.num_global_views,
            )
            return dataset, dataset
        else:
            raise ValueError(f"Unknown DINO dataset: {dataset_name}")

    else:
        raise ValueError(f"Unknown mode for dataset creation: {mode}")


def determine_dataset_mode_from_eval_modes(eval_modes):
    """
    Determine the appropriate dataset mode based on evaluation modes.
    All evaluation modes use supervised/labeled datasets.

    Args:
        eval_modes (str or list): Single mode or list of evaluation modes

    Returns:
        str: The dataset mode to use for data loading
    """
    if isinstance(eval_modes, str):
        eval_modes = [eval_modes]

    supervised_modes = {
        "eval_knn",
        "eval_linear",
        "eval_umap",
        "supervised",
        "finetune",
    }

    if any(mode in supervised_modes for mode in eval_modes):
        return "supervised"

    return eval_modes[0] if eval_modes else "supervised"


def prepare_dataloaders(config, transforms, mode):
    """
    Prepare train and validation dataloaders based on the configuration.

    This function is the single entry point for creating dataloaders for any
    training or evaluation mode.

    Args:
        config (OmegaConf): The global configuration object.
        transforms (dict): A dictionary of transforms, e.g., {'train': ..., 'val': ...}.
        mode(str): type of dataset

    Returns:
        tuple: (train_loader, val_loader)
    """
    if OmegaConf.is_list(mode):
        data_loading_mode = determine_dataset_mode_from_eval_modes(mode)
        print(f"Multiple evaluation modes detected: {mode}")
        print(f"Using dataset mode: '{data_loading_mode}' for data loading")
    else:
        data_loading_mode = mode.lower()
        print(f"Preparing dataloaders for mode: '{data_loading_mode}'")

    train_dataset_full, val_dataset_full = _get_dataset(
        config, data_loading_mode, transforms
    )

    total_size = len(train_dataset_full)
    val_split_ratio = config.data.val_split

    if val_split_ratio <= 0 or val_split_ratio >= 1:
        train_size = total_size
        val_size = 0
    else:
        val_size = int(total_size * val_split_ratio)
        train_size = total_size - val_size

    generator = torch.Generator().manual_seed(config.training.random_seed)

    if val_size > 0:
        train_subset_indices, val_subset_indices = random_split(
            range(total_size), [train_size, val_size], generator=generator
        )
        train_dataset = Subset(train_dataset_full, train_subset_indices.indices)
        val_dataset = Subset(val_dataset_full, val_subset_indices.indices)
    else:

        train_dataset = train_dataset_full
        val_dataset = None

    batch_size = config.get("training", {}).get(
        "batch_size", config.get("eval", {}).get("batch_size")
    )
    num_workers = config.data.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
