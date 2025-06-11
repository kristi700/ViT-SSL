import hydra
import torch

from utils.schemas import Config
from utils.trainers import SimMIMTrainer
from utils.train_utils import get_transforms
from vit_core.ssl.simmim.model import SimMIMViT
from torch.utils.data import DataLoader, random_split, Subset
from data.datasets import CIFAR10Dataset, STL10Dataset, STL10UnsupervisedDataset


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def _get_dataset(config, transform):
    mode = config["training"]["type"].lower()
    dataset_name = config["data"]["dataset_name"].lower()

    if mode == "supervised":
        Warning(
            "Supervised training mode is meant to be used with train_supervised.py!"
        )
        if dataset_name == "cifar10":
            return CIFAR10Dataset(
                config["data"]["data_csv"], config["data"]["data_dir"], transform
            )
        elif dataset_name == "stl10":
            return STL10Dataset(
                config["data"]["data_csv"], config["data"]["data_dir"], transform
            )
        else:
            raise ValueError(f"Unknown supervised dataset {dataset_name}")
    elif mode == "unsupervised":
        if dataset_name == "stl10":
            return STL10UnsupervisedDataset(config["data"]["data_dir"], transform)
        else:
            raise ValueError(f"Unknown unsupervised dataset {dataset_name}")
    else:
        raise ValueError(f"Unknown training type {mode}")


def prepare_dataloaders(config, transforms):
    train_dataset_full = _get_dataset(config, transforms["val"])
    val_dataset_full = _get_dataset(config, transforms["train"])

    total_size = len(train_dataset_full)
    val_size = int(total_size * config["data"]["val_split"])
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(config["training"]["random_seed"])
    train_subset_indices, val_subset_indices = random_split(
        range(total_size), [train_size, val_size], generator=generator
    )

    train_dataset = Subset(train_dataset_full, train_subset_indices.indices)
    val_dataset = Subset(val_dataset_full, val_subset_indices.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(config):
    image_shape = (
        config["model"]["in_channels"],
        config["data"]["img_size"],
        config["data"]["img_size"],
    )
    model = SimMIMViT(
        input_shape=image_shape,
        patch_size=config["model"]["patch_size"],
        embed_dim=config["model"]["embed_dim"],
        num_blocks=config["model"]["num_blocks"],
        num_heads=config["model"]["num_heads"],
        mlp_dim=config["model"]["mlp_dim"],
        dropout=config["model"]["dropout"],
        mask_ratio=config["model"]["mask_ratio"],
    )
    return model

def get_save_path():
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

@hydra.main(config_path="configs", config_name="simmim", version_base=None)
def main(config: Config):
    device = setup_device()
    transforms = get_transforms(config)
    train_loader, val_loader = prepare_dataloaders(config, transforms)
    model = build_model(config).to(device)
    trainer = SimMIMTrainer(model, get_save_path(), config, train_loader, val_loader, device)
    trainer.fit(config["training"]["num_epochs"])

if __name__ == "__main__":
    main()
