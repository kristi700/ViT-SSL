import os
import hydra
import torch

from torch.utils.data import DataLoader, random_split, Subset

from vit_core.vit import ViT
from utils.train_utils import get_transforms
from vit_core.ssl.dino.model import DINOViT
from vit_core.ssl.simmim.model import SimMIMViT
from utils.schemas.training_schemas import Config
from utils.trainers import SupervisedTrainer, SimMIMTrainer, DINOTrainer
from data.datasets import (
    CIFAR10Dataset,
    STL10Dataset,
    STL10UnsupervisedDataset,
    STL10DINODataset,
)


def setup_device():
    """Setup and return the appropriate device (CUDA/CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def _get_dataset(config, transform=None, transforms=None):
    """
    Get the appropriate dataset based on training mode and dataset name.

    Args:
        config: Training configuration
        transform: Single transform (for supervised/unsupervised)
        transforms: Dict of transforms (for DINO)
    """
    mode = config["training"]["type"].lower()
    dataset_name = config["data"]["dataset_name"].lower()

    if mode in ["supervised", "finetune"]:
        if dataset_name == "cifar10":
            return CIFAR10Dataset(
                config["data"]["data_csv"], config["data"]["data_dir"], transform
            )
        elif dataset_name == "stl10":
            return STL10Dataset(
                config["data"]["data_csv"], config["data"]["data_dir"], transform
            )
        else:
            raise ValueError(f"Unknown supervised dataset: {dataset_name}")

    elif mode == "unsupervised":
        if dataset_name == "stl10":
            return STL10UnsupervisedDataset(config["data"]["data_dir"], transform)
        else:
            raise ValueError(f"Unknown unsupervised dataset: {dataset_name}")

    elif mode == "dino_unsupervised":
        if dataset_name == "stl10":
            return STL10DINODataset(
                config["data"]["data_dir"],
                transforms=transforms,
                num_all_views=config["training"]["num_all_views"],
                num_global_views=config["training"]["num_global_views"],
            )
        else:
            raise ValueError(f"Unknown DINO dataset: {dataset_name}")

    else:
        raise ValueError(f"Unknown training type: {mode}")


def prepare_dataloaders(config, transforms):
    """Prepare train and validation dataloaders based on training mode."""
    mode = config["training"]["type"].lower()

    if mode == "dino_unsupervised":
        train_dataset_full = _get_dataset(config, transforms=transforms)
        val_dataset_full = _get_dataset(config, transforms=transforms)
    else:
        train_dataset_full = _get_dataset(config, transform=transforms["train"])
        val_dataset_full = _get_dataset(config, transform=transforms["val"])

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
    """Build the appropriate model based on training mode."""
    mode = config["training"]["type"].lower()

    image_shape = (
        config["model"]["in_channels"],
        config["data"]["img_size"],
        config["data"]["img_size"],
    )

    if mode in ["supervised", "finetune"]:
        model = ViT(
            input_shape=image_shape,
            patch_size=config["model"]["patch_size"],
            num_classes=config["model"]["num_classes"],
            embed_dim=config["model"]["embed_dim"],
            num_blocks=config["model"]["num_blocks"],
            num_heads=config["model"]["num_heads"],
            mlp_dim=config["model"]["mlp_dim"],
            dropout=config["model"]["dropout"],
        )

        if mode == "finetune":
            model = load_pretrained_model(config, model)
            if config["training"]["freeze_backbone"]:
                freeze_backbone(model)
            _check_loaded_model(model, config)

        return model

    elif mode == "unsupervised":
        return SimMIMViT(
            input_shape=image_shape,
            patch_size=config["model"]["patch_size"],
            embed_dim=config["model"]["embed_dim"],
            num_blocks=config["model"]["num_blocks"],
            num_heads=config["model"]["num_heads"],
            mlp_dim=config["model"]["mlp_dim"],
            dropout=config["model"]["dropout"],
            mask_ratio=config["model"]["mask_ratio"],
        )

    elif mode == "dino_unsupervised":
        return DINOViT(
            input_shape=image_shape,
            patch_size=config["model"]["patch_size"],
            embed_dim=config["model"]["embed_dim"],
            num_blocks=config["model"]["num_blocks"],
            num_heads=config["model"]["num_heads"],
            mlp_dim=config["model"]["mlp_dim"],
            dropout=config["model"]["dropout"],
            output_dim=config["model"]["output_dim"],
            center_momentum=config["model"]["center_momentum"],
        )

    else:
        raise ValueError(f"Unknown training mode: {mode}")


def freeze_backbone(model):
    """Freeze the backbone parameters for fine-tuning."""
    for param in model.encoder_blocks.parameters():
        param.requires_grad = False
    for name, param in model.patch_embedding.named_parameters():
        if "cls_token" not in name:  # Don't freeze CLS token as it's newly initialized
            param.requires_grad = False


def load_pretrained_model(config, model: ViT) -> ViT:
    """Load pretrained weights into the model."""
    pretrained_checkpoint = torch.load(config["training"]["pretrained_path"])
    pretrained_state_dict = pretrained_checkpoint["model_state_dict"]
    model_ft_state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k in model_ft_state_dict:
            if v.shape == model_ft_state_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(
                    f"Shape mismatch for {k}: Pretrained {v.shape} vs Fine-tune {model_ft_state_dict[k].shape}"
                )
        elif (
            k.startswith("projection.")
            and f"patch_embedding.{k}" in model_ft_state_dict
        ):
            new_key = f"patch_embedding.{k}"
            if v.shape == model_ft_state_dict[new_key].shape:
                new_state_dict[new_key] = v
            else:
                print(f"Shape mismatch for {new_key} (from {k})")
        elif k == "positional_embedding":
            if "patch_embedding.positional_embedding" in model_ft_state_dict:
                ft_pe = model_ft_state_dict["patch_embedding.positional_embedding"]
                if v.shape[1] == ft_pe.shape[1] - 1 and v.shape[2] == ft_pe.shape[2]:
                    print(f"Interpolating positional embedding for {k}...")
                    new_pe = torch.zeros_like(ft_pe)
                    new_pe[:, 1:, :] = v
                    new_state_dict["patch_embedding.positional_embedding"] = new_pe
                else:
                    print(
                        f"Cannot interpolate positional_embedding: Pretrained {v.shape} vs Fine-tune {ft_pe.shape}"
                    )
        elif "simmim_head" in k or "mask_token" in k:
            print(f"Skipping SimMIM specific key: {k}")
        else:
            print(f"Key {k} from pretrained checkpoint not found in fine-tuning model.")

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"\nMissing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    return model


def _check_loaded_model(model, config):
    """Check which parameters are frozen/trainable and verify loading."""
    print("\n=== Checking loaded model ===")

    frozen = []
    trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    print(f"\nTrainable parameters ({len(trainable)}):")
    for name in trainable:
        print(f"[✓] {name}")

    print(f"\nFrozen parameters ({len(frozen)}):")
    for name in frozen:
        print(f"[-] {name}")

    if config["training"]["type"].lower() == "finetune":
        pretrained_checkpoint = torch.load(
            config["training"]["pretrained_path"],
            map_location=next(model.parameters()).device,
        )
        pretrained_state_dict = pretrained_checkpoint["model_state_dict"]

        matched = 0
        mismatched = 0
        for name, param in model.named_parameters():
            if name in pretrained_state_dict:
                pre_param = pretrained_state_dict[name]
                if pre_param.shape == param.shape and torch.allclose(
                    param.data, pre_param, atol=1e-5
                ):
                    matched += 1
                else:
                    print(f"[!] Weight mismatch in: {name}")
                    mismatched += 1

        print(f"\nMatched parameters from checkpoint: {matched}")
        print(f"Mismatched parameters: {mismatched}")

    print("\n=== Model check complete ===\n")


def load_checkpoint_if_exists(config, model, device):
    """
    Load checkpoint if resume_from_checkpoint is specified in config.
    """
    resume_path = config["training"].get("resume_from_checkpoint", None)

    if resume_path is None or not os.path.exists(resume_path):
        if resume_path is not None:
            print(
                f"Warning: Resume path {resume_path} does not exist. Starting from scratch."
            )
        return 0, float("inf")

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"Resuming from epoch {start_epoch + 1}")
    print(f"Best validation loss so far: {best_val_loss}")

    return start_epoch, best_val_loss


def get_trainer(
    mode,
    model,
    save_path,
    config,
    train_loader,
    val_loader,
    device,
    start_epoch=0,
    best_val_loss=float("inf"),
):
    """Get the appropriate trainer based on training mode."""
    if mode in ["supervised", "finetune"]:
        trainer = SupervisedTrainer(
            model, save_path, config, train_loader, val_loader, device
        )
    elif mode == "unsupervised":
        trainer = SimMIMTrainer(
            model, save_path, config, train_loader, val_loader, device
        )
    elif mode == "dino_unsupervised":
        trainer = DINOTrainer(
            model, save_path, config, train_loader, val_loader, device
        )
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    if start_epoch > 0:
        resume_path = config["training"].get("resume_from_checkpoint", None)
        if resume_path and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            if hasattr(trainer, "optimizer") and "optimizer_state_dict" in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Loaded optimizer state from checkpoint")

        trainer.start_epoch = start_epoch
        trainer.best_val_loss = best_val_loss

        if mode == "dino_unsupervised" and hasattr(trainer, "momentum_schedule"):
            current_momentum = trainer.momentum_schedule.get_momentum(start_epoch)
            print(f"Current teacher momentum: {current_momentum:.6f}")

    return trainer


def get_save_path():
    """Get the save path from Hydra configuration."""
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: Config):
    """Main training function."""
    mode = config["training"]["type"].lower()
    print(f"Starting training with mode: {mode}")

    device = setup_device()
    transforms = get_transforms(config)

    train_loader, val_loader = prepare_dataloaders(config, transforms)
    model = build_model(config).to(device)
    
    start_epoch, best_val_loss = load_checkpoint_if_exists(config, model, device)

    trainer = get_trainer(
        mode,
        model,
        get_save_path(),
        config,
        train_loader,
        val_loader,
        device,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
    )
    trainer.fit(config["training"]["num_epochs"])

    print(f"Training completed for mode: {mode}")


if __name__ == "__main__":
    main()
