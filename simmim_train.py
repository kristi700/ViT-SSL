import os
import math
import torch
import argparse

from datetime import datetime

from utils.logger import Logger
from utils.metrics import MetricHandler
from utils.history import TrainingHistory
from utils.config_parser import load_config
from vit_core.ssl.simmim.model import SimMIMViT
from torch.utils.data import DataLoader, random_split, Subset
from data.datasets import CIFAR10Dataset, STL10Dataset, STL10UnsupervisedDataset
from utils.train_utils import (
    make_criterion,
    make_optimizer,
    make_schedulers,
    get_transforms,
)

# NOTE - will need refactoring (alongside /w supervised_train), for testing purposes as of rightnow!


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stl10_simmim.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    return args


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


def train_one_epoch(
    config,
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    logger: Logger=None,
    scheduler=None,
    warmup_scheduler=None,
    epoch=None,
    warmup_epochs=0,
):
    model.train()
    running_loss = 0
    total = 0
    patch_size = config["model"]["patch_size"]
    in_channels = config["model"]["in_channels"]
    all_pred_patches, all_target_patches = [], []

    for idx, inputs in enumerate(dataloader):
        inputs = inputs.to(device)

        optimizer.zero_grad()
        preds_flat, targets_flat = model(inputs)
        loss = criterion(preds_flat, targets_flat)
        loss.backward()
        optimizer.step()

        if warmup_scheduler is not None and epoch <= warmup_epochs:
            warmup_scheduler.step()

        running_loss += loss.item()
        total += 1

        preds_patches = torch.clamp(
            preds_flat.reshape(-1, in_channels, patch_size, patch_size), 0, 1
        )
        targets_patches = targets_flat.reshape(-1, in_channels, patch_size, patch_size)
        all_pred_patches.append(preds_patches)
        all_target_patches.append(targets_patches)
        logger.train_log_step(epoch, idx)

    all_pred_patches = torch.cat(all_pred_patches, dim=0)
    all_target_patches = torch.cat(all_target_patches, dim=0)
    avg_loss = running_loss / total
    return avg_loss, all_pred_patches, all_target_patches


def evaluate(config, model, dataloader, criterion, device, logger: Logger):
    model.eval()
    total = 0
    running_loss = 0

    patch_size = config["model"]["patch_size"]
    in_channels = config["model"]["in_channels"]
    all_pred_patches, all_target_patches = [], []

    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            preds_flat, targets_flat = model(inputs)
            loss = criterion(preds_flat, targets_flat)
            running_loss += loss.item()
            total += 1

            preds_patches = torch.clamp(
                preds_flat.reshape(-1, in_channels, patch_size, patch_size), 0, 1
            )
            targets_patches = targets_flat.reshape(
                -1, in_channels, patch_size, patch_size
            )
            all_pred_patches.append(preds_patches)
            all_target_patches.append(targets_patches)
            logger.val_log_step(idx)

    all_pred_patches = torch.cat(all_pred_patches, dim=0)
    all_target_patches = torch.cat(all_target_patches, dim=0)
    avg_loss = running_loss / total
    return avg_loss, all_pred_patches, all_target_patches


def main():
    args = parse_args()
    config = load_config(args.config)
    device = setup_device()

    transforms = get_transforms(config)
    train_loader, val_loader = prepare_dataloaders(config, transforms)
    model = build_model(config).to(device)

    num_epochs = config["training"]["num_epochs"]
    warmup_epochs = config["training"]["warmup_epochs"]
    warmup_steps = warmup_epochs * len(train_loader)

    criterion = make_criterion(config)
    optimizer = make_optimizer(config, model)
    schedulers = make_schedulers(config, optimizer, num_epochs, warmup_steps)
    metric_handler = MetricHandler(config)
    rich_logger = Logger(metric_handler.metric_names, len(train_loader), len(val_loader), num_epochs+1)

    save_path = os.path.join(
        config["training"]["checkpoint_dir"],
        config["training"]["type"],
        str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
    )
    model_history = TrainingHistory(save_path)
    best_val_loss = math.inf
    with rich_logger:
        for epoch in range(1, config["training"]["num_epochs"] + 1):
            train_loss, train_pred_patches, train_target_patches = train_one_epoch(
                config,
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                logger=rich_logger,
                scheduler=schedulers["main"],
                warmup_scheduler=schedulers["warmup"],
                epoch=epoch,
                warmup_epochs=warmup_epochs,
            )
            val_loss, val_pred_patches, val_target_patches = evaluate(
                config, model, val_loader, criterion, device, rich_logger
            )

            train_metrics = metric_handler.calculate_metrics(
                preds_patches=train_pred_patches,
                targets_patches=train_target_patches,
            )
            train_metrics["loss"] = train_loss
            rich_logger.log_train_epoch(train_loss, **train_metrics)

            val_metrics = metric_handler.calculate_metrics(
                preds_patches=val_pred_patches,
                targets_patches=val_target_patches,
            )
            val_metrics["loss"] = val_loss
            rich_logger.log_val_epoch(val_loss, **val_metrics)

            model_history.update(train_metrics, val_metrics, epoch)

            if epoch > warmup_epochs:
                schedulers["main"].step()

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                os.makedirs(save_path, exist_ok=True)
                torch.save(checkpoint, os.path.join(save_path, "best_model.pth"))

    model_history.vizualize(num_epochs)


if __name__ == "__main__":
    main()
