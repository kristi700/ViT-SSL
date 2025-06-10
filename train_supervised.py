import os
import torch
import argparse

from datetime import datetime
from utils.logger import Logger
from data.datasets import CIFAR10Dataset, STL10Dataset
from torch.utils.data import DataLoader, random_split, Subset

from vit_core.vit import ViT
from utils.metrics import MetricHandler
from utils.history import TrainingHistory
from utils.config_parser import load_config
from utils.train_utils import (
    make_criterion,
    make_optimizer,
    make_schedulers,
    get_transforms,
)


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stl10.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    return args


def _get_dataset(config, transform):
    mode = config["training"]["type"].lower()
    dataset_name = config["data"]["dataset_name"].lower()

    if mode == "supervised" or mode == "finetune":
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
        raise NotImplementedError("For unsupervised training use simmim_train.py!")
    else:
        raise ValueError(f"Unknown training type {mode}")


def prepare_dataloaders(config, transforms):
    train_dataset_full = _get_dataset(config, transforms["train"])
    val_dataset_full = _get_dataset(config, transforms["val"])

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
    if config["training"]["type"].lower() == "supervised":
        return model
    elif config["training"]["type"].lower() == "finetune":
        model = load_pretrained_model(config, model)
        if config["training"]["freeze_backbone"]:
            for param in model.encoder_blocks.parameters():
                param.requires_grad = False
            for name, param in model.patch_embedding.named_parameters():
                if (
                    "cls_token" not in name
                ):  # NOTE - as SimMIM ViT didnt have CLS token, its initialized w/ 0 now - dont wanna freeze
                    param.requires_grad = False
        _check_loaded_model(model, config)
        return model
    else:
        raise KeyError("Not supported training type for train_supervised.py")


def load_pretrained_model(config, model: ViT) -> ViT:
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
                ft_pe = model_ft_state_dict[
                    "patch_embedding.positional_embedding"
                ]  # Shape (1, N+1, D)
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
    print(f"\nMissing keys: {missing_keys}\n")
    print(f"Unexpected_keys keys: {unexpected_keys}\n")
    return model


def _check_loaded_model(model, config):
    print("\n=== Checking loaded model ===\n")

    frozen = []
    trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    print(f"Trainable parameters ({len(trainable)}):")
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


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    logger: Logger = None,
    scheduler=None,
    warmup_scheduler=None,
    epoch=None,
    warmup_epochs=0,
):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        if warmup_scheduler is not None and epoch <= warmup_epochs:
            warmup_scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)
        logger.train_log_step(epoch, idx)

    avg_loss = running_loss / total
    return avg_loss, correct, total


def evaluate(model, dataloader, criterion, device, logger):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)
            logger.val_log_step(idx)

    avg_loss = total_loss / total
    return avg_loss, correct, total


def _unfreeze_backbone(model):
    for param in model.patch_embedding.parameters():
        param.requires_grad = True
    for param in model.encoder_blocks.parameters():
        param.requires_grad = True
    return model


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
    
    best_val_acc = 0.0
    save_path = os.path.join(
        config["training"]["checkpoint_dir"],
        config["training"]["type"],
        str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
    )
    model_history = TrainingHistory(save_path)

    with rich_logger:
        for epoch in range(1, config["training"]["num_epochs"] + 1):
            if config["training"].get("freeze_backbone", False) and epoch > config[
                "training"
            ].get("freeze_backbone_epochs", float("inf")):
                model = _unfreeze_backbone(model)
                optimizer = make_optimizer(config, model)

            train_loss, train_correct, train_total = train_one_epoch(
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
            val_loss, val_correct, val_total = evaluate(
                model, val_loader, criterion, device, rich_logger
            )

            train_metrics = metric_handler.calculate_metrics(
                correct=train_correct, total=train_total
            )
            train_metrics["loss"] = train_loss
            rich_logger.log_train_epoch(train_loss, **train_metrics)

            val_metrics = metric_handler.calculate_metrics(
                correct=val_correct, total=val_total
            )
            val_metrics["loss"] = val_loss
            rich_logger.log_val_epoch(val_loss, **val_metrics)

            model_history.update(train_metrics, val_metrics, epoch)

            if epoch > warmup_epochs:
                schedulers["main"].step()

            if val_metrics["Accuracy"] > best_val_acc:
                best_val_acc = val_metrics["Accuracy"]
                print(f"New best validation accuracy: {best_val_acc:.4f}. Saving model...")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": config,
                }
                os.makedirs(save_path, exist_ok=True)
                torch.save(checkpoint, os.path.join(save_path, "best_model.pth"))

    model_history.vizualize(num_epochs)


if __name__ == "__main__":
    main()
