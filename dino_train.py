import os
import math
import torch
import argparse

from tqdm import tqdm
from datetime import datetime

from utils.logger import Logger
from utils.metrics import MetricHandler
from utils.history import TrainingHistory
from data.datasets import STL10DINODataset
from vit_core.ssl.dino.loss import DINOLoss
from utils.config_parser import load_config
from vit_core.ssl.dino.model import DINOViT
from torch.utils.data import DataLoader, random_split, Subset
from vit_core.ssl.dino.dino_utils import DINOMomentumScheduler
from utils.train_utils import make_optimizer, make_schedulers, get_transforms

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
        default="configs/stl10_dino.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    return args


def _get_dataset(config, transforms):
    mode = config["training"]["type"].lower()
    dataset_name = config["data"]["dataset_name"].lower()

    if mode == "dino_unsupervised":
        if dataset_name == "stl10":
            return STL10DINODataset(
                config["data"]["data_dir"],
                transforms=transforms,
                num_all_views=config["training"]["num_all_views"],
                num_global_views=config["training"]["num_global_views"],
            )
        else:
            raise ValueError(f"Unknown unsupervised dataset {dataset_name}")
    else:
        raise ValueError(f"Not supported type {mode}")


def prepare_dataloaders(config, transforms):
    train_dataset_full = _get_dataset(config, transforms)
    val_dataset_full = _get_dataset(config, transforms)

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
    model = DINOViT(
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
    return model


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    current_teach_momentum,
    epoch_desc="Training",
    scheduler=None,
    warmup_scheduler=None,
    current_epoch=None,
    warmup_epochs=0,
):
    model.train()
    running_loss = 0
    total = 0
    pbar = tqdm(dataloader, desc=epoch_desc, leave=False)
    num_global_views = (
        dataloader.dataset.dataset.num_global_views
    )  # TODO - might not be ideal like this

    for inputs in pbar:
        inputs = [x.to(device) for x in inputs]

        optimizer.zero_grad()
        teacher_output, student_output = model(inputs, num_global_views)

        teacher_output = teacher_output.view(
            num_global_views,
            int(teacher_output.shape[0] / num_global_views),
            teacher_output.shape[1],
        )
        student_output = student_output.view(
            len(inputs),
            int(student_output.shape[0] / len(inputs)),
            student_output.shape[1],
        )

        loss = criterion(teacher_output, student_output, model.center)
        loss.backward()
        optimizer.step()
        model.momentum_update_teacher(current_teach_momentum)
        if warmup_scheduler is not None and current_epoch <= warmup_epochs:
            warmup_scheduler.step()

        running_loss += loss.item()
        total += 1
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}"
        )

    avg_loss = running_loss / total

    return avg_loss, teacher_output, student_output


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total = 0
    running_loss = 0
    num_global_views = (
        dataloader.dataset.dataset.num_global_views
    )  # TODO - might not be ideal like this

    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Validation"):
            inputs = [x.to(device) for x in inputs]
            teacher_output, student_output = model(inputs, num_global_views)

            teacher_output = teacher_output.view(
                num_global_views,
                int(teacher_output.shape[0] / num_global_views),
                teacher_output.shape[1],
            )
            student_output = student_output.view(
                len(inputs),
                int(student_output.shape[0] / len(inputs)),
                student_output.shape[1],
            )

            loss = criterion(teacher_output, student_output, model.center)
            running_loss += loss.item()
            total += 1

    avg_loss = running_loss / total

    return avg_loss, teacher_output, student_output


def print_epoch_summary(epoch: int, metrics):

    print(
        f"\nEpoch {epoch} Summary: "
        f"Center Norm: {metrics['center_norm']:.4f} | "
        f"Train Loss: {metrics['train_loss']:.4f} | "
        f"Train Teacher Embed (Mean: {metrics['train_teacher_embed_mean']:.4f}, Std: {metrics['train_teacher_embed_std']:.4f}, Var: {metrics['train_teacher_embed_var']:.4f}) | "
        f"Train Student Embed (Mean: {metrics['train_student_embed_mean']:.4f}, Std: {metrics['train_student_embed_std']:.4f}, Var: {metrics['train_student_embed_var']:.4f}) | "
        f"Train Cosine: {metrics['train_cosine_similarity']:.4f} | "
        f"Val Loss: {metrics['validation_loss']:.4f} | "
        f"Val Teacher Embed (Mean: {metrics['val_teacher_embed_mean']:.4f}, Std: {metrics['val_teacher_embed_std']:.4f}, Var: {metrics['val_teacher_embed_var']:.4f}) | "
        f"Val Student Embed (Mean: {metrics['val_student_embed_mean']:.4f}, Std: {metrics['val_student_embed_std']:.4f}, Var: {metrics['val_student_embed_var']:.4f}) | "
        f"Val Cosine: {metrics['validation_cosine_similarity']:.4f} | "
        f"LR: {metrics['learning_rate']:.1e}"
    )


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

    criterion = DINOLoss(config.training.teacher_temp, config.training.student_temp)
    optimizer = make_optimizer(config, model)
    schedulers = make_schedulers(config, optimizer, num_epochs, warmup_steps)
    momentum_schedule = DINOMomentumScheduler(
        config.training.teacher_momentum_start,
        config.training.teacher_momentum_final,
        num_epochs,
    )
    metric_handler = MetricHandler(config)
    #rich_logger = Logger(metric_handler.metric_names, 1)  # TODO - steps needs to be implemented!

    save_path = os.path.join(
        config["training"]["checkpoint_dir"],
        config["training"]["type"],
        str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
    )
    model_history = TrainingHistory(save_path)
    best_val_loss = math.inf
    for epoch in range(1, config["training"]["num_epochs"] + 1):
        epoch_desc = f"Epoch {epoch}/{config['training']['num_epochs']}"
        current_teach_momentum = momentum_schedule.get_momentum(epoch)
        train_loss, train_teacher_output, train_student_output = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            current_teach_momentum,
            epoch_desc=epoch_desc,
            scheduler=schedulers["main"],
            warmup_scheduler=schedulers["warmup"],
            current_epoch=epoch,
            warmup_epochs=warmup_epochs,
        )
        val_loss, val_teacher_output, val_student_output = evaluate(
            model, val_loader, criterion, device
        )

        train_metrics = metric_handler.calculate_metrics(
            center=model.center,
            teacher_distribution=train_teacher_output,
            student_distribution=train_student_output,
        )
        train_metrics["loss"] = train_loss

        val_metrics = metric_handler.calculate_metrics(
            center=model.center,
            teacher_distribution=val_teacher_output,
            student_distribution=val_student_output,
        )
        val_metrics["loss"] = val_loss
        model_history.update(train_metrics, val_metrics, epoch)

        if epoch > warmup_epochs:
            schedulers["main"].step()

        # print_epoch_summary(epoch, metric_handler.get_metric_values())

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
