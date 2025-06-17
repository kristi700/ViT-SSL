import os
import hydra
import torch

from utils.model_builder import build_model 
from data.data_builder import prepare_dataloaders
from utils.schemas.training_schemas import TrainConfig
from utils.train_utils import get_transforms, setup_device
from utils.trainers import SupervisedTrainer, SimMIMTrainer, DINOTrainer

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
def main(config: TrainConfig):
    """Main training function."""
    mode = config["training"]["type"].lower()
    print(f"Starting training with mode: {mode}")

    device = setup_device()
    transforms = get_transforms(config)

    train_loader, val_loader = prepare_dataloaders(config, transforms, config["training"]["type"].lower())
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
