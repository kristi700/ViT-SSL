import os
import hydra
import torch
import shutil
import logging 

from utils.model_builder import build_model 
from data.data_builder import prepare_dataloaders
from utils.schemas.training_schemas import TrainConfig
from utils.train_utils import get_transforms, setup_device
from utils.trainers import SupervisedTrainer, SimMIMTrainer, DINOTrainer, MAETrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log")
    ]
)
logger = logging.getLogger(__name__)

def load_checkpoint_if_exists(config, model, device):
    """
    Load checkpoint if resume_from_checkpoint is specified in config.
    """
    resume_path = config["training"].get("resume_from_checkpoint", None)

    if resume_path is None or not os.path.exists(resume_path):
        if resume_path is not None:
            logger.warning(f"Resume path {resume_path} does not exist. Starting from scratch.")
        return 0, float("inf")

    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    logger.info(f"Resuming from epoch {start_epoch + 1}.")
    logger.info(f"Best validation loss so far: {best_val_loss}.")
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
    elif mode == "simmim":
        trainer = SimMIMTrainer(
            model, save_path, config, train_loader, val_loader, device
        )
    elif mode == "dino":
        trainer = DINOTrainer(
            model, save_path, config, train_loader, val_loader, device
        )
    elif mode == "mae":
        trainer = MAETrainer(
            model, save_path, config, train_loader, val_loader, device
        )       
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    if start_epoch > 0:
        resume_path = config["training"].get("resume_from_checkpoint", None)
        if resume_path and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            if hasattr(trainer, "optimizer") and "optimizer_state_dict" in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded optimizer state from checkpoint.")

        trainer.start_epoch = start_epoch
        trainer.best_val_loss = best_val_loss

        if mode == "dino" and hasattr(trainer, "momentum_schedule"):
            current_momentum = trainer.momentum_schedule.get_momentum(start_epoch)
            logger.info(f"Current teacher momentum: {current_momentum:.6f}")

    return trainer


def get_save_path(config: TrainConfig):
    """Get the save path from Hydra configuration."""
    resume_model = config['training'].get('resume_from_checkpoint', None)
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if resume_model is not None:
        resume_path=os.path.dirname(resume_model)
        assert os.path.exists(resume_path), f"resume_from_checkpoint: {resume_path} does not exist!"
        shutil.rmtree(hydra_output_dir, ignore_errors=True)

        return resume_path
    else:
        return hydra_output_dir

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: TrainConfig):
    """Main training function."""
    mode = config["training"]["type"].lower()
    logger.info(f"Starting training with mode: {mode}")

    device = setup_device()
    transforms = get_transforms(config)

    train_loader, val_loader = prepare_dataloaders(config, transforms, config["training"]["type"].lower())
    model = build_model(config).to(device)
    
    start_epoch, best_val_loss = load_checkpoint_if_exists(config, model, device)

    trainer = get_trainer(
        mode,
        model,
        get_save_path(config),
        config,
        train_loader,
        val_loader,
        device,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
    )
    trainer.fit(config["training"]["num_epochs"])

    logger.info(f"Training completed for mode: {mode}")

if __name__ == "__main__":
    main()
