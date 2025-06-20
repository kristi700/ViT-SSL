import torch
import logging

from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms as T

from .schedulers import LinearWarmupScheduler

logger = logging.getLogger(__name__)

def setup_device():
    """Setup and return the appropriate device (CUDA/CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def make_criterion(config):
    crit_config = config["training"]["criterion"]
    cls = getattr(nn, crit_config["name"])
    return cls(**crit_config.get("params", {}))


def make_optimizer(config, model):
    opt_config = config["training"]["optimizer"]
    cls = getattr(optim, opt_config["name"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return cls(trainable_params, **opt_config.get("params", {}))


def make_schedulers(config, optimizer, num_epochs, warmup_steps):
    sched_config = config["training"]["lr_scheduler"]
    main = sched_config["main"]
    warm = sched_config["warmup"]
    main_cls = getattr(lr_scheduler, main["name"])

    main_kwargs = dict(
        main.get("params", {}), T_max=num_epochs - config["training"]["warmup_epochs"]
    )
    warm_kwargs = dict(
        warm.get("params", {}),
        warmup_steps=warmup_steps,
        start_lr=config["training"]["warmup_initial_learning_rate"],
        target_lr=config["training"]["warmup_final_learning_rate"],
    )

    return {
        "main": main_cls(optimizer, **main_kwargs),
        "warmup": LinearWarmupScheduler(optimizer, **warm_kwargs),
    }


def get_transforms(config):
    transform_cfg = config["transforms"]
    transforms_dict = {}
    for key, sequence in transform_cfg.items():
        transforms_dict[key] = _make_transforms(sequence)
    return transforms_dict


def _make_transforms(sequence):
    ops = []
    for entry in sequence:
        cls = getattr(T, entry["name"])
        params = entry.get("params") or {}
        ops.append(cls(**params))
    return T.Compose(ops)
