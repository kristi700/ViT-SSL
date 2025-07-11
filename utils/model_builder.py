import os
import torch
import logging

from vit_core.vit import ViT
from vit_core.ssl.mae import MAEViT
from vit_core.ssl.dino.model import DINOViT
from vit_core.ssl.simmim.model import SimMIMViT
logger = logging.getLogger(__name__)

def load_weights(model, checkpoint_path: str):
    """
    Loads weights from a checkpoint file into a model.

    This function handles various scenarios, including:
    - Mismatched keys between the checkpoint and the model.
    - Interpolating positional embeddings for different input sizes.
    - Renaming keys from a DINO-style projection head to a standard patch embedding.
    - Skipping SSL-specific keys (e.g., SimMIM head) when loading into a fine-tuning model.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the .pth or .pt checkpoint file.

    Returns:
        torch.nn.Module: The model with weights loaded.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Loading weights from: {checkpoint_path}")

    pretrained_checkpoint = torch.load(checkpoint_path, map_location="cpu")

    pretrained_state_dict = pretrained_checkpoint.get(
        "model_state_dict", pretrained_checkpoint
    )

    model_ft_state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in pretrained_state_dict.items():
        if k in model_ft_state_dict:
            if v.shape == model_ft_state_dict[k].shape:
                new_state_dict[k] = v
            else:
                logger.warning(f"Shape mismatch for '{k}': Pretrained {v.shape} vs Model {model_ft_state_dict[k].shape}")

        elif (
            k.startswith("projection.")
            and f"patch_embedding.{k}" in model_ft_state_dict
        ):
            new_key = f"patch_embedding.{k}"
            if v.shape == model_ft_state_dict[new_key].shape:
                new_state_dict[new_key] = v
                logger.info(f"Remapped key '{k}' to '{new_key}'")
            else:
                logger.warning(f"Shape mismatch for remapped key '{new_key}' (from '{k}')")

        elif (
            k == "positional_embedding"
            and "patch_embedding.positional_embedding" in model_ft_state_dict
        ):
            ft_pe = model_ft_state_dict["patch_embedding.positional_embedding"]

            if v.shape[1] == ft_pe.shape[1] - 1 and v.shape[2] == ft_pe.shape[2]:
                logger.info(f"Interpolating positional embedding for '{k}'...")
                new_pe = torch.zeros_like(ft_pe)
                new_pe[:, 1:, :] = v
                new_state_dict["patch_embedding.positional_embedding"] = new_pe
            else:
                logger.warning( f"Cannot interpolate positional_embedding: Pretrained {v.shape} vs Model {ft_pe.shape}")

        elif (
            "simmim_head" in k
            or "mask_token" in k
            or k.startswith("teacher.")
            or k.startswith("center")
        ):
            logger.info(f"Skipping SSL-specific key: {k}")

        else:
            logger.warning(f"Key '{k}' from checkpoint not found in the model.")

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"Successfully loaded weights.")
    logger.warning(f"Missing keys in model: {missing_keys}")
    logger.warning(f"Unexpected keys in model (from checkpoint but not used): {unexpected_keys}")
    return model


def freeze_backbone(model: ViT):
    """Freezes the backbone parameters of a Vision Transformer for fine-tuning."""
    logger.info("Freezing model backbone...")
    for param in model.encoder_blocks.parameters():
        param.requires_grad = False
    for name, param in model.patch_embedding.named_parameters():

        if "cls_token" not in name:
            param.requires_grad = False
    logger.info("Backbone frozen.")


def build_model(config):
    """
    Builds the appropriate model based on the configuration.
    This function acts as a single entry point for creating any model in the project.

    Args:
        config (OmegaConf): The global configuration object.

    Returns:
        torch.nn.Module: The constructed model, potentially with loaded weights.
    """

    mode = config.get("training", {}).get("type", None) or config.get("eval", {}).get(
        "mode", None
    )
    if mode is None:
        raise ValueError(
            "Could not determine mode. Set either 'training.type' or 'eval.mode' in config."
        )
    mode = mode.lower()

    image_shape = (
        config.model.in_channels,
        config.data.img_size,
        config.data.img_size,
    )

    logger.info(f"Building model for mode: '{mode}'")
    model = None

    if mode in ["supervised", "finetune"]:
        model = ViT(
            input_shape=image_shape,
            patch_size=config.model.patch_size,
            num_classes=config.model.num_classes,
            encoder_embed_dim=config.model.encoder_embed_dim,
            encoder_depth=config.model.encoder_depth,
            encoder_num_heads=config.model.encoder_num_heads,
            mlp_dim=config.model.mlp_dim,
            dropout=config.model.dropout,
        )
    elif mode == "simmim":
        model = SimMIMViT(
            input_shape=image_shape,
            patch_size=config.model.patch_size,
            encoder_embed_dim=config.model.encoder_embed_dim,
            encoder_depth=config.model.encoder_depth,
            encoder_num_heads=config.model.encoder_num_heads,
            mlp_dim=config.model.mlp_dim,
            dropout=config.model.dropout,
            mask_ratio=config.model.mask_ratio,
        )
    elif mode in ["dino", "eval_dino"]:
        model = DINOViT(
            input_shape=image_shape,
            patch_size=config.model.patch_size,
            encoder_embed_dim=config.model.encoder_embed_dim,
            encoder_depth=config.model.encoder_depth,
            encoder_num_heads=config.model.encoder_num_heads,
            mlp_dim=config.model.mlp_dim,
            dropout=config.model.dropout,
            output_dim=config.model.output_dim,
            center_momentum=config.model.center_momentum,
        )
    elif mode == "mae":
        model = MAEViT(
            input_shape=image_shape,
            patch_size=config.model.patch_size,
            encoder_embed_dim=config.model.encoder_embed_dim,
            decoder_embed_dim=config.model.decoder_embed_dim,
            encoder_depth=config.model.encoder_depth,
            decoder_depth=config.model.decoder_depth,
            encoder_num_heads=config.model.encoder_num_heads,
            decoder_num_heads =config.model.decoder_num_heads,
            mlp_dim=config.model.mlp_dim,
            dropout=config.model.dropout,
            mask_ratio=config.model.mask_ratio,
        ) 
    else:
        raise ValueError(f"Unknown model-building mode: {mode}")

    if mode == "finetune":
        checkpoint_path = config.training.pretrained_path
        model = load_weights(model, checkpoint_path)
        if config.training.freeze_backbone:
            freeze_backbone(model)
        _check_loaded_model(model, config)

    elif mode == "eval_dino":
        checkpoint_path = os.path.join(config.eval.experiment_path, "best_model.pth")
        model = load_weights(model, checkpoint_path)

    if hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def _check_loaded_model(model, config):
    """Check which parameters are frozen/trainable and verify loading."""
    logger.info("Checking loaded model")
    frozen = []
    trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    logger.info("Trainable parameters ({len(trainable)}):")
    for name in trainable:
        logger.info(f"[✓] {name}")

    logger.info(f"Frozen parameters ({len(frozen)}):")
    for name in frozen:
        logger.info(f"[-] {name}")

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
                    logger.warning(f"[!] Weight mismatch in: {name}")
                    mismatched += 1

        logger.info(f"Matched parameters from checkpoint: {matched}")
        logger.warning(f"Mismatched parameters: {mismatched}")
    logger.info("Model check complete")
