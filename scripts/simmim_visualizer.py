import os
import yaml
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

from PIL import Image

from vit_core.ssl.simmim import SimMIMViT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log")
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):
    """Instantiates the ViT model from configuration."""
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


def load_model_for_eval(checkpoint_path, device):
    """Loads the model and config from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found at {checkpoint_path}")
        return None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]

        model = build_model(config)
        if model is None:
            return None, None

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info(f"Model loaded from {checkpoint_path}")
        logger.info(f"Saved at Epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"Best Val Acc: {checkpoint.get('best_val_acc', 'N/A')}")

        return model, config
    except Exception as e:
        logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")

def get_inference_transforms(config):
    """Gets the transforms for inference (matching validation)."""
    img_size = config["data"]["img_size"]

    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((img_size, img_size)),
        ]
    )


def visualize_simmim_reconstruction(
    original_image_tensor: torch.Tensor,
    bool_mask_img: torch.Tensor,
    predicted_pixels_for_img: torch.Tensor,
    target_pixels_for_img: torch.Tensor,
    config: dict,
    output_filename: str = "simmim_reconstruction.png",
    save_dir: str = ".",
):
    patch_size = config["model"]["patch_size"]
    in_channels = config["model"]["in_channels"]
    img_h, img_w = original_image_tensor.shape[1], original_image_tensor.shape[2]

    num_patches_h = img_h // patch_size
    num_patches_w = img_w // patch_size

    display_original_img_tensor = original_image_tensor.cpu()
    bool_mask_img = bool_mask_img.cpu()
    predicted_pixels_for_img = predicted_pixels_for_img.cpu().float()
    target_pixels_for_img = target_pixels_for_img.cpu().float()

    original_img_hwc = display_original_img_tensor.permute(1, 2, 0).numpy()

    masked_input_display = display_original_img_tensor.clone()
    reconstructed_img_chw = display_original_img_tensor.clone()

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    patch_idx_mask = 0
    for r_patch in range(num_patches_h):
        for c_patch in range(num_patches_w):
            patch_linear_idx = r_patch * num_patches_w + c_patch
            h_start, w_start = r_patch * patch_size, c_patch * patch_size
            h_end, w_end = h_start + patch_size, w_start + patch_size

            if bool_mask_img[patch_linear_idx]:
                masked_input_display[:, h_start:h_end, w_start:w_end] = 0.5

                pred_patch_flat = predicted_pixels_for_img[patch_idx_mask]
                pred_patch_chw = pred_patch_flat.reshape(
                    in_channels, patch_size, patch_size
                )
                reconstructed_img_chw[:, h_start:h_end, w_start:w_end] = torch.clamp(
                    pred_patch_chw, 0, 1
                )
                patch_idx_mask += 1

    axs[0].imshow(np.clip(original_img_hwc, 0, 1))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(np.clip(masked_input_display.permute(1, 2, 0).numpy(), 0, 1))
    axs[1].set_title("Masked Input")
    axs[1].axis("off")

    axs[2].imshow(np.clip(reconstructed_img_chw.permute(1, 2, 0).numpy(), 0, 1))
    axs[2].set_title("Model Reconstruction")
    axs[2].axis("off")

    fig.suptitle(f"SimMIM Reconstruction", fontsize=16)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        full_output_path = os.path.join(save_dir, output_filename)
        plt.savefig(full_output_path)
        logger.info(f"Reconstruction visualization saved to {full_output_path}")


def main_test():
    parser = argparse.ArgumentParser(
        description="Test ViT Model and Visualize Attention"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file (.pth)",
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image file"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, config = load_model_for_eval(args.checkpoint, device)
    if model is None:
        return

    img_pil = Image.open(args.image).convert("RGB")
    img_pil = img_pil.resize((config["data"]["img_size"], config["data"]["img_size"]))

    inference_transform = get_inference_transforms(config)
    img_tensor = inference_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output, targets, bool_mask = model(img_tensor, return_bool_mask=True)

    visualize_simmim_reconstruction(
        img_tensor.squeeze(0), bool_mask.squeeze(0), output, targets, config
    )


if __name__ == "__main__":
    main_test()
