import os
import cv2
import yaml
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image

from vit_core.vit import ViT

# NOTE - for CIFAR10
IDX_TO_CLASS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


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
    return model


def load_model_for_eval(checkpoint_path, device):
    """Loads the model and config from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
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

        print(f"Model loaded from {checkpoint_path}")
        print(f" - Saved at Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f" - Best Val Acc: {checkpoint.get('best_val_acc', 'N/A')}")

        return model, config
    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {e}")


def get_inference_transforms(config):
    """Gets the transforms for inference (matching validation)."""
    img_size = config["data"]["img_size"]

    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((img_size, img_size)),
        ]
    )


def process_attention(attention_probs, image_size_hw, patch_size):
    """Processes attention probabilities for visualization."""
    if attention_probs is None:
        print("Warning: No attention probabilities found.")
        return None

    cls_attn = attention_probs[0, :, 0, 1:]
    cls_attn_avg = cls_attn.mean(dim=0)

    h, w = image_size_hw
    patch_h = h // patch_size
    patch_w = w // patch_size

    attn_grid = cls_attn_avg.reshape(patch_h, patch_w)
    attn_map_resized = cv2.resize(
        attn_grid.detach().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR
    )

    return attn_map_resized


def visualize(
    original_pil_img,
    attention_map,
    prediction_text,
    output_filename="attention_visualization.png",
    attention_alpha=0.5,
):
    """
    Displays and saves a combined visualization:
    Original image with attention map overlaid (with opacity),
    and the prediction text on top.
    """
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(original_pil_img)
    ax.imshow(
        attention_map,
        cmap="viridis",
        alpha=attention_alpha,
        interpolation="bilinear",
    )

    ax.text(
        0.5,
        0.95,
        prediction_text,
        color="white",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
        transform=ax.transAxes,
        bbox=dict(facecolor="black", alpha=0.6, pad=5, boxstyle="round,pad=0.5"),
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0.1)


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
    print(f"Using device: {device}")

    model, config = load_model_for_eval(args.checkpoint, device)
    if model is None:
        return

    img_pil = Image.open(args.image).convert("RGB")
    img_pil = img_pil.resize((config["data"]["img_size"], config["data"]["img_size"]))

    inference_transform = get_inference_transforms(config)
    img_tensor = inference_transform(img_pil).unsqueeze(0).to(device)

    last_attn_probs = None
    logits = None
    with torch.no_grad():
        output = model(img_tensor, return_attn=True)
        logits, last_attn_probs = output

    probabilities = F.softmax(logits, dim=-1)[0]
    top_prob, top_idx = torch.max(probabilities, dim=0)
    pred_class_idx = top_idx.item()
    pred_prob = top_prob.item()
    pred_class_name = IDX_TO_CLASS.get(
        pred_class_idx, f"Unknown Index {pred_class_idx}"
    )
    prediction_text = f"Predicted: {pred_class_name} ({pred_prob*100:.1f}%)"

    image_size_hw = img_pil.size[::-1]
    patch_size = config["model"]["patch_size"]
    attention_map = process_attention(last_attn_probs, image_size_hw, patch_size)
    visualize(img_pil, attention_map, prediction_text)


if __name__ == "__main__":
    main_test()
