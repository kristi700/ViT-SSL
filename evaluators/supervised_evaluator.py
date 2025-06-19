import os
import hydra
import torch
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix

from utils.metrics import Accuracy
from utils.model_builder import build_model
from data.data_builder import prepare_dataloaders
from utils.train_utils import get_transforms, setup_device
from evaluators.unsupervised_evaluators.evaluator_utils import merge_with_experiment_config

logger = logging.getLogger(__name__)

def load_model_for_eval(config, device: torch.device) -> Tuple[torch.nn.Module, OmegaConf]:
    """Load a model and its config from a checkpoint."""
    checkpoint_path = os.path.join(config['eval']['experiment_path'], "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = merge_with_experiment_config(config)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint '{checkpoint_path}'")
    return model, config


def evaluate(model: torch.nn.Module, dataloader, device: torch.device):
    """Run inference over a dataloader and compute accuracy."""
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    acc = Accuracy().compute(correct=correct, total=total)
    preds_tensor = torch.cat(all_preds)
    labels_tensor = torch.cat(all_labels)
    return acc, preds_tensor, labels_tensor


def save_results(acc: float, preds: torch.Tensor, labels: torch.Tensor, output_dir: str, save_cm: bool):
    """Save accuracy, predictions and confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({"label": labels.tolist(), "prediction": preds.tolist()})
    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    results = {"top1_accuracy": acc}
    if save_cm:
        cm = confusion_matrix(labels.numpy(), preds.numpy())
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        heatmap_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(heatmap_path)
        plt.close()

        results["confusion_matrix_image"] = heatmap_path

    logger.info(f"Top-1 Accuracy: {acc*100:.2f}%")
    logger.info(f"Results saved to {output_dir}")


@hydra.main(config_path="../configs", config_name="supervised_eval", version_base=None)
def main(config): # TODO schema
    device = setup_device()

    model, config = load_model_for_eval(config, device)

    transforms = get_transforms(config)
    _, val_loader = prepare_dataloaders(config, transforms, "supervised")

    acc, preds, labels = evaluate(model, val_loader, device)
    save_results(
        acc,
        preds,
        labels,
        config['eval']['experiment_path'],
        config.eval.get("confusion_matrix", False),
    )


if __name__ == "__main__":
    main()