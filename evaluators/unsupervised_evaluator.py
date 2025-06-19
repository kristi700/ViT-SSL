import os
import torch
import hydra
import logging
import pandas as pd

from typing import Optional
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from utils.train_utils import setup_device
from utils.model_builder import build_model
from data.data_builder import prepare_dataloaders
from utils.schemas.eval_schemas import EvaluationConfig
from evaluators.unsupervised_evaluators.evaluator_utils import (
    extract_features,
    merge_with_experiment_config,
)
from evaluators.unsupervised_evaluators.umap_visualization import (
    prepare_combined_features,
    run_umap_analysis,
)

logger = logging.getLogger(__name__)

def _default_transforms(img_size: int):
    from torchvision import transforms as T

    resize = T.Resize((img_size, img_size))
    return {
        "train": T.Compose([resize, T.ToTensor()]),
        "val": T.Compose([resize, T.ToTensor()]),
    }


def run_knn_evaluation(
    train_features, train_labels, val_features, val_labels, num_classes
):
    """
    Run k-NN evaluation.

    Args:
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        num_classes: Number of classes for k-NN

    Returns:
        dict: Results dictionary with accuracy and predictions
    """
    knn = KNeighborsClassifier(n_neighbors=num_classes, metric="cosine")
    knn.fit(train_features, train_labels)
    preds = knn.predict(val_features)
    accuracy = accuracy_score(val_labels, preds)

    logger.info(f"Top-1 k-NN Accuracy: {accuracy * 100:.2f}%")

    return {
        "method": "knn",
        "accuracy": accuracy,
        "predictions": preds,
        "num_neighbors": num_classes,
    }


def run_linear_evaluation(train_features, train_labels, val_features, val_labels):
    """
    Run linear probing evaluation.

    Args:
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels

    Returns:
        dict: Results dictionary with accuracy and predictions
    """
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(train_features, train_labels)
    preds = clf.predict(val_features)
    accuracy = accuracy_score(val_labels, preds)

    logger.info(f"Top-1 Linear Probing Accuracy: {accuracy * 100:.2f}%")

    return {"method": "linear", "accuracy": accuracy, "predictions": preds}


def run_multiple_evaluations(
    config, train_features, train_labels, val_features, val_labels, save_path
):
    """
    Run multiple evaluation modes.

    Args:
        config: Configuration object
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        eval_modes: List of evaluation modes to run. If None, uses config.eval.mode

    Returns:
        dict: Dictionary with results for each evaluation mode
    """
    eval_modes = config.eval.mode
    if not OmegaConf.is_list(eval_modes):
        eval_modes = [config.eval.mode] if config.eval.mode else []

    results = {}
    for mode in eval_modes:
        logger.info(f"Running evaluation mode: {mode}")

        if mode == "eval_knn":
            results[mode] = run_knn_evaluation(
                train_features,
                train_labels,
                val_features,
                val_labels,
                config.eval.num_classes,
            )

        elif mode == "eval_linear":
            results[mode] = run_linear_evaluation(
                train_features, train_labels, val_features, val_labels
            )

        elif mode == "eval_umap":

            features, labels = prepare_combined_features(
                train_features, train_labels, val_features, val_labels
            )

            embedding, metrics, quality, feedback = run_umap_analysis(
                features, labels, save_path
            )

            results[mode] = {
                "method": "umap",
                "embedding": embedding,
                "metrics": metrics,
                "quality": quality,
                "feedback": feedback,
            }

        else:
            logger.warning(f"Unknown evaluation mode '{mode}' - skipping")
            continue

    return results


def save_combined_results(results, output_path: str):
    """
    Save combined results from multiple evaluation modes.

    Args:
        results: Dictionary of results from run_multiple_evaluations
        output_path: Path to save the combined results
    """
    summary_data = []

    for mode, result in results.items():
        if result["method"] in ["knn", "linear"]:
            summary_data.append(
                {
                    "Evaluation_Mode": mode,
                    "Method": result["method"].upper(),
                    "Accuracy": f"{result['accuracy']*100:.2f}%",
                    "Additional_Info": f"k={result.get('num_neighbors', 'N/A')}"
                    if result["method"] == "knn"
                    else "Logistic Regression",
                }
            )
        elif result["method"] == "umap":
            summary_data.append(
                {
                    "Evaluation_Mode": mode,
                    "Method": "UMAP",
                    "Quality": result["quality"],
                    "Additional_Info": f"Silhouette: {result['metrics']['silhouette_features']:.3f}",
                }
            )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            os.path.join(output_path, "evaluation_summary.csv"), index=False
        )

        with open(os.path.join(output_path, "evaluation_summary.txt"), "w") as f:
            f.write("Multi-Evaluation Summary Report\n")
            f.write("=" * 40 + "\n\n")

            for mode, result in results.items():
                f.write(f"{mode.upper()}:\n")
                f.write("-" * 20 + "\n")

                if result["method"] in ["knn", "linear"]:
                    f.write(f"  Method: {result['method'].upper()}\n")
                    f.write(f"  Accuracy: {result['accuracy']*100:.2f}%\n")
                    if result["method"] == "knn":
                        f.write(f"  Number of neighbors: {result['num_neighbors']}\n")
                elif result["method"] == "umap":
                    f.write(f"  Method: UMAP\n")
                    f.write(f"  Quality: {result['quality']}\n")
                    f.write(
                        f"  Silhouette Score: {result['metrics']['silhouette_features']:.4f}\n"
                    )

                f.write("\n")

        logger.info(f"Combined results saved to {output_path}")


def run_evaluation(
    config,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
):
    """Run unsupervised evaluation based on ``config.eval.mode``.

    Features are extracted only once and reused for the selected evaluation.
    """
    device = device or setup_device()

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    if "experiment_path" in config.get("eval", {}):
        config = merge_with_experiment_config(config)

    if model is None:
        model = build_model(config).to(device)

    transforms = _default_transforms(config.data.img_size)
    train_loader, val_loader = prepare_dataloaders(config, transforms, config.eval.mode)

    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)

    results = run_multiple_evaluations(
        config,
        train_features,
        train_labels,
        val_features,
        val_labels,
        config["eval"].get("experiment_path", save_path),
    )
    # NOTE - rethink maybemaybe
    save_combined_results(results, config["eval"].get("experiment_path", save_path))


@hydra.main(config_path="../configs", config_name="eval_config", version_base=None)
def main(cfg: EvaluationConfig):
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
