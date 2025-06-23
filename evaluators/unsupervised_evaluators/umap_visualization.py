import os
import sys
import hydra
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from evaluators.unsupervised_evaluators.evaluator_utils import *
from utils.model_builder import build_model
from data.data_builder import prepare_dataloaders
from utils.schemas.eval_schemas import EvaluationConfig
from utils.train_utils import get_transforms, setup_device

logger = logging.getLogger(__name__)

def evaluate_feature_quality(features, labels, embedding, sample_size=2000):
    """Compute various metrics to assess feature quality - optimized for large datasets"""

    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    logger.info( f"Evaluating features: {features.shape[0]} samples, {features.shape[1]} dimensions")
    if len(features) > sample_size:

        logger.info(f"Sampling {sample_size} points for expensive computations...")
        from sklearn.model_selection import train_test_split

        _, sampled_features, _, sampled_labels = train_test_split(
            features, labels, test_size=sample_size, stratify=labels, random_state=42
        )
    else:
        sampled_features = features
        sampled_labels = labels

    sil_score_umap = silhouette_score(embedding, labels)

    logger.info("Computing silhouette score on features...")
    sil_score_features = silhouette_score(sampled_features, sampled_labels)

    logger.info("Running K-means clustering...")
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=100)
    kmeans_labels = kmeans.fit_predict(sampled_features)
    ari_score = adjusted_rand_score(sampled_labels, kmeans_labels)

    logger.info("Computing class separation metrics...")
    unique_labels = np.unique(labels)

    class_centers = {}
    intra_distances = []

    for label in unique_labels:
        class_mask = labels == label
        class_features = features[class_mask]
        class_center = class_features.mean(0)
        class_centers[label] = class_center

        if len(class_features) > 1:

            if len(class_features) > 500:
                sample_indices = np.random.choice(
                    len(class_features), 500, replace=False
                )
                class_sample = class_features[sample_indices]
            else:
                class_sample = class_features

            distances_to_center = np.linalg.norm(class_sample - class_center, axis=1)
            intra_distances.append(np.mean(distances_to_center))

    inter_distances = []
    centers_list = list(class_centers.values())
    for i in range(len(centers_list)):
        for j in range(i + 1, len(centers_list)):
            inter_dist = np.linalg.norm(centers_list[i] - centers_list[j])
            inter_distances.append(inter_dist)

    avg_intra = np.mean(intra_distances) if intra_distances else 0
    avg_inter = np.mean(inter_distances) if inter_distances else 0
    separation_ratio = avg_inter / avg_intra if avg_intra > 0 else 0

    return {
        "silhouette_features": sil_score_features,
        "silhouette_umap": sil_score_umap,
        "adjusted_rand_index": ari_score,
        "avg_intra_distance": avg_intra,
        "avg_inter_distance": avg_inter,
        "separation_ratio": separation_ratio,
        "n_samples": len(features),
        "n_features": features.shape[1],
        "n_classes": n_clusters,
        "sampled_for_computation": len(features) > sample_size,
    }


def assess_quality(metrics):
    """Provide quality assessment based on metrics"""
    sil_score = metrics["silhouette_features"]
    sep_ratio = metrics["separation_ratio"]
    ari_score = metrics["adjusted_rand_index"]

    score = 0
    feedback = []

    if sil_score > 0.7:
        score += 3
        feedback.append("Excellent cluster cohesion")
    elif sil_score > 0.5:
        score += 2
        feedback.append("Good cluster cohesion")
    elif sil_score > 0.2:
        score += 1
        feedback.append("Fair cluster cohesion")
    else:
        feedback.append("Poor cluster cohesion")

    if sep_ratio > 3:
        score += 3
        feedback.append("Excellent class separation")
    elif sep_ratio > 2:
        score += 2
        feedback.append("Good class separation")
    elif sep_ratio > 1.5:
        score += 1
        feedback.append("Fair class separation")
    else:
        feedback.append("Poor class separation")

    if ari_score > 0.8:
        score += 3
        feedback.append("Excellent clustering agreement")
    elif ari_score > 0.6:
        score += 2
        feedback.append("Good clustering agreement")
    elif ari_score > 0.4:
        score += 1
        feedback.append("Fair clustering agreement")
    else:
        feedback.append("Poor clustering agreement")

    if score >= 7:
        quality = "Excellent"
    elif score >= 5:
        quality = "Good"
    elif score >= 3:
        quality = "Fair"
    else:
        quality = "Poor"

    return quality, feedback


def create_comprehensive_umap_analysis(embedding, labels, features, save_path):
    """Create multiple visualizations for comprehensive analysis"""

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    scatter1 = axes[0, 0].scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=15, alpha=0.7
    )
    axes[0, 0].set_title("UMAP Projection by True Labels")
    axes[0, 0].set_xlabel("UMAP 1")
    axes[0, 0].set_ylabel("UMAP 2")
    plt.colorbar(scatter1, ax=axes[0, 0])

    axes[0, 1].hexbin(embedding[:, 0], embedding[:, 1], gridsize=30, cmap="Blues")
    axes[0, 1].set_title("UMAP Density Plot")
    axes[0, 1].set_xlabel("UMAP 1")
    axes[0, 1].set_ylabel("UMAP 2")

    n_clusters = len(np.unique(labels))
    kmeans_umap = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    umap_clusters = kmeans_umap.fit_predict(embedding)
    scatter3 = axes[1, 0].scatter(
        embedding[:, 0], embedding[:, 1], c=umap_clusters, cmap="tab10", s=15, alpha=0.7
    )
    axes[1, 0].set_title("K-means Clusters in UMAP Space")
    axes[1, 0].set_xlabel("UMAP 1")
    axes[1, 0].set_ylabel("UMAP 2")
    plt.colorbar(scatter3, ax=axes[1, 0])

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        axes[1, 1].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[i]],
            label=f"Class {label}",
            s=15,
            alpha=0.7,
        )

    axes[1, 1].set_title("UMAP Projection with Class Labels")
    axes[1, 1].set_xlabel("UMAP 1")
    axes[1, 1].set_ylabel("UMAP 2")
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_umap_analysis(features, labels, output_dir, umap_params=None):
    """
    Run UMAP analysis on features and generate visualizations and quality reports.

    Args:
        features (torch.Tensor): Feature tensor
        labels (torch.Tensor): Label tensor
        output_dir (str): Directory to save outputs
        umap_params (dict, optional): UMAP parameters. Defaults to standard params.

    Returns:
        tuple: (embedding, metrics, quality, feedback)
    """

    if umap_params is None:
        umap_params = {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "verbose": True,
        }

    logger.info(f"Starting UMAP on {features.shape[0]} samples with {features.shape[1]} dimensions")
    reducer = UMAP(**umap_params)
    embedding = reducer.fit_transform(features)

    create_basic_umap_plot(embedding, labels, output_dir)

    logger.info("Starting feature quality evaluation...")
    metrics = evaluate_feature_quality(features, labels, embedding, sample_size=2000)
    quality, feedback = assess_quality(metrics)

    create_comprehensive_umap_analysis(
        embedding,
        labels,
        features,
        os.path.join(output_dir, "comprehensive_umap_analysis.png"),
    )

    save_umap_results(metrics, quality, feedback, output_dir)

    logger.info("Analysis complete!")
    return embedding, metrics, quality, feedback


def create_basic_umap_plot(embedding, labels, output_dir):
    """
    Create and save basic UMAP scatter plot.

    Args:
        embedding (numpy.ndarray): UMAP embedding coordinates
        labels (torch.Tensor): Labels for coloring
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="Spectral", s=5)
    plt.colorbar()
    plt.title("UMAP projection of learned features")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(
        os.path.join(output_dir, "umap_visualization.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def save_umap_results(metrics, quality, feedback, output_dir):
    """
    Save UMAP analysis results to CSV and text files.

    Args:
        metrics (dict): Quality metrics dictionary
        quality (str): Overall quality assessment
        feedback (list): List of feedback strings
        output_dir (str): Directory to save results
    """

    results_data = {
        "Metric": [
            "Overall Quality",
            "Silhouette Score (Features)",
            "Silhouette Score (UMAP)",
            "Adjusted Rand Index",
            "Average Intra-class Distance",
            "Average Inter-class Distance",
            "Separation Ratio",
            "Number of Samples",
            "Number of Features",
            "Number of Classes",
        ],
        "Value": [
            quality,
            f"{metrics['silhouette_features']:.4f}",
            f"{metrics['silhouette_umap']:.4f}",
            f"{metrics['adjusted_rand_index']:.4f}",
            f"{metrics['avg_intra_distance']:.4f}",
            f"{metrics['avg_inter_distance']:.4f}",
            f"{metrics['separation_ratio']:.4f}",
            metrics["n_samples"],
            metrics["n_features"],
            metrics["n_classes"],
        ],
        "Interpretation": [
            f"Features are {quality.lower()} quality",
            "Higher is better (max: 1.0)",
            "Higher is better (max: 1.0)",
            "Higher is better (max: 1.0)",
            "Distance within classes (lower is better)",
            "Distance between classes (higher is better)",
            "Inter/Intra ratio (higher is better)",
            "Total data points analyzed",
            "Feature dimensionality",
            "Number of unique classes",
        ],
    }

    for i, fb in enumerate(feedback):
        results_data["Metric"].append(f"Quality Indicator {i+1}")
        results_data["Value"].append("✓")
        results_data["Interpretation"].append(fb)

    if metrics.get("sampled_for_computation", False):
        results_data["Metric"].append("Computation Method")
        results_data["Value"].append("Sampled (2000 points)")
        results_data["Interpretation"].append(
            "Large dataset - used sampling for expensive computations"
        )

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(
        os.path.join(output_dir, "umap_feature_quality_results.csv"),
        index=False,
    )

    with open(
        os.path.join(output_dir, "umap_feature_quality_report.txt"),
        "w",
    ) as f:
        f.write("UMAP Feature Quality Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Overall Assessment: {quality}\n\n")
        f.write("Detailed Metrics:\n")
        f.write("-" * 20 + "\n")
        for _, row in results_df.iterrows():
            if "Quality Indicator" not in row["Metric"]:
                f.write(f"{row['Metric']}: {row['Value']}\n")
                f.write(f"  → {row['Interpretation']}\n\n")

        f.write("Quality Indicators:\n")
        f.write("-" * 20 + "\n")
        for fb in feedback:
            f.write(f"• {fb}\n")

def create_3d_umap_animation(features, labels, output_dir, umap_params=None):
    """Create a rotating 3D UMAP visualization saved as GIF."""

    if umap_params is None:
        umap_params = {
            "n_components": 3,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "verbose": True,
        }

    logger.info("Fitting 3D UMAP embedding...")
    reducer = UMAP(**umap_params)
    embedding = reducer.fit_transform(features)

    frames_dir = os.path.join(output_dir, "umap_frames")
    os.makedirs(frames_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    frame_files = []
    for angle in range(0, 360, 4):
        ax.clear()
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=labels,
            cmap="Spectral",
            s=5,
            alpha=0.7,
        )
        ax.view_init(elev=20, azim=angle)
        ax.set_title(f"3D UMAP - Rotation {angle}°")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        if angle == 0:
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        frame_file = os.path.join(frames_dir, f"frame_{angle:03d}.png")
        plt.savefig(frame_file, dpi=100, bbox_inches="tight")
        frame_files.append(frame_file)

    plt.close(fig)

    if frame_files:
        images = [Image.open(f) for f in frame_files if os.path.exists(f)]
        if images:
            gif_path = os.path.join(output_dir, "umap_3d_rotation.gif")
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0,
            )
            logger.info(f"3D UMAP animation saved to: {gif_path}")
        for f in frame_files:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(frames_dir)
        except OSError:
            pass

    return embedding


def prepare_combined_features(train_features, train_labels, val_features, val_labels):
    """
    Combine training and validation features and labels.

    Args:
        train_features (torch.Tensor): Training features
        train_labels (torch.Tensor): Training labels
        val_features (torch.Tensor): Validation features
        val_labels (torch.Tensor): Validation labels

    Returns:
        tuple: (combined_features, combined_labels)
    """
    features = torch.cat((train_features, val_features))
    labels = torch.cat((train_labels, val_labels))
    return features, labels


@hydra.main(config_path="../../configs", config_name="eval_config", version_base=None)
def main(config: EvaluationConfig):
    device = setup_device()
    config = merge_with_experiment_config(config)

    model = build_model(config).to(device)

    transforms = get_transforms(config["eval"])
    train_loader, val_loader = prepare_dataloaders(
        config, transforms, config["eval"]["mode"]
    )

    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)

    features, labels = prepare_combined_features(
        train_features, train_labels, val_features, val_labels
    )

    embedding, metrics, quality, feedback = run_umap_analysis(
        features, labels, config["eval"]["experiment_path"]
    )


if __name__ == "__main__":
    main()
