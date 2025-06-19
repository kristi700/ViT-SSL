"""
NOTE - This script is quite redundant /w evaluators/umap_vizualization.py (though this one is much simpler) as this is only intended for flashy visualization, 
not for training auto evaluation.
"""

import os
import sys
import hydra
import torch
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from umap import UMAP
from PIL import Image

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from utils.model_builder import build_model
from data.data_builder import prepare_dataloaders
from utils.schemas.eval_schemas import EvaluationConfig
from utils.train_utils import get_transforms, setup_device
from evaluators.unsupervised_evaluators.evaluator_utils import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log")
    ]
)
logger = logging.getLogger(__name__)

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

def run_umap(features, labels, output_dir):
    """
    Run UMAP and create 3D visualization with rotating animation.
    
    Args:
        features: Feature vectors
        labels: Corresponding labels
        output_dir: Output directory for saving files
        
    Returns:
        tuple: (embedding, metrics, quality, feedback) - for compatibility with main function
    """
    logger.info("3D UMAP fitting...")
    reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean', verbose=True)
    embedding_3d = reducer.fit_transform(features)

    frames_dir = os.path.join(output_dir, "umap_frames")
    os.makedirs(frames_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    logger.info("Creating 3D UMAP visualization frames...")
    frame_files = []
    
    for angle in range(0, 360, 4):
        ax.clear()
        scatter = ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2],
                           c=labels, cmap='Spectral', s=5, alpha=0.7)
        ax.view_init(elev=20, azim=angle)
        ax.set_title(f"3D UMAP Visualization - Rotation {angle}°", fontsize=14)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        
        if angle == 0:
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        
        frame_file = os.path.join(frames_dir, f"frame_{angle:03d}.png")
        plt.savefig(frame_file, dpi=100, bbox_inches='tight')
        frame_files.append(frame_file)
    
    plt.close(fig)
    
    logger.info("Creating animated GIF...")
    if frame_files:
        images = []
        for frame_file in frame_files:
            if os.path.exists(frame_file):
                images.append(Image.open(frame_file))
        
        if images:
            gif_path = os.path.join(output_dir, "umap_3d_rotation.gif")
            images[0].save(
                gif_path, 
                save_all=True, 
                append_images=images[1:], 
                duration=100, 
                loop=0
            )
            logger.info(f"3D UMAP animated visualization saved to: {gif_path}")
            
            for frame_file in frame_files:
                try:
                    os.remove(frame_file)
                except OSError:
                    pass
            
            try:
                os.rmdir(frames_dir)
            except OSError:
                pass
        else:
            logger.error("No valid frames created for GIF")
    else:
        logger.error("No frame files created")
    logger.info("3D UMAP visualization finished")

@hydra.main(config_path="../configs", config_name="eval_config", version_base=None)
def main(config: EvaluationConfig):
    try:
        device = setup_device()
        config = merge_with_experiment_config(config)

        model = build_model(config).to(device)

        transforms = get_transforms(config["eval"])
        train_loader, val_loader = prepare_dataloaders(
            config, transforms, config["eval"]["mode"]
        )

        logger.info("Extracting features from training data...")
        train_features, train_labels = extract_features(model, train_loader, device)
        
        logger.info("Extracting features from validation data...")
        val_features, val_labels = extract_features(model, val_loader, device)

        logger.info("Combining features...")
        features, labels = prepare_combined_features(
            train_features, train_labels, val_features, val_labels
        )

        logger.info(f"Total samples: {len(features)}, Feature dimension: {features.shape[1]}")
        
        run_umap(
            features, labels, config["eval"]["experiment_path"]
        )
        
        logger.info("UMAP visualization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()