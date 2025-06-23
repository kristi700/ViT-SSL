import os
import sys
import hydra
import logging

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from utils.model_builder import build_model
from data.data_builder import prepare_dataloaders
from utils.schemas.eval_schemas import EvaluationConfig
from utils.train_utils import get_transforms, setup_device
from evaluators.unsupervised_evaluators.evaluator_utils import (
    extract_features,
    merge_with_experiment_config,
)
from evaluators.unsupervised_evaluators.umap_visualization import (
    prepare_combined_features,
    create_3d_umap_animation,
)

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="eval_config", version_base=None)
def main(config: EvaluationConfig):
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
    
    output_dir = config["eval"]["experiment_path"]
    os.makedirs(output_dir, exist_ok=True)
    create_3d_umap_animation(features, labels, output_dir)
    
    logger.info("UMAP visualization completed successfully!")


if __name__ == "__main__":
    main()