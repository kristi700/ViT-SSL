import os
import sys
import hydra
import logging

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from evaluator_utils import *
from utils.model_builder import build_model
from data.data_builder import prepare_dataloaders
from utils.schemas.eval_schemas import EvaluationConfig
from utils.train_utils import get_transforms, setup_device

logger = logging.getLogger(__name__)

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

    knn = KNeighborsClassifier(
        n_neighbors=config["eval"]["num_classes"], metric="cosine"
    )
    knn.fit(train_features, train_labels)
    preds = knn.predict(val_features)
    accuracy = accuracy_score(val_labels, preds)
    logger.info(f"Top-1 k-NN Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
