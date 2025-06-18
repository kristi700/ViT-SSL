import os
import sys
import hydra

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from evaluator_utils import *
from utils.model_builder import build_model
from data.data_builder import prepare_dataloaders
from utils.schemas.eval_schemas import EvaluationConfig
from utils.train_utils import get_transforms, setup_device


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

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(train_features, train_labels)
    preds = clf.predict(val_features)
    accuracy = accuracy_score(val_labels, preds)
    print(f"Top-1 Linear Probing Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
