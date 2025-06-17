import os
import sys
import hydra
import torch

from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.model_builder import build_model
from data.data_builder import prepare_dataloaders
from utils.train_utils import get_transforms, setup_device

# TODO - make it work with different methods, not just DINO

def load_experiment_config(path: str):
    """Loads saved Hydra config and overrides from an experiment folder (using os.path)."""
    hydra_dir = os.path.join(path, ".hydra")
    config_path = os.path.join(hydra_dir, "config.yaml")
    overrides_path = os.path.join(hydra_dir, "overrides.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing: {config_path}")
    
    base_cfg = OmegaConf.load(config_path)

    if os.path.exists(overrides_path):
        overrides = OmegaConf.load(overrides_path)
        if overrides:
            base_cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(overrides))

    return base_cfg

def extract_features(model, dataloader, device):
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model.inference_forward(images)
            features.append(feats.cpu())
            labels.append(lbls)

    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels

@hydra.main(config_path="../../configs", config_name="eval_config", version_base=None)
def main(config): # TODO add schema?
    device = setup_device()
    experiment_config = load_experiment_config(config["eval"]["experiment_path"])
    
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, experiment_config)
    
    model = build_model(config).to(device)
    
    transforms = get_transforms(config['eval'])
    train_loader, val_loader = prepare_dataloaders(config, transforms, config['eval']['mode'])

    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)

    knn = KNeighborsClassifier(n_neighbors=config["eval"]["num_classes"], metric='cosine')
    knn.fit(train_features, train_labels)
    preds = knn.predict(val_features)
    accuracy = accuracy_score(val_labels, preds)
    print(f"Top-1 k-NN Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
