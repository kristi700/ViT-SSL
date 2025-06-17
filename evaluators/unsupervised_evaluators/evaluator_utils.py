import os
import torch

from tqdm import tqdm
from omegaconf import OmegaConf

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

def _load_experiment_config(path: str):
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

def merge_with_experiment_config(config):
    exp_cfg = _load_experiment_config(config["eval"]["experiment_path"])
    OmegaConf.set_struct(config, False)
    return OmegaConf.merge(config, exp_cfg)