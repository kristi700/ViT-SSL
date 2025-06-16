import os
import sys
import hydra
import torch

from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, random_split, Subset

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.train_utils import get_transforms
from vit_core.ssl.dino.model import DINOViT
from data.datasets import (
    CIFAR10Dataset,
    STL10Dataset,
)
# TODO - make it work with different methods, not just DINO
# TODO - unify these model buildings + loadings into one ModelBuilder!

def setup_device():
    """Setup and return the appropriate device (CUDA/CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def _load_weights(model, config):
    """Load pretrained weights into the model."""
    pretrained_checkpoint = torch.load(os.path.join(config["eval"]["experiment_path"], "best_model.pth"), weights_only=False)
    pretrained_state_dict = pretrained_checkpoint["model_state_dict"]
    model_ft_state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k in model_ft_state_dict:
            if v.shape == model_ft_state_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(
                    f"Shape mismatch for {k}: Pretrained {v.shape} vs Fine-tune {model_ft_state_dict[k].shape}"
                )
        elif (
            k.startswith("projection.")
            and f"patch_embedding.{k}" in model_ft_state_dict
        ):
            new_key = f"patch_embedding.{k}"
            if v.shape == model_ft_state_dict[new_key].shape:
                new_state_dict[new_key] = v
            else:
                print(f"Shape mismatch for {new_key} (from {k})")
        elif k == "positional_embedding":
            if "patch_embedding.positional_embedding" in model_ft_state_dict:
                ft_pe = model_ft_state_dict["patch_embedding.positional_embedding"]
                if v.shape[1] == ft_pe.shape[1] - 1 and v.shape[2] == ft_pe.shape[2]:
                    print(f"Interpolating positional embedding for {k}...")
                    new_pe = torch.zeros_like(ft_pe)
                    new_pe[:, 1:, :] = v
                    new_state_dict["patch_embedding.positional_embedding"] = new_pe
                else:
                    print(
                        f"Cannot interpolate positional_embedding: Pretrained {v.shape} vs Fine-tune {ft_pe.shape}"
                    )
        else:
            print(f"Key {k} from pretrained checkpoint not found in fine-tuning model.")

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"\nMissing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    return model

def _freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

# TODO - add simmim in here later as well!
def build_model(config):
    mode = config["eval"]["mode"].lower()

    image_shape = (
        config["model"]["in_channels"],
        config["data"]["img_size"],
        config["data"]["img_size"],
    )
    if mode == "eval_dino":
        model =  DINOViT(
            input_shape=image_shape,
            patch_size=config["model"]["patch_size"],
            embed_dim=config["model"]["embed_dim"],
            num_blocks=config["model"]["num_blocks"],
            num_heads=config["model"]["num_heads"],
            mlp_dim=config["model"]["mlp_dim"],
            dropout=config["model"]["dropout"],
            output_dim=config["model"]["output_dim"],
            center_momentum=config["model"]["center_momentum"],
        )
        model = _load_weights(model, config) # TODO doublecheck
        model = _freeze_weights(model)
    return model

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
            feats = model._student_forward(images) # TODO - dont use private methods
            features.append(feats.cpu())
            labels.append(lbls)

    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels

def _get_dataset(config, transform):
    if config['eval']['dataset_name'].lower() == 'stl10':
        return STL10Dataset(
            config["eval"]["data_csv"], config["eval"]["data_dir"], transform
        )
    elif config['eval']['dataset_name'].lower() == 'stl10':
        return CIFAR10Dataset(
            config["eval"]["data_csv"], config["eval"]["data_dir"], transform
        )

def prepare_dataloaders(config, transforms):
    train_dataset_full = _get_dataset(config, transform=transforms["train"])
    val_dataset_full = _get_dataset(config, transform=transforms["val"])
    total_size = len(train_dataset_full)
    val_size = int(total_size * config["data"]["val_split"])
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(config["training"]["random_seed"])
    train_subset_indices, val_subset_indices = random_split(
        range(total_size), [train_size, val_size], generator=generator
    )

    train_dataset = Subset(train_dataset_full, train_subset_indices.indices)
    val_dataset = Subset(val_dataset_full, val_subset_indices.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader

@hydra.main(config_path="../../configs", config_name="eval_config", version_base=None)
def main(config): # TODO add schema?
    device = setup_device()
    experiment_config = load_experiment_config(config["eval"]["experiment_path"])
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, experiment_config)
    model = build_model(config).to(device)
    # TODO - instantiate dataloader
    transforms = get_transforms(config['eval']) # TODO match the loaded models input size!!!!
    train_loader, val_loader = prepare_dataloaders(config, transforms)

    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)

    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(train_features, train_labels)
    preds = knn.predict(val_features)
    accuracy = accuracy_score(val_labels, preds)
    print(f"Top-1 k-NN Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
