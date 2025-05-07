import torch
import argparse
import torchvision.transforms as transforms


from torch.utils.data import DataLoader, random_split, Subset
from data.datasets import CIFAR10Dataset, STL10Dataset, STL10UnsupervisedDataset
from vit_core.vit import ViT
from utils.config_parser import load_config

# NOTE - will need refactoring (alongside /w supervised_train), for testing purposes as of rightnow!

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT Model')
    parser.add_argument('--config', type=str, default='configs/stl10_unsupervised.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    return args

def get_transforms(config):
    img_size = config['data']['img_size']

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return train_transform, val_transform

def prepare_dataloaders(config, train_transform, val_transform):
    # TODO - to separate func!
    if config['training']['type'].lower() == 'supervised':
        if config['data']['dataset_name'].lower() == 'cifar10':
            full_dataset_train_transforms = CIFAR10Dataset(
                config['data']['data_csv'], config['data']['data_dir'], transform=train_transform
            )
            full_dataset_val_transforms = CIFAR10Dataset(
                config['data']['data_csv'], config['data']['data_dir'], transform=val_transform
            )
        elif config['data']['dataset_name'].lower() == 'stl10':
            full_dataset_train_transforms = STL10Dataset(
                config['data']['data_csv'], config['data']['data_dir'], transform=train_transform
            )
            full_dataset_val_transforms = STL10Dataset(
                config['data']['data_csv'], config['data']['data_dir'], transform=val_transform
            )
    elif config['training']['type'].lower() == 'unsupervised':
            if config['data']['dataset_name'].lower() == 'stl10':
                full_dataset_train_transforms = STL10UnsupervisedDataset(
                    config['data']['data_dir'], transform=train_transform
                )
                full_dataset_val_transforms = STL10UnsupervisedDataset(
                    config['data']['data_dir'], transform=val_transform
                )

    total_size = len(full_dataset_train_transforms)
    val_size = int(total_size * config['data']['val_split'])
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(config['training']['random_seed'])
    train_subset_indices, val_subset_indices = random_split(
        range(total_size), [train_size, val_size], generator=generator)

    train_dataset = Subset(full_dataset_train_transforms, train_subset_indices.indices)
    val_dataset = Subset(full_dataset_val_transforms, val_subset_indices.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    return train_loader, val_loader

def build_model(config):
    image_shape = (config['model']['in_channels'], config['data']['img_size'], config['data']['img_size'])
    model = ViT(
        input_shape=image_shape,
        patch_size=config['model']['patch_size'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        num_blocks=config['model']['num_blocks'],
        num_heads=config['model']['num_heads'],
        mlp_dim=config['model']['mlp_dim'],
        dropout=config['model']['dropout'],
    )
    return model

def main():
    args = parse_args()
    config = load_config(args.config)
    device = setup_device()
    
    train_transform, val_transform = get_transforms(config)
    train_loader, val_loader = prepare_dataloaders(config, train_transform, val_transform)
    model = build_model(config).to(device)

    for i, image in enumerate(train_loader):
        print(image.shape)

if __name__ == "__main__":
    main()