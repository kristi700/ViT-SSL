import os
import torch
import argparse
import torchvision.transforms as transforms

from tqdm import tqdm
from datetime import datetime
from data.datasets import CIFAR10Dataset, STL10Dataset
from torch.utils.data import DataLoader, random_split, Subset

from vit_core.vit import ViT
from utils.config_parser import load_config

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT Model')
    parser.add_argument('--config', type=str, default='configs/stl10.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    return args

def get_transforms(config):
    img_size = config['data']['img_size']

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return train_transform, val_transform

def prepare_dataloaders(config, train_transform, val_transform):
    # TODO - to separate func if more datasets
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
    if config['training']['type'].lower() == 'supervised':
        return model
    elif config['training']['type'].lower() == 'finetune':
        return load_pretrained_model(config, model)
    else:
        raise KeyError('Not supported training type for train_supervised.py')
    
def load_pretrained_model(config, model: ViT) -> ViT:
    pretrained_checkpoint = torch.load(config['training']['pretrained_path'])
    pretrained_state_dict = pretrained_checkpoint['model_state_dict']
    model_ft_state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k in model_ft_state_dict:
            if v.shape == model_ft_state_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(f"Shape mismatch for {k}: Pretrained {v.shape} vs Fine-tune {model_ft_state_dict[k].shape}")
        elif k.startswith("projection.") and f"patch_embedding.{k}" in model_ft_state_dict:
            new_key = f"patch_embedding.{k}"
            if v.shape == model_ft_state_dict[new_key].shape:
                new_state_dict[new_key] = v
            else:
                print(f"Shape mismatch for {new_key} (from {k})")

        elif k == "positional_embedding": 
            if "patch_embedding.positional_embedding" in model_ft_state_dict:
                ft_pe = model_ft_state_dict["patch_embedding.positional_embedding"] # Shape (1, N+1, D)
                if v.shape[1] == ft_pe.shape[1] - 1 and v.shape[2] == ft_pe.shape[2]:
                    print(f"Interpolating positional embedding for {k}...")
                    new_pe = torch.zeros_like(ft_pe)
                    new_pe[:, 1:, :] = v 
                    new_state_dict["patch_embedding.positional_embedding"] = new_pe
                else:
                    print(f"Cannot interpolate positional_embedding: Pretrained {v.shape} vs Fine-tune {ft_pe.shape}")
        elif "simmim_head" in k or "mask_token" in k:
            print(f"Skipping SimMIM specific key: {k}")
        else:
            print(f"Key {k} from pretrained checkpoint not found in fine-tuning model.")
    model.load_state_dict(new_state_dict, strict=False)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_desc="Training", scheduler=None, warmup_scheduler=None, current_epoch=None, warmup_epochs=0):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc=epoch_desc, leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        if warmup_scheduler is not None and current_epoch <= warmup_epochs:
             warmup_scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, target_lr, start_lr):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = max(1, warmup_steps)
        self.target_lr = target_lr
        self.start_lr = start_lr
        self.lr_steps = [
            (target_lr - start_lr) / self.warmup_steps
            for _ in optimizer.param_groups
        ]

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            lr_scale = float(self._step) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.start_lr + lr_scale * (self.target_lr - self.start_lr)

def main():
    args = parse_args()
    config = load_config(args.config)
    device = setup_device()
    
    train_transform, val_transform = get_transforms(config)
    train_loader, val_loader = prepare_dataloaders(config, train_transform, val_transform)
    model = build_model(config).to(device)

    warmup_initial_lr = float(config['training']['warmup_initial_learning_rate'])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=warmup_initial_lr, weight_decay=config['training']['weight_decay'])

    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training']['warmup_epochs']
    warmup_steps = warmup_epochs * len(train_loader)

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=float(config['training']['lr_final'])
    )

    warmup_scheduler = LinearWarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        target_lr=float(config['training']['warmup_final_learning_rate']),
        start_lr = warmup_initial_lr
    )

    best_val_acc = 0.0
    save_path = os.path.join(config['training']['checkpoint_dir'], config['training']['type'], str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    for epoch in range(1, config['training']['num_epochs'] + 1):
        epoch_desc = f"Epoch {epoch}/{config['training']['num_epochs']}"
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch_desc,
            scheduler=main_scheduler,
            warmup_scheduler=warmup_scheduler,
            current_epoch=epoch,
            warmup_epochs=warmup_epochs
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if epoch > warmup_epochs:
            main_scheduler.step()

        print(f"\nEpoch {epoch} Summary: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}. Saving model...")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }
            os.makedirs(save_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))

    print("Training finished.")

if __name__ == "__main__":
    main()