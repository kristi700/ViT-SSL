import os
import math
import torch
import argparse
import torchvision.transforms as transforms

from tqdm import tqdm
from datetime import datetime
from ignite.metrics import SSIM
from torcheval.metrics import PeakSignalNoiseRatio


from utils.config_parser import load_config
from vit_core.ssl.simmim.model import SimMIMViT
from torch.utils.data import DataLoader, random_split, Subset
from data.datasets import CIFAR10Dataset, STL10Dataset, STL10UnsupervisedDataset

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
    model = SimMIMViT(
        input_shape=image_shape,
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        num_blocks=config['model']['num_blocks'],
        num_heads=config['model']['num_heads'],
        mlp_dim=config['model']['mlp_dim'],
        dropout=config['model']['dropout'],
        mask_ratio=config['model']['mask_ratio']
    )
    return model

def train_one_epoch(config, model, dataloader, criterion, optimizer, device, psnr, ssim, epoch_desc="Training", scheduler=None, warmup_scheduler=None, current_epoch=None, warmup_epochs=0):
    model.train()
    running_loss = 0
    total = 0
    pbar = tqdm(dataloader, desc=epoch_desc, leave=False)
    psnr.reset()

    patch_size = config['model']['patch_size']
    in_channels = config['model']['in_channels']

    for inputs in pbar:
        inputs = inputs.to(device)

        optimizer.zero_grad()
        preds_flat, targets_flat = model(inputs)
        loss = criterion(preds_flat, targets_flat)
        loss.backward()
        optimizer.step()

        if warmup_scheduler is not None and current_epoch <= warmup_epochs:
             warmup_scheduler.step()

        running_loss += loss.item()
        total += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        preds_patches = torch.clamp(preds_flat.reshape(-1, in_channels, patch_size, patch_size), 0, 1)
        targets_patches = targets_flat.reshape(-1, in_channels, patch_size, patch_size)
        psnr.update(preds_patches, targets_patches)
        ssim.update((preds_patches, targets_patches))

    avg_loss = running_loss / total
    return avg_loss, psnr.compute(), ssim.compute()

def evaluate(config, model, dataloader, criterion, device, psnr, ssim):
    model.eval()
    total = 0
    psnr.reset()
    running_loss=0

    patch_size = config['model']['patch_size']
    in_channels = config['model']['in_channels']

    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            preds_flat, targets_flat = model(inputs)
            loss = criterion(preds_flat, targets_flat)
            running_loss += loss.item()
            total += 1

            preds_patches = torch.clamp(preds_flat.reshape(-1, in_channels, patch_size, patch_size), 0, 1)
            targets_patches = targets_flat.reshape(-1, in_channels, patch_size, patch_size)
            psnr.update(preds_patches, targets_patches)
            ssim.update((preds_patches, targets_patches))

    avg_loss = running_loss / total
    return avg_loss, psnr.compute(), ssim.compute()

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
    criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=warmup_initial_lr, weight_decay=config['training']['weight_decay'])
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = SSIM(data_range=1.0)

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

    save_path = os.path.join(config['training']['checkpoint_dir'], config['training']['type'], str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    best_val_loss = math.inf
    for epoch in range(1, config['training']['num_epochs'] + 1):
        epoch_desc = f"Epoch {epoch}/{config['training']['num_epochs']}"
        train_loss, train_psnr, train_ssim= train_one_epoch(
            config, model, train_loader, criterion, optimizer, device,
            epoch_desc = epoch_desc,
            scheduler=main_scheduler,
            warmup_scheduler=warmup_scheduler,
            current_epoch=epoch,
            warmup_epochs=warmup_epochs,
            psnr = psnr,
            ssim=ssim
        )
        val_loss, val_psnr, val_ssim = evaluate(config, model, val_loader, criterion, device, psnr = psnr,ssim=ssim)

        if epoch > warmup_epochs:
            main_scheduler.step()

        print(f"\nEpoch {epoch} Summary: "
              f"Train Loss: {train_loss:.4f} , Train PSNR: {train_psnr:.4f}, Train SSIM: {train_ssim:.4f}| "
              f"Val Loss: {val_loss:.4f} , Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}")
        
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }
            os.makedirs(save_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))
        

if __name__ == "__main__":
    main()