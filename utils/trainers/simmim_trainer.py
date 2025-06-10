import os
import torch

from .base_trainer import BaseTrainer

class SimMIMTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = self.config["model"]["patch_size"]
        self.in_channels = self.config["model"]["in_channels"]

    def train_epoch(
        self,
        epoch: int,
    ):
        self.model.train()
        running_loss = 0
        total = 0
        all_pred_patches, all_target_patches = [], []

        for idx, inputs in enumerate(self.train_loader):
            inputs = inputs.to(self.device)

            self.optimizer.zero_grad()
            preds_flat, targets_flat = self.model(inputs)
            loss = self.criterion(preds_flat, targets_flat)
            loss.backward()
            self.optimizer.step()

            if self.schedulers['warmup'] is not None and epoch <= self.warmup_epochs:
                self.schedulers['warmup'].step()

            running_loss += loss.item()
            total += 1

            preds_patches = torch.clamp(
                preds_flat.reshape(-1, self.in_channels, self.patch_size, self.patch_size), 0, 1
            )
            targets_patches = targets_flat.reshape(-1, self.in_channels, self.patch_size, self.patch_size)
            all_pred_patches.append(preds_patches)
            all_target_patches.append(targets_patches)
            self.logger.train_log_step(epoch, idx)

        metrics = self.metric_handler.calculate_metrics(
            preds_patches=torch.cat(all_pred_patches, dim=0),
            targets_patches=torch.cat(all_target_patches, dim=0),
        )
        metrics["Loss"] = running_loss / total
        return metrics

    def validate(self):
        self.model.eval()
        total = 0
        running_loss = 0
        all_pred_patches, all_target_patches = [], []
        
        with torch.no_grad():
            for idx, inputs in enumerate(self.val_loader):
                inputs = inputs.to(self.device)
                preds_flat, targets_flat = self.model(inputs)
                loss = self.criterion(preds_flat, targets_flat)
                running_loss += loss.item()
                total += 1

                preds_patches = torch.clamp(
                    preds_flat.reshape(-1, self.in_channels, self.patch_size, self.patch_size), 0, 1
                )
                targets_patches = targets_flat.reshape(
                    -1, self.in_channels, self.patch_size, self.patch_size
                )
                all_pred_patches.append(preds_patches)
                all_target_patches.append(targets_patches)
                self.logger.val_log_step(idx)

        metrics = self.metric_handler.calculate_metrics(
            preds_patches=torch.cat(all_pred_patches, dim=0),
            targets_patches=torch.cat(all_target_patches, dim=0),
        )
        metrics["Loss"] = running_loss / total
        return metrics
    
    def _update_schedulers(self, epoch):
        if epoch > self.warmup_epochs:
            self.schedulers["main"].step()

    def _save_if_best(self, epoch:int, val_loss: float):
            if self.best_val_loss > val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": self.config,
                }
                os.makedirs(self.save_path, exist_ok=True)
                torch.save(checkpoint, os.path.join(self.save_path, "best_model.pth"))
    