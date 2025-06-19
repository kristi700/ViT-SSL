import os
import math
import torch
import logging

from .base_trainer import BaseTrainer
from utils.train_utils import make_optimizer

logger = logging.getLogger(__name__)

class SupervisedTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_backbone = self.config["training"].get("freeze_backbone", False)
        self.freeze_backbone_epochs = self.config.get(
            "freeze_backbone_epochs", float("inf")
        )
        self.best_val_acc = -math.inf

    def train_epoch(
        self,
        epoch: int,
    ):
        self.model.train()
        running_loss, total, correct = 0, 0, 0

        for idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

            if self.schedulers["warmup"] is not None and epoch <= self.warmup_epochs:
                self.schedulers["warmup"].step()

            running_loss += loss.item() * inputs.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)
            self.logger.train_log_step(epoch, idx)

        metrics = self.metric_handler.calculate_metrics(correct=correct, total=total)
        metrics["Loss"] = running_loss / total
        return metrics

    def validate(self):
        self.model.eval()
        running_loss, total, correct = 0, 0, 0

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                preds = self.model(inputs)
                loss = self.criterion(preds, labels)
                running_loss += loss.item() * inputs.size(0)
                correct += (preds.argmax(1) == labels).sum().item()
                total += labels.size(0)
                self.logger.val_log_step(idx)

        metrics = self.metric_handler.calculate_metrics(correct=correct, total=total)
        metrics["Loss"] = running_loss / total
        return metrics

    def fit(self, num_epochs: int):
        end_epoch = self.start_epoch + num_epochs

        with self.logger:
            for epoch in range(self.start_epoch + 1, end_epoch + 1):
                self.current_epoch = epoch
                if self.freeze_backbone and epoch == self.freeze_backbone_epochs:
                    self._unfreeze_backbone()
                    self.optimizer = make_optimizer(self.config, self.model)
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate()
                self._update_schedulers(epoch)
                self._log_metrics(train_metrics, val_metrics)
                self._save_if_best(epoch, val_metrics["Accuracy"])
        self._vizualize()

    def _unfreeze_backbone(self):
        for param in self.model.patch_embedding.parameters():
            param.requires_grad = True
        for param in self.model.encoder_blocks.parameters():
            param.requires_grad = True

    def _save_if_best(self, epoch: int, val_accuracy: float):
        if val_accuracy > self.best_val_acc:
            best_val_acc = val_accuracy
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}. Saving model...")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "config": self.config,
            }
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(self.save_path, "best_model.pth"))
