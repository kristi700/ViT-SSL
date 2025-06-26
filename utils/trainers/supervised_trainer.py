import os
import math
import torch
import logging

from torch.cuda.amp import autocast

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
        running_loss  = 0
        all_preds, all_labels = [], []

        for idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                preds = self.model(inputs)
                loss = self.criterion(preds, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.schedulers["warmup"] is not None and epoch <= self.warmup_epochs:
                self.schedulers["warmup"].step()

            running_loss += loss.item() * inputs.size(0)
            all_preds.append(preds.argmax(1).cpu())
            all_labels.append(labels.cpu())
            self.train_logger.train_log_step(epoch, idx)

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)

        metrics = self.metric_handler.calculate_metrics(correct=(y_pred == y_true).sum().item(), total=len(y_true), y_pred=y_pred, y_true=y_true)
        metrics["Loss"] = running_loss / len(y_true)
        return metrics

    def validate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        running_loss = 0
        
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)

                running_loss += loss.item() * inputs.size(0)
                self.train_logger.val_log_step(idx)
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)

        metrics = self.metric_handler.calculate_metrics(correct=(y_pred == y_true).sum().item(), total=len(y_true), y_pred=y_pred, y_true=y_true)
        metrics["Loss"] = running_loss / len(y_true)
        return metrics, torch.cat(all_preds), torch.cat(all_labels)

    def fit(self, num_epochs: int):
        end_epoch = self.start_epoch + num_epochs

        with self.train_logger:
            for epoch in range(self.start_epoch + 1, end_epoch + 1):
                self.current_epoch = epoch
                if self.freeze_backbone and epoch == self.freeze_backbone_epochs:
                    self._unfreeze_backbone()
                    self.optimizer = make_optimizer(self.config, self.model)
                train_metrics = self.train_epoch(epoch)
                val_metrics, preds, labels = self.validate()
                self._update_schedulers(epoch)
                self._log_metrics(train_metrics, val_metrics)
                self._save_if_best(epoch, val_metrics["Accuracy"])
                self._save_last(epoch)
                if (
                    self.eval_interval
                    and epoch % self.eval_interval == 0
                ):
                    logger.info(f"Running automatic evaluation...")
                    from evaluators.supervised_evaluator import (
                        run_evaluation,
                    )

                    self.train_logger.pause()
                    run_evaluation(
                        self.config,
                        self.model,
                        self.device,
                        os.path.join(self.save_path, f"epoch_{epoch}"),
                        val_metrics["Accuracy"],
                        preds,
                        labels

                    )
                    self.train_logger.resume()
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
