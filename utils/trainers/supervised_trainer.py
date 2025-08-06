import os
import math
import torch
import logging

from torch.amp import autocast

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
        total = 0
        metrics = {}
        metrics_count = 0

        for idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.schedulers["warmup"] is not None and epoch <= self.warmup_epochs:
                self.schedulers["warmup"].step()

            running_loss += loss.item() * inputs.size(0)
            self.train_logger.train_log_step(epoch, idx)
            total += 1
            preds_cpu = logits.argmax(1).detach().cpu()
            labels_cpu = labels.detach().cpu()
            batch_metrics = self.metric_handler.calculate_metrics(
                correct=(preds_cpu == labels_cpu).sum().item(),
                total=labels_cpu.size(0),
                y_pred=preds_cpu,
                y_true=labels_cpu,
            )

            if metrics_count == 0:
                metrics = batch_metrics.copy()
            else:
                for key, value in batch_metrics.items():
                    if key in metrics:
                        metrics[key] = (metrics[key] * metrics_count + value) / (
                            metrics_count + 1
                        )
                    else:
                        metrics[key] = value

            metrics_count += 1

            del preds_cpu, labels_cpu, batch_metrics
        metrics["Loss"] = running_loss / total
        return metrics

    def validate(self):
        self.model.eval()
        running_loss = 0
        total = 0
        all_preds, all_labels = [], []
        metrics = {}
        metrics_count = 0

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)

                running_loss += loss.item() * inputs.size(0)
                self.train_logger.val_log_step(idx)
                total += 1

                preds_cpu = logits.argmax(1).detach().cpu()
                labels_cpu = labels.detach().cpu()
                all_preds.append(preds_cpu)
                all_labels.append(labels_cpu)
                batch_metrics = self.metric_handler.calculate_metrics(
                    correct=(preds_cpu == labels_cpu).sum().item(),
                    total=labels_cpu.size(0),
                    y_pred=preds_cpu,
                    y_true=labels_cpu,
                )

                if metrics_count == 0:
                    metrics = batch_metrics.copy()
                else:
                    for key, value in batch_metrics.items():
                        if key in metrics:
                            metrics[key] = (metrics[key] * metrics_count + value) / (
                                metrics_count + 1
                            )
                        else:
                            metrics[key] = value

                metrics_count += 1

                del preds_cpu, labels_cpu, batch_metrics

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        metrics["Loss"] = running_loss / total
        return metrics, y_pred, y_true

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
            self.best_val_acc = val_accuracy
            logger.info(f"New best validation accuracy: {self.best_val_acc:.4f}. Saving model...")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
                "config": self.config,
            }
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(self.save_path, "best_model.pth"))
