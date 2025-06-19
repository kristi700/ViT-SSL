import os
import torch
import logging

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class SimMIMTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = self.config["model"]["patch_size"]
        self.in_channels = self.config["model"]["in_channels"]

        self.eval_interval = self.config["eval"].get("interval", 0)
        self.eval_mode = self.config["eval"].get("mode")

    def fit(self, num_epochs: int):
        """Common training loop with unsupervised validation"""
        end_epoch = self.start_epoch + num_epochs

        with self.logger:
            for epoch in range(self.start_epoch + 1, end_epoch + 1):
                self.current_epoch = epoch
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate()
                self._update_schedulers(epoch)
                self._log_metrics(train_metrics, val_metrics)
                self._save_if_best(epoch, val_metrics["Loss"])
                if (
                    self.eval_interval
                    and self.eval_mode
                    and epoch % self.eval_interval == 0
                ):
                    logger.info(f"Running automatic evaluation (mode: {self.eval_mode})...")
                    from evaluators.unsupervised_evaluator import (
                        run_evaluation,
                    )

                    self.logger.pause()
                    run_evaluation(
                        self.config,
                        self.model,
                        self.device,
                        os.path.join(self.save_path, f"epoch_{epoch}"),
                    )
                    self.logger.resume()
        self._vizualize()

    def train_epoch(
        self,
        epoch: int,
    ):
        self.model.train()
        total, running_loss = 0, 0
        all_pred_patches, all_target_patches = [], []

        for idx, inputs in enumerate(self.train_loader):
            inputs = inputs.to(self.device)

            self.optimizer.zero_grad()
            preds_flat, targets_flat = self.model(inputs)
            loss = self.criterion(preds_flat, targets_flat)
            loss.backward()
            self.optimizer.step()

            if self.schedulers["warmup"] is not None and epoch <= self.warmup_epochs:
                self.schedulers["warmup"].step()

            running_loss += loss.item()
            total += 1

            preds_patches = torch.clamp(
                preds_flat.reshape(
                    -1, self.in_channels, self.patch_size, self.patch_size
                ),
                0,
                1,
            )
            targets_patches = targets_flat.reshape(
                -1, self.in_channels, self.patch_size, self.patch_size
            )
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
        total, running_loss = 0, 0
        all_pred_patches, all_target_patches = [], []

        with torch.no_grad():
            for idx, inputs in enumerate(self.val_loader):
                inputs = inputs.to(self.device)
                preds_flat, targets_flat = self.model(inputs)
                loss = self.criterion(preds_flat, targets_flat)
                running_loss += loss.item()
                total += 1

                preds_patches = torch.clamp(
                    preds_flat.reshape(
                        -1, self.in_channels, self.patch_size, self.patch_size
                    ),
                    0,
                    1,
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
