import os
import torch
import logging

from .base_trainer import BaseTrainer
from vit_core.ssl.dino.loss import DINOLoss
from vit_core.ssl.dino.dino_utils import DINOMomentumScheduler

logger = logging.getLogger(__name__)

class DINOTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum_schedule = DINOMomentumScheduler(
            self.config.training.teacher_momentum_start,
            self.config.training.teacher_momentum_final,
            self.num_epochs,
        )

        self.eval_interval = self.config["eval"].get("interval", 0)
        self.eval_mode = self.config["eval"].get("mode")

    def create_criterion(self):
        return DINOLoss(
            self.config.training.teacher_temp, self.config.training.student_temp
        )

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
        num_global_views = (
            self.train_loader.dataset.dataset.num_global_views
        )  # TODO - might not be ideal like this
        current_teach_momentum = self.momentum_schedule.get_momentum(epoch)

        for idx, inputs in enumerate(self.train_loader):
            inputs = [x.to(self.device) for x in inputs]

            self.optimizer.zero_grad()
            teacher_output, student_output = self.model(inputs, num_global_views)

            teacher_output = teacher_output.view(
                num_global_views,
                int(teacher_output.shape[0] / num_global_views),
                teacher_output.shape[1],
            )
            student_output = student_output.view(
                len(inputs),
                int(student_output.shape[0] / len(inputs)),
                student_output.shape[1],
            )

            loss = self.criterion(teacher_output, student_output, self.model.center)
            loss.backward()
            self.optimizer.step()
            self.model.momentum_update_teacher(current_teach_momentum)
            if self.schedulers["warmup"] is not None and epoch <= self.warmup_epochs:
                self.schedulers["warmup"].step()

            running_loss += loss.item()
            total += 1

            self.logger.train_log_step(epoch, idx)

        metrics = self.metric_handler.calculate_metrics(
            center=self.model.center,
            teacher_distribution=teacher_output,
            student_distribution=student_output,
        )
        metrics["Loss"] = running_loss / total
        return metrics

    def validate(self):
        self.model.eval()
        total, running_loss = 0, 0
        num_global_views = (
            self.val_loader.dataset.dataset.num_global_views
        )  # TODO - might not be ideal like this

        with torch.no_grad():
            for idx, inputs in enumerate(self.val_loader):
                inputs = [x.to(self.device) for x in inputs]
                teacher_output, student_output = self.model(inputs, num_global_views)

                teacher_output = teacher_output.view(
                    num_global_views,
                    int(teacher_output.shape[0] / num_global_views),
                    teacher_output.shape[1],
                )
                student_output = student_output.view(
                    len(inputs),
                    int(student_output.shape[0] / len(inputs)),
                    student_output.shape[1],
                )

                loss = self.criterion(teacher_output, student_output, self.model.center)
                running_loss += loss.item()
                total += 1
                self.logger.val_log_step(idx)

        metrics = self.metric_handler.calculate_metrics(
            center=self.model.center,
            teacher_distribution=teacher_output,
            student_distribution=student_output,
        )
        metrics["Loss"] = running_loss / total
        return metrics
