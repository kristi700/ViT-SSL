import os
import math
import torch
import logging

from torch.amp import autocast

from .base_trainer import BaseTrainer
from vit_core.ssl.dino.loss import DINOLoss
from vit_core.ssl.dino.dino_utils import DINOMomentumScheduler, DINOTeacherTempScheduler

logger = logging.getLogger(__name__)

class DINOTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum_schedule = DINOMomentumScheduler(
            self.config.training.teacher_momentum_start,
            self.config.training.teacher_momentum_final,
            self.num_epochs,
        )
        temp_final = self.config.training.get("teacher_temp_final", None)
        if temp_final is None:
            temp_final = self.config.training.teacher_temp
        self.temp_schedule = DINOTeacherTempScheduler(
            self.config.training.teacher_temp,
            temp_final,
            self.num_epochs,
            self.config.training.get("teacher_temp_scheduler", "cosine"),
        )
        self.eval_mode = self.config["eval"].get("mode")
        self.best_score = - math.inf

    def create_criterion(self):
        return DINOLoss(
            self.config.training.teacher_temp, self.config.training.student_temp
        )

    def fit(self, num_epochs: int):
        """Common training loop with unsupervised validation"""
        end_epoch = self.start_epoch + num_epochs

        with self.train_logger:
            for epoch in range(self.start_epoch + 1, end_epoch + 1):
                self.current_epoch = epoch
                self.criterion.teacher_temp = self.temp_schedule.get_temp(epoch)
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate()
                self._update_schedulers(epoch)
                self._log_metrics(train_metrics, val_metrics)
                self._save_if_best(epoch, val_metrics)
                self._save_last(epoch)
                if (
                    self.eval_interval
                    and self.eval_mode
                    and epoch % self.eval_interval == 0
                ):
                    logger.info(f"Running automatic evaluation (mode: {self.eval_mode})...")
                    from evaluators.unsupervised_evaluator import (
                        run_evaluation,
                    )

                    self.train_logger.pause()
                    run_evaluation(
                        self.config,
                        self.model,
                        self.device,
                        os.path.join(self.save_path, f"epoch_{epoch}"),
                    )
                    self.train_logger.resume()
        self._vizualize()

    def train_epoch(
        self,
        epoch: int,
    ):
        self.model.train()
        total, running_loss = 0, 0
        num_global_views = self.train_loader.dataset.num_global_views
        current_teach_momentum = self.momentum_schedule.get_momentum(epoch)

        metrics = {}
        metrics_count = 0

        for idx, inputs in enumerate(self.train_loader):
            inputs = [x.to(self.device) for x in inputs]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
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
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.model.momentum_update_teacher(current_teach_momentum)
            if self.schedulers["warmup"] is not None and epoch <= self.warmup_epochs:
                self.schedulers["warmup"].step()

            running_loss += loss.item()
            total += 1
            teacher_cpu = teacher_output.detach().cpu()
            student_cpu = student_output.detach().cpu()
            center_cpu = self.model.center.detach().cpu()
            batch_metrics = self.metric_handler.calculate_metrics(
                center=center_cpu,
                teacher_distribution=teacher_cpu,
                student_distribution=student_cpu,
            )
            if metrics_count == 0:
                metrics = batch_metrics.copy()
            else:
                for key, value in batch_metrics.items():
                    if key in metrics:
                        metrics[key] = (metrics[key] * metrics_count + value) / (metrics_count + 1)
                    else:
                        metrics[key] = value
            
            metrics_count += 1
            
            del teacher_cpu, student_cpu, center_cpu, batch_metrics
            self.train_logger.train_log_step(epoch, idx)
        metrics["Loss"] = running_loss / total
        return metrics

    def validate(self):
        self.model.eval()
        total, running_loss = 0, 0
        num_global_views = self.val_loader.dataset.num_global_views
    
        metrics = {}
        metrics_count = 0
        
        with torch.no_grad():
            for idx, inputs in enumerate(self.val_loader):
                inputs = [x.to(self.device) for x in inputs]
                with autocast(device_type="cuda", dtype=torch.bfloat16):
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
                
                teacher_cpu = teacher_output.detach().cpu()
                student_cpu = student_output.detach().cpu()
                center_cpu = self.model.center.detach().cpu()
                batch_metrics = self.metric_handler.calculate_metrics(
                    center=center_cpu,
                    teacher_distribution=teacher_cpu,
                    student_distribution=student_cpu,
                )
                if metrics_count == 0:
                    metrics = batch_metrics.copy()
                else:
                    for key, value in batch_metrics.items():
                        if key in metrics:
                            metrics[key] = (metrics[key] * metrics_count + value) / (metrics_count + 1)
                        else:
                            metrics[key] = value
                
                del teacher_cpu, student_cpu, center_cpu, batch_metrics
                metrics_count += 1
                total += 1
                running_loss += loss.item()
                self.train_logger.val_log_step(idx)

        metrics["Loss"] = running_loss / total
        return metrics

    def _save_if_best(self, epoch, val_metrics):
        score = (val_metrics['CosineSim'] - abs(val_metrics['CenterNorm'] - 1) - abs(val_metrics['StudentSTD'] - val_metrics['TeacherSTD']))
        if score > self.best_score:
            self.best_score = score
            logger.info(
                f"New best validation score: {self.best_score:.4f}. Saving model..."
            )
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_score": self.best_score,
                "config": self.config,
            }
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(self.save_path, "best_model.pth"))
            self.train_logger.resume()