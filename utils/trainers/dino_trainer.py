import os
import torch

from .base_trainer import BaseTrainer
from vit_core.ssl.dino.loss import DINOLoss
from vit_core.ssl.dino.dino_utils import DINOMomentumScheduler

class DINOTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum_schedule = DINOMomentumScheduler(
        self.config.training.teacher_momentum_start,
        self.config.training.teacher_momentum_final,
        self.num_epochs,
        )
        self.criterion = self.create_criterion()
    
    def create_criterion(self):
        return DINOLoss(
            self.config.training.teacher_temp,
            self.config.training.student_temp
        )
    
    def train_epoch(
        self,
        epoch: int,
    ):
        self.model.train()
        running_loss = 0
        total = 0
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
            if self.schedulers['warmup'] is not None and epoch <= self.warmup_epochs:
                self.schedulers['warmup'].step()

            running_loss += loss.item()
            total += 1

            self.logger.train_log_step(epoch, idx)

        metrics = self.metric_handler.calculate_metrics(
            center=self.model.center,
            teacher_distribution=teacher_output,
            student_distribution=student_output,
        )
        metrics["loss"] = running_loss / total
        return metrics

    def validate(self):
        self.model.eval()
        total = 0
        running_loss = 0
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
        metrics["loss"] = running_loss / total
        return metrics
    
    def _update_schedulers(self, epoch):
        if epoch > self.warmup_epochs:
            self.schedulers["main"].step()

    def _log_metrics(self, train_metrics, val_metrics):
        self.logger.log_train_epoch(**train_metrics)
        self.logger.log_val_epoch(**val_metrics)

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
    