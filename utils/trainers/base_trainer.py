import os
import math
import torch
import logging

from abc import ABC, abstractmethod

from utils.logger import Logger
from utils.metrics import MetricHandler
from utils.history import TrainingHistory
from utils.train_utils import make_optimizer, make_schedulers, make_criterion

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    def __init__(self, model, save_path: str, config, train_loader, val_loader, device):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path
        self.warmup_epochs = config["training"]["warmup_epochs"]
        self.num_epochs = self.config["training"]["num_epochs"]
        self.eval_interval = self.config["eval"].get("interval", 0)
        
        self.criterion = self.create_criterion()
        self.optimizer = make_optimizer(config, model)
        self.schedulers = make_schedulers(
            config,
            self.optimizer,
            self.num_epochs,
            self.warmup_epochs * len(train_loader),
        )
        self.metric_handler = MetricHandler(config)
        self.train_logger = Logger(
            self.metric_handler.metric_names,
            len(train_loader),
            len(val_loader),
            self.num_epochs + 1,
        )
        self.history = TrainingHistory()

        self.best_val_loss = math.inf
        self.current_epoch = 0
        self.start_epoch = 0

    @abstractmethod
    def train_epoch(self, epoch):
        """Training logic for one epoch - varies by training type"""
        pass

    @abstractmethod
    def validate(self):
        """Validation logic - varies by training type"""
        pass

    def create_criterion(self):
        """Loss function creation"""
        return make_criterion(self.config)

    def fit(self, num_epochs: int):
        """Common training loop"""
        end_epoch = self.start_epoch + num_epochs

        with self.train_logger:
            for epoch in range(self.start_epoch + 1, end_epoch + 1):
                self.current_epoch = epoch
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate()
                self._update_schedulers(epoch)
                self._log_metrics(train_metrics, val_metrics)
                self._save_if_best(epoch, val_metrics["Loss"])
        self._vizualize()

    def _update_schedulers(self, epoch):
        """Common scheduler update logic"""
        if epoch > self.warmup_epochs:
            self.schedulers["main"].step()

    def _log_metrics(self, train_metrics, val_metrics):
        """Common logging logic"""
        self.train_logger.log_train_epoch(**train_metrics)
        self.train_logger.log_val_epoch(**val_metrics)

    def _save_if_best(self, epoch, val_loss):
        """Common checkpointing logic"""
        if self.best_val_loss >= val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": self.config,
            }
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(self.save_path, "best_model.pth"))

    def _vizualize(self):
        """Common vizualizer func"""
        self.history.vizualize(self.num_epochs)
