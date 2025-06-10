import os
import math

from datetime import datetime
from abc import ABC, abstractmethod

from utils.logger import Logger
from utils.metrics import MetricHandler
from utils.history import TrainingHistory
from utils.train_utils import make_optimizer, make_schedulers

class BaseTrainer(ABC):
    def __init__(self, model, config, train_loader, val_loader, device):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs= self.config["training"]["num_epochs"]
        self.warmup_epochs = config["training"]["warmup_epochs"]
        self._get_save_path()

        self.optimizer = make_optimizer(config, model)
        self.schedulers = make_schedulers(config, self.optimizer, self.num_epochs, self.warmup_epochs * len(train_loader))
        self.metric_handler = MetricHandler(config)
        self.logger = Logger(self.metric_handler.metric_names, len(train_loader), len(val_loader), self.num_epochs+1)
        self.history = TrainingHistory()
        
        self.best_val_loss = math.inf
        self.current_epoch = 0
    
    def _get_save_path(self):
        self.save_path = os.path.join(
            self.config["training"]["checkpoint_dir"],
            self.config["training"]["type"],
            str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
        )

    @abstractmethod
    def train_epoch(self, epoch):
        """Training logic for one epoch - varies by training type"""
        pass
    
    @abstractmethod
    def validate(self):
        """Validation logic - varies by training type"""
        pass
    
    @abstractmethod
    def create_criterion(self):
        """Loss function creation - specific to each method"""
        pass
    
    def fit(self, num_epochs):
        """Common training loop"""
        with self.logger:
            for epoch in range(1, num_epochs + 1):
                self.current_epoch = epoch
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate()
                self._update_schedulers(epoch)
                self._log_metrics(train_metrics, val_metrics)
                self._save_if_best(epoch, val_metrics['loss'])
        self._vizualize()
    
    def _update_schedulers(self, epoch):
        """Common scheduler update logic"""
        pass
    
    def _log_metrics(self, train_metrics, val_metrics):
        """Common logging logic"""
        pass
    
    def _save_if_best(self, epoch, val_loss):
        """Common checkpointing logic"""
        pass

    def _vizualize(self):
        """ Common vizualizer func"""
        self.history.vizualize(self.num_epochs)