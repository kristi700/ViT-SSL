class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, start_lr, target_lr):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = max(1, warmup_steps)
        self.target_lr = target_lr
        self.start_lr = start_lr
        self.lr_steps = [
            (target_lr - start_lr) / self.warmup_steps for _ in optimizer.param_groups
        ]

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            lr_scale = float(self._step) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.start_lr + lr_scale * (
                    self.target_lr - self.start_lr
                )
