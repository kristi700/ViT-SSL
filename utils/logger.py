import os
from typing import List
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

class Logger:
    def __init__(self, metric_names: List[str], total_batches: int, num_epochs: int):
        self.console = Console()
        self.metric_names = metric_names
        self.total_batches = total_batches
        self.num_epochs =num_epochs
        self._make_table()

        self.progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self.task = None
        
        self.container = Group(self.progress, self.table)
        self.live = Live(self.container, refresh_per_second=10, console=self.console)

    def _clear_console(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def _make_table(self):
        self.table = Table(expand=True)
        for col in ["Epoch", "Train Loss", "Val Loss", *self.metric_names]:
            self.table.add_column(col, justify="center")

    def __enter__(self):
        self._clear_console()
        self.live.start()
        self.task = self.progress.add_task("Train Epoch", total=self.total_batches)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.live.stop()

    def log_step(self, epoch:int, epoch_type: str):
        """Call this once per batch."""
        self.progress.update(self.task, description=f"{epoch_type.capitalize()} Epoch: {epoch} / {self.num_epochs}")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        **metrics: float
    ):
        self._clear_console()
        self.progress.reset(self.task)
        self.progress.start_task(self.task)

        self.table = Table(expand=True)
        for col in ["Epoch", "Train Loss", "Val Loss", *self.metric_names]:
            self.table.add_column(col, justify="center")

        row = [
            str(epoch),
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            *[f"{metrics.get(name, 0):.4f}" for name in self.metric_names]
        ]
        self.table.add_row(*row)

        self.container = Group(self.progress, self.table)
        self.live.update(self.container)
