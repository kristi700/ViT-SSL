import os
from typing import List
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.layout import Layout


class Logger:
    def __init__(
        self,
        metric_names: List[str],
        train_total_batches: int,
        val_total_batches: int,
        num_epochs: int,
        metrics_per_row: int = 4,
    ):
        self.console = Console()
        self.metric_names = metric_names + ["Loss"]
        self.train_total_batches = train_total_batches
        self.val_total_batches = val_total_batches
        self.num_epochs = num_epochs
        self.metrics_per_row = metrics_per_row

        self._make_tables()

        self.left_progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )
        self.right_progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )

        self.layout = Layout()
        self.layout.split_row(Layout(name="left"), Layout(name="right"))
        self.layout["left"].ratio = 1
        self.layout["right"].ratio = 1

        self._refresh_layout()
        self.live = Live(self.layout, refresh_per_second=10, console=self.console)

    def _clear_console(self):
        os.system("cls" if os.name == "nt" else "clear")

    def _make_tables(self):

        self.train_table = Table(expand=True, title="Training", show_lines=True)
        self.train_table.add_column("Type")
        self.train_table.add_column("Value")

        self.val_table = Table(expand=True, title="Validation", show_lines=True)
        self.val_table.add_column("Type")
        self.val_table.add_column("Value")

    def _refresh_layout(self):
        left_group = Group(self.left_progress, self.train_table)
        right_group = Group(self.right_progress, self.val_table)
        self.layout["left"].update(left_group)
        self.layout["right"].update(right_group)

    def __enter__(self):
        self._clear_console()
        self.live.start()
        self.train_task = self.left_progress.add_task(
            "Train", total=self.train_total_batches
        )
        self.val_task = self.right_progress.add_task(
            "Val", total=self.val_total_batches
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.live.stop()

    def pause(self):
        self.live.stop()
        self._clear_console()

    def resume(self):
        self._refresh_layout()
        self.live.start()

    def train_log_step(self, epoch: int, batch_idx: int):
        """Call once per training batch."""
        desc = f"Epoch {epoch} / {self.num_epochs} Train"
        self.left_progress.update(
            self.train_task, description=desc, completed=batch_idx + 1
        )

    def val_log_step(self, batch_idx: int):
        """Call once per training batch."""
        self.right_progress.update(
            self.val_task, description="Val", completed=batch_idx + 1
        )

    def log_train_epoch(self, **metrics: float):
        """Update train-side table after training epoch."""
        self.train_table = Table(expand=True, title="Train", show_lines=True)
        self.train_table.add_column("Type")
        self.train_table.add_column("Value")

        for name in self.metric_names:
            value = metrics.get(name, 0)
            self.train_table.add_row(name, f"{value:.4f}")

        print(f"Train table now has {len(self.train_table.rows)} rows")
        self._refresh_layout()

    def log_val_epoch(self, **metrics: float):
        """Update val-side table after validation epoch."""
        self.val_table = Table(expand=True, title="Validation", show_lines=True)
        self.val_table.add_column("Type")
        self.val_table.add_column("Value")

        for name in self.metric_names:
            value = metrics.get(name, 0)
            self.val_table.add_row(name, f"{value:.4f}")

        print(f"Val table now has {len(self.val_table.rows)} rows")
        self._refresh_layout()
