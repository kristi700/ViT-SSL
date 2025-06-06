import os
import torch
import matplotlib.pyplot as plt

from collections import defaultdict
from typing import Dict, List, Any


class TrainingHistory:
    def __init__(self, save_path: str = None):
        self.save_path = save_path
        self.history: Dict[str, List[Any]] = defaultdict(list)
        self.epoch_count = 0

    def update(
        self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any], epoch: int
    ):
        """
        Expects two dicts:
          - train_metrics: e.g. {"CenterNorm": 1.23, "Mean": 0.45, ... , "loss": 0.678}
          - val_metrics:   e.g. {"CenterNorm": 1.10, "Mean": 0.40, ... , "loss": 0.712}
        Prefixes keys internally so that plots will pair train_... vs. val_...
        """
        self.epoch_count = max(self.epoch_count, epoch)

        def _to_scalar(v):
            if torch.is_tensor(v):
                if v.is_cuda:
                    v = v.cpu()
                if v.numel() == 1:
                    return v.item()
            return v

        for name, value in train_metrics.items():
            scalar = _to_scalar(value)
            self.history[f"train_{name.lower()}"].append(scalar)

        for name, value in val_metrics.items():
            scalar = _to_scalar(value)
            self.history[f"val_{name.lower()}"].append(scalar)

    def _get_plot_configs(self):
        """
        Groups metrics by the suffix (after the first underscore),
        so that e.g. "train_loss" and "val_loss" are plotted together,
        "train_centernorm" and "val_centernorm" are grouped, etc.
        """
        groups: Dict[str, List[str]] = {}
        for full_name in self.history:
            if "_" in full_name:
                base = full_name.split("_", 1)[1]
            else:
                base = full_name
            groups.setdefault(base, []).append(full_name)

        configs: List[Dict[str, Any]] = []
        for base, names in groups.items():
            configs.append(
                {
                    "title": f"{base.replace('_',' ').title()} Over Epochs",
                    "ylabel": base.upper() if base == "lr" else base.title(),
                    "metrics_to_plot": [
                        {"name": n, "label": n.replace("_", " ").title()}
                        for n in sorted(names)
                    ],
                    "filename": f"{base}_plot.png",
                }
            )
        return configs

    def vizualize(self, num_epochs: int):
        """
        Plot each group (e.g. train_loss vs. val_loss) over `num_epochs`.
        Saves each figure into self.save_path.
        """
        if not self.save_path:
            return

        epochs_range = range(1, num_epochs + 1)
        plot_configs = self._get_plot_configs()

        os.makedirs(self.save_path, exist_ok=True)

        for cfg in plot_configs:
            title = cfg["title"]
            ylabel = cfg["ylabel"]
            metrics_to_plot = cfg["metrics_to_plot"]
            filename = cfg["filename"]

            plt.figure()
            for md in metrics_to_plot:
                name = md["name"]
                label = md["label"]
                data = self.history.get(name, [])
                if len(data) < num_epochs:
                    continue
                plt.plot(epochs_range, data, label=label)

            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(title)
            if any(m["label"] for m in metrics_to_plot):
                plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            save_path = os.path.join(self.save_path, filename)
            plt.savefig(save_path)
            plt.close()
