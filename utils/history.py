import os
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from collections import defaultdict


class TrainingHistory:
    def __init__(self, save_path: str = None):
        self.save_path = save_path
        self.history: Dict[str, List[Any]] = defaultdict(list)
        self.epoch_count = 0

    def update(self, metrics: Dict[str, Any], epoch: int):
        if epoch > self.epoch_count:
            self.epoch_count = epoch # shouldnt happen
        
        for name, value in metrics.items():
            if torch.is_tensor(value):
                if value.is_cuda:
                    value = value.cpu()
                if value.numel() ==1:
                    value = value.item()

            self.history[name] = value
    
    def _get_plot_configs(self):
        metrics = {
            name: getattr(self, name)
            for name, val in vars(self).items()
            if isinstance(val, list)
        }

        groups: Dict[str, List[str]] = {}
        for name in metrics:
            base = name.split('_', 1)[1] if '_' in name else name
            groups.setdefault(base, []).append(name)

        configs: List[Dict[str, Any]] = []
        for base, names in groups.items():
            configs.append({
                "title": f"{base.replace('_',' ').title()} Over Epochs",
                "ylabel": base.upper() if base == "lr" else base.title(),
                "metrics_to_plot": [
                    {"name": n, "label": n.replace('_',' ').title()}
                    for n in sorted(names)
                ],
                "filename": f"{base}_plot.png"
            })
        return configs

    def vizualize(self, num_epochs: int):
        epochs_range = range(1, num_epochs + 1)
        plots_configs = self._get_plot_configs()
        for plot_info in plots_configs:
            title = plot_info.get("title", "Metric Plot")
            ylabel = plot_info.get("ylabel", "Value")
            metrics_to_plot = plot_info.get("metrics_to_plot", [])
            filename = plot_info.get("filename", f"{title.lower().replace(' ', '_')}.png")

            for metric_detail in metrics_to_plot:
                metric_name = metric_detail["name"]
                metric_label = metric_detail.get("label", metric_name)
                data = self.get_metric(metric_name)
                plt.plot(epochs_range, data, label=metric_label)

            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(title)

            if any(m["label"] for m in metrics_to_plot):
                 plt.legend()

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            os.makedirs(self.save_path, exist_ok=True)
            plot_save_path = os.path.join(self.save_path, filename)
            plt.savefig(plot_save_path)
            plt.close()