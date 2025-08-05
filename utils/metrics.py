import torch
from typing import List, Dict, Any, Optional
from torcheval.metrics import (
    PeakSignalNoiseRatio,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassRecall,
    MulticlassPrecision,
    Mean,
)
from torcheval.metrics import StructuralSimilarity

class MetricHandler:
    """
    Handles metric computation using TorchEval for memory efficiency and consistency.
    All metrics are reset after each computation to prevent memory leaks.
    """

    def __init__(self, config: Dict[str, Any]):
        active_metric_names = config.get("metrics", [])
        self.device = config.get("device", "cpu")
        self._metric_calculators = self._get_metric_calculators(active_metric_names)

    def _get_metric_calculators(self, active_metric_names: List[str]):
        registry = {
            "CenterNorm": CenterNormMetric,
            "TeacherMean": TeacherMeanMetric,
            "TeacherSTD": TeacherSTDMetric,
            "TeacherVar": TeacherVarMetric,
            "StudentMean": StudentMeanMetric,
            "StudentSTD": StudentSTDMetric,
            "StudentVar": StudentVarMetric,
            "CosineSim": CosineSimMetric,
            "PSNR": PSNRMetric,
            "SSIM": SSIMMetric,
            "Accuracy": AccuracyMetric,
            "F1Score": F1ScoreMetric,
            "Recall": RecallMetric,
            "Precision": PrecisionMetric,
        }
        calculators = {}
        for name in active_metric_names:
            if name not in registry:
                raise ValueError(f"Unknown metric '{name}'")
            calculators[name] = registry[name](device=self.device)
        return calculators

    def calculate_metrics(self, **kwargs):
        """
        Calculate all active metrics and reset their states to prevent memory leaks.
        """
        latest = {}
        for name, calc in self._metric_calculators.items():
            try:
                latest[name] = calc.compute(**kwargs)

                calc.reset()
            except Exception as e:
                print(f"Error computing metric {name}: {e}")
                latest[name] = float("nan")
        return latest

    @property
    def metric_names(self) -> List[str]:
        return list(self._metric_calculators.keys())

    def reset_all_metrics(self):
        """Reset all metrics to clear accumulated state."""
        for calc in self._metric_calculators.values():
            calc.reset()


class BaseMetric:
    """Base class for all metrics with consistent interface."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._metric = None

    def compute(self, **kwargs) -> float:
        raise NotImplementedError

    def reset(self):
        """Reset metric state if applicable."""
        if hasattr(self._metric, "reset"):
            self._metric.reset()


class CenterNormMetric(BaseMetric):
    """Calculates the L2 norm of center"""

    def compute(self, *, center: torch.Tensor, **kwargs) -> float:
        return torch.linalg.norm(center).item()

    def reset(self):
        pass


class TeacherMeanMetric(BaseMetric):
    """Calculates the mean of the teacher distribution using TorchEval"""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._metric = Mean(device=torch.device(device))

    def compute(self, *, teacher_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(teacher_distribution)
        self._metric.update(flat)
        result = self._metric.compute().item()
        return result


class TeacherSTDMetric(BaseMetric):
    """Calculates the std of the teacher distribution"""

    def compute(self, *, teacher_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(teacher_distribution)
        return flat.std().item()

    def reset(self):
        pass


class TeacherVarMetric(BaseMetric):
    """Calculates the var of the teacher distribution"""

    def compute(self, *, teacher_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(teacher_distribution)
        return flat.var().item()

    def reset(self):
        pass


class StudentMeanMetric(BaseMetric):
    """Calculates the mean of the student distribution using TorchEval"""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._metric = Mean(device=torch.device(device))

    def compute(self, *, student_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(student_distribution)
        self._metric.update(flat)
        result = self._metric.compute().item()
        return result


class StudentSTDMetric(BaseMetric):
    """Calculates the std of the student distribution"""

    def compute(self, *, student_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(student_distribution)
        return flat.std().item()

    def reset(self):
        pass


class StudentVarMetric(BaseMetric):
    """Calculates the var of the student distribution"""

    def compute(self, *, student_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(student_distribution)
        return flat.var().item()

    def reset(self):
        pass


class CosineSimMetric(BaseMetric):
    """
    Computes cosine similarity between teacher and student distributions.
    Optimized version using torch operations.
    """

    def compute(
        self,
        *,
        teacher_distribution: torch.Tensor,
        student_distribution: torch.Tensor,
        **kwargs,
    ) -> float:

        if teacher_distribution.device != student_distribution.device:
            student_distribution = student_distribution.to(teacher_distribution.device)

        teacher_flat = teacher_distribution.view(teacher_distribution.size(0), -1)
        student_flat = student_distribution.view(student_distribution.size(0), -1)

        cos_sim = torch.nn.functional.cosine_similarity(
            teacher_flat, student_flat, dim=1
        )
        return cos_sim.mean().item()

    def reset(self):
        pass


class PSNRMetric(BaseMetric):
    """Calculates PSNR Metric using TorchEval"""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._metric = PeakSignalNoiseRatio(data_range=1.0, device=torch.device(device))

    def compute(
        self, *, preds_patches: torch.Tensor, targets_patches: torch.Tensor, **kwargs
    ) -> float:

        preds_patches = preds_patches.to(self.device)
        targets_patches = targets_patches.to(self.device)

        self._metric.update(preds_patches, targets_patches)
        result = self._metric.compute().item()
        return result


class SSIMMetric(BaseMetric):
    """
    Calculates SSIM Metric using TorchEval functional interface
    to avoid state accumulation issues.
    """

    def compute(
        self, *, preds_patches: torch.Tensor, targets_patches: torch.Tensor, **kwargs
    ) -> float:

        if preds_patches.device != targets_patches.device:
            targets_patches = targets_patches.to(preds_patches.device)

        ssim_value = StructuralSimilarity(
            preds_patches, targets_patches, data_range=1.0
        )
        return ssim_value.item()

    def reset(self):
        pass


class AccuracyMetric(BaseMetric):
    """Calculates Accuracy using TorchEval"""

    def __init__(self, device: str = "cpu", num_classes: Optional[int] = None):
        super().__init__(device)
        self.num_classes = num_classes
        self._metric = MulticlassAccuracy(device=torch.device(device))

    def compute(self, *, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
        if "correct" in kwargs and "total" in kwargs:
            return kwargs["correct"] / kwargs["total"]

        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)

        self._metric.update(y_pred, y_true)
        result = self._metric.compute().item()
        return result


class F1ScoreMetric(BaseMetric):
    """Calculates F1 Score using TorchEval"""

    def __init__(
        self,
        device: str = "cpu",
        num_classes: Optional[int] = None,
        average: str = "macro",
    ):
        super().__init__(device)
        self.num_classes = num_classes
        self._metric = MulticlassF1Score(
            num_classes=num_classes, average=average, device=torch.device(device)
        )

    def compute(self, *, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
        if self.num_classes is None:
            self.num_classes = (
                max(torch.max(y_true).item(), torch.max(y_pred).item()) + 1
            )
            self._metric = MulticlassF1Score(
                num_classes=self.num_classes,
                average="macro",
                device=torch.device(self.device),
            )

        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)

        self._metric.update(y_pred, y_true)
        result = self._metric.compute().item()
        return result


class RecallMetric(BaseMetric):
    """Calculates Recall using TorchEval"""

    def __init__(
        self,
        device: str = "cpu",
        num_classes: Optional[int] = None,
        average: str = "macro",
    ):
        super().__init__(device)
        self.num_classes = num_classes
        self._metric = MulticlassRecall(
            num_classes=num_classes, average=average, device=torch.device(device)
        )

    def compute(self, *, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
        if self.num_classes is None:
            self.num_classes = (
                max(torch.max(y_true).item(), torch.max(y_pred).item()) + 1
            )
            self._metric = MulticlassRecall(
                num_classes=self.num_classes,
                average="macro",
                device=torch.device(self.device),
            )

        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)

        self._metric.update(y_pred, y_true)
        result = self._metric.compute().item()
        return result


class PrecisionMetric(BaseMetric):
    """Calculates Precision using TorchEval"""

    def __init__(
        self,
        device: str = "cpu",
        num_classes: Optional[int] = None,
        average: str = "macro",
    ):
        super().__init__(device)
        self.num_classes = num_classes
        self._metric = MulticlassPrecision(
            num_classes=num_classes, average=average, device=torch.device(self.device)
        )

    def compute(self, *, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
        if self.num_classes is None:
            self.num_classes = (
                max(torch.max(y_true).item(), torch.max(y_pred).item()) + 1
            )
            self._metric = MulticlassPrecision(
                num_classes=self.num_classes,
                average="macro",
                device=torch.device(self.device),
            )

        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)

        self._metric.update(y_pred, y_true)
        result = self._metric.compute().item()
        return result
