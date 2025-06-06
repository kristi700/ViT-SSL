import torch

from ignite.metrics import SSIM
from typing import List, Dict, Any
from torcheval.metrics import PeakSignalNoiseRatio


class MetricHandler:
    def __init__(self, config: Dict[str, Any]):
        active_metric_names = config.get("metrics", [])
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
            "Accuracy": Accuracy,
        }
        calculators = {}
        for name in active_metric_names:
            if name not in registry:
                raise ValueError(f"Unknown metric '{name}'")
            calculators[name] = registry[name]()
        return calculators

    def calculate_metrics(self, **kwargs):
        latest = {}
        for name, calc in self._metric_calculators.items():
            latest[name] = calc.compute(**kwargs)
        return latest

    @property
    def metric_names(self) -> List[str]:
        return list(self._metric_calculators.keys())


# ------------------------------------------------------------------------------------------
# -----------------------------------------METRICS------------------------------------------
# ------------------------------------------------------------------------------------------


class BaseMetric:
    def compute(self, **kwargs) -> float:
        raise NotImplementedError


class CenterNormMetric(BaseMetric):
    """
    Calculates the L2 norm of center
    """

    def compute(self, *, center: torch.Tensor, **kwargs) -> float:
        return torch.linalg.norm(center).item()


class TeacherMeanMetric(BaseMetric):
    """
    Calculates the mean of the given distribution
    """

    def compute(self, *, teacher_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(teacher_distribution)
        return flat.mean().item()


class TeacherSTDMetric(BaseMetric):
    """
    Calculates the std of the given distribution
    """

    def compute(self, *, teacher_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(teacher_distribution)
        return flat.std().item()


class TeacherVarMetric(BaseMetric):
    """
    Calculates the var of the given distribution
    """

    def compute(self, *, teacher_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(teacher_distribution)
        return flat.var().item()


class StudentMeanMetric(BaseMetric):
    """
    Calculates the mean of the given distribution
    """

    def compute(self, *, student_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(student_distribution)
        return flat.mean().item()


class StudentSTDMetric(BaseMetric):
    """
    Calculates the std of the given distribution
    """

    def compute(self, *, student_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(student_distribution)
        return flat.std().item()


class StudentVarMetric(BaseMetric):
    """
    Calculates the var of the given distribution
    """

    def compute(self, *, student_distribution: torch.Tensor, **kwargs) -> float:
        flat = torch.flatten(student_distribution)
        return flat.var().item()


class CosineSimMetric(BaseMetric):
    """
    Computes cosine similarity between the teacher's outputs and the student's.
        args:
            - teacher_output - num_global_views * batch_size * dim
            - student_output - num_all_views * batch_size * dim

    Implemented as in: https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
    """

    def compute(
        self,
        *,
        teacher_distribution: torch.Tensor,
        student_distribution: torch.Tensor,
        **kwargs,
    ) -> float:
        teacher_normed = torch.linalg.norm(teacher_distribution, dim=-1)
        student_normed = torch.linalg.norm(student_distribution, dim=-1)

        teacher_exp = teacher_distribution.unsqueeze(1)
        student_exp = student_distribution.unsqueeze(0)

        dot_product = (teacher_exp * student_exp).sum(dim=-1)

        teacher_normed = teacher_normed.unsqueeze(1)
        student_normed = student_normed.unsqueeze(0)

        cosine_similarities = dot_product / (teacher_normed * student_normed + 1e-8)
        return cosine_similarities.mean()


class PSNRMetric(BaseMetric):
    """
    Calculates PSNR Metric
    """

    def __init__(self):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

    def compute(
        self, *, preds_patches: torch.Tensor, targets_patches: torch.Tensor, **kwargs
    ) -> float:
        self.psnr.reset()
        self.psnr.update(preds_patches, targets_patches)
        return self.psnr.compute()


class SSIMMetric(BaseMetric):
    """
    Calculates SSIM Metric
    """

    def __init__(self):
        self.ssim = SSIM(data_range=1.0)

    def compute(
        self, *, preds_patches: torch.Tensor, targets_patches: torch.Tensor, **kwargs
    ) -> float:
        self.ssim.update((preds_patches, targets_patches))
        return self.ssim.compute()


class Accuracy(BaseMetric):
    """
    Calculates Accuracy Metric
    """

    def compute(self, *, correct: int, total: int, **kwargs) -> float:
        return correct / total
