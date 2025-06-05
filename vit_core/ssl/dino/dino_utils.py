import math
import torch

from typing import List

class DINOMomentumScheduler:
    def __init__(self, m_start: float, m_end: float, total_iters: int):
        self.m_start = m_start
        self.m_end = m_end
        self.total_iters = total_iters

    def get_momentum(self, current_step: int) -> float:
        if current_step >= self.total_iters:
            return self.m_end
        cos_term = math.cos(math.pi * current_step / self.total_iters)
        return self.m_end - (self.m_end - self.m_start) * 0.5 * (1 + cos_term)


def cosine_similarity(teacher_output: torch.Tensor, student_output: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Computes cosine similarity between the teacher's outputs and the student's.
        args:
            - teacher_output - num_global_views * batch_size * dim
            - student_output - num_all_views * batch_size * dim

    Implemented as in: https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
    """
    teacher_normed = torch.linalg.norm(teacher_output, dim=-1)
    student_normed = torch.linalg.norm(student_output, dim=-1)

    teacher_exp = teacher_output.unsqueeze(1)
    student_exp = student_output.unsqueeze(0)

    dot_product = (teacher_exp*student_exp).sum(dim=-1)

    teacher_normed = teacher_normed.unsqueeze(1)
    student_normed = student_normed.unsqueeze(0)

    cosine_similarities = dot_product / (teacher_normed*student_normed + eps)
    return cosine_similarities.mean()

def embedding_distribution(model_output: torch.Tensor) -> List[torch.Tensor]:
    """
    Calculates the given inputs mean, std and variance
        args:
            - model_output - num_views * batch_size * dim
    """
    flat = torch.flatten(model_output)
    return [flat.mean(), flat.std(), flat.var()]

def center_norm(center: torch.Tensor):
    """
    Calculates the L2 norm of center
    """
    return torch.linalg.norm(center)

