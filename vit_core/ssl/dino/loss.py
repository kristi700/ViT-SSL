import torch

from torch import nn
from torch.nn.functional import softmax, log_softmax

class DINOLoss(nn.Module):
    def __init__(self, teacher_temp: float, student_temp: float):
        super().__init__()
        self.teacher_temp = teacher_temp # TODO - add scheduling for this!
        self.student_temp = student_temp

    def forward(self, teacher_output: torch.Tensor, student_output: torch.Tensor, center: torch.Tensor):
        """
        Implemented as described in https://arxiv.org/pdf/2104.14294 - Algorithm 1.
        """
        teacher_output = teacher_output.detach() # stop gradient
        student_probs = log_softmax(student_output / self.student_temp, dim=1)
        teacher_probs = softmax((teacher_output - center) / self.teacher_temp, dim=1)
        
        teacher_probs = teacher_probs.unsqueeze(1)
        student_probs = student_probs.unsqueeze(0)

        return -(teacher_probs*student_probs).sum(dim=1).mean()
