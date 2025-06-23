import math


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

class DINOTeacherTempScheduler:
    def __init__(
        self,
        temp_start: float,
        temp_end: float,
        total_iters: int,
        schedule_type: str = "cosine",
    ):
        self.t_start = temp_start
        self.t_end = temp_end
        self.total_iters = total_iters
        self.schedule_type = schedule_type

    def get_temp(self, current_step: int) -> float:
        if current_step >= self.total_iters:
            return self.t_end
        progress = current_step / self.total_iters
        if self.schedule_type == "linear":
            return self.t_start + (self.t_end - self.t_start) * progress
        cos_term = math.cos(math.pi * progress)
        return self.t_end - (self.t_end - self.t_start) * 0.5 * (1 + cos_term)