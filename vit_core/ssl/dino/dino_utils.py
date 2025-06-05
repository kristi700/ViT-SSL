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
