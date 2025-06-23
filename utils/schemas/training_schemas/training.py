from typing import Optional, List
from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    name: str
    params: dict


@dataclass
class SchedulerConfig:
    main: dict
    warmup: dict


@dataclass
class CriterionConfig:
    name: str
    params: dict


@dataclass
class TrainingConfig:
    type: str
    student_temp: Optional[float]
    teacher_temp: Optional[float]
    teacher_temp_final: Optional[float] = None
    teacher_temp_scheduler: Optional[str] = ""
    teacher_momentum_start: Optional[float]
    teacher_momentum_final: Optional[float]
    num_all_views: Optional[int]
    num_global_views: Optional[int]
    resume_from_checkpoint: Optional[str]
    random_seed: int
    batch_size: int
    num_epochs: int
    warmup_initial_learning_rate: float
    warmup_final_learning_rate: float
    warmup_epochs: int
    lr_final: float
    weight_decay: float
    criterion: Optional[CriterionConfig]
    optimizer: OptimizerConfig
    lr_scheduler: SchedulerConfig
