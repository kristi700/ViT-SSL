from enum import Enum
from typing import List
from .data import DataConfig
from .model import ModelConfig
from omegaconf import OmegaConf
from .training import TrainingConfig
from .transforms import TransformConfig
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


class MetricName(str, Enum):
    CenterNorm = "CenterNorm"
    TeacherMean = "TeacherMean"
    TeacherSTD = "TeacherSTD"
    TeacherVar = "TeacherVar"
    StudentMean = "StudentMean"
    StudentSTD = "StudentSTD"
    StudentVar = "StudentVar"
    CosineSim = "CosineSim"
    Accuracy = "Accuracy"
    PSNR = "PSNR"
    SSIM = "SSIM"


@dataclass
class TransformsConfig:
    globals: List[TransformConfig] = field(default_factory=list)
    locals: List[TransformConfig] = field(default_factory=list)
    train: List[TransformConfig] = field(default_factory=list)
    val: List[TransformConfig] = field(default_factory=list)


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    transforms: TransformsConfig
    metrics: List[MetricName] = field(default_factory=list)


cs = ConfigStore.instance()
cs.store(name="training_config", node=TrainConfig)