from typing import List
from .eval import EvalConfig
from omegaconf import OmegaConf
from .transforms import TransformConfig
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class TransformsConfig:
    globals: List[TransformConfig] = field(default_factory=list)
    locals: List[TransformConfig] = field(default_factory=list)
    train: List[TransformConfig] = field(default_factory=list)
    val: List[TransformConfig] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    eval: EvalConfig
    transforms: TransformsConfig


cs = ConfigStore.instance()
cs.store(name="evaluation_config", node=EvaluationConfig)
