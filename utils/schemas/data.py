from typing import Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_name: str
    data_dir: str
    data_csv: Optional[str]
    val_split: float
    num_workers: int
    img_size: int
