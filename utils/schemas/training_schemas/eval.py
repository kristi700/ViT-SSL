from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EvalConfig:
    dataset_name: Optional[str]
    data_dir: Optional[str]
    data_csv: Optional[str]
    num_classes: Optional[int]
    mode: Optional[List[str]]
    save_confusion_matrix: Optional[bool]
    interval: int
