from dataclasses import dataclass


@dataclass
class EvalConfig:
    dataset_name: str
    data_dir: str
    data_csv: str
    num_classes: int
    mode: str
    experiment_path: str
