from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TransformConfig:
    name: str
    params: Optional[Dict] = None