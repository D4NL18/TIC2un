from dataclasses import dataclass
from typing import Any


@dataclass
class NeuroFuzzyResult:
    mse: float
    mae: float
    y_true: Any
    y_pred: Any
