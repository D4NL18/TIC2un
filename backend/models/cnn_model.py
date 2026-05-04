from dataclasses import dataclass


@dataclass
class CNNResult:
    accuracy: float
    f1_score: float
