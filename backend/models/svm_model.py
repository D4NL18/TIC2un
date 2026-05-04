from dataclasses import dataclass


@dataclass
class SVMResult:
    accuracy: float
    confusion_matrix: list
    classification_report: str
    image_data: bytes
