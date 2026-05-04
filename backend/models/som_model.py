from dataclasses import dataclass
from typing import Optional


@dataclass
class SOMResult:
    manual_accuracy: Optional[float]
    minisom_accuracy: Optional[float]
    manual_image_path: Optional[str]
    minisom_image_path: Optional[str]
