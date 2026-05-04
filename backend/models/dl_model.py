from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class DLResult:
    confusion_matrix_tf: Optional[Any] = None
    confusion_matrix_pt: Optional[Any] = None
    accuracy_tf: Optional[float] = None
    accuracy_pt: Optional[float] = None
