from typing import Optional
from models.dl_model import DLResult


class DLRepository:
    def __init__(self):
        self._result: Optional[DLResult] = None

    def store(self, result: DLResult) -> None:
        self._result = result

    def get(self) -> Optional[DLResult]:
        return self._result
