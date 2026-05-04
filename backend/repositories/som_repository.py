from typing import Optional
from models.som_model import SOMResult


class SOMRepository:
    def __init__(self):
        self._result: Optional[SOMResult] = None

    def store(self, result: SOMResult) -> None:
        self._result = result

    def get(self) -> Optional[SOMResult]:
        return self._result
