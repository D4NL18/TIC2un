from typing import Optional
from models.svm_model import SVMResult


class SVMRepository:
    def __init__(self):
        self._result: Optional[SVMResult] = None

    def store(self, result: SVMResult) -> None:
        self._result = result

    def get(self) -> Optional[SVMResult]:
        return self._result
