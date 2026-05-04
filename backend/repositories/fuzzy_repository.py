from typing import Optional
from models.fuzzy_model import NeuroFuzzyResult


class FuzzyRepository:
    def __init__(self):
        self._nf_result: Optional[NeuroFuzzyResult] = None

    def store_nf(self, result: NeuroFuzzyResult) -> None:
        self._nf_result = result

    def get_nf(self) -> Optional[NeuroFuzzyResult]:
        return self._nf_result
