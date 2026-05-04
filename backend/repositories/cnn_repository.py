from typing import Optional
from models.cnn_model import CNNResult


class CNNRepository:
    def __init__(self):
        self._cnn_result: Optional[CNNResult] = None
        self._ft_result: Optional[CNNResult] = None

    def store_cnn(self, result: CNNResult) -> None:
        self._cnn_result = result

    def get_cnn(self) -> Optional[CNNResult]:
        return self._cnn_result

    def store_ft(self, result: CNNResult) -> None:
        self._ft_result = result

    def get_ft(self) -> Optional[CNNResult]:
        return self._ft_result
