from typing import Optional
from models.clustering_model import KMeansResult, CMeansResult


class ClusteringRepository:
    def __init__(self):
        self._kmeans_result: Optional[KMeansResult] = None
        self._cmeans_result: Optional[CMeansResult] = None

    def store_kmeans(self, result: KMeansResult) -> None:
        self._kmeans_result = result

    def get_kmeans(self) -> Optional[KMeansResult]:
        return self._kmeans_result

    def store_cmeans(self, result: CMeansResult) -> None:
        self._cmeans_result = result

    def get_cmeans(self) -> Optional[CMeansResult]:
        return self._cmeans_result
