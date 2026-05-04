from dataclasses import dataclass


@dataclass
class KMeansResult:
    silhouette_score: float
    elbow_plot_path: str


@dataclass
class CMeansResult:
    fpc: float
    cmeans_plot_path: str
