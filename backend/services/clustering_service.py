import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import skfuzzy as fuzz

from models.clustering_model import KMeansResult, CMeansResult

plt.switch_backend('Agg')


class ClusteringService:
    def train_kmeans(self) -> KMeansResult:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df = df[['sepal length (cm)', 'sepal width (cm)']]

        scaler = MinMaxScaler()
        df[['sepal length (cm)', 'sepal width (cm)']] = scaler.fit_transform(
            df[['sepal length (cm)', 'sepal width (cm)']]
        )

        K = 3
        kmeans = KMeans(n_clusters=K, random_state=0)
        kmeans.fit(df)
        df['Cluster'] = kmeans.labels_

        sse = []
        for k in range(1, 10):
            km = KMeans(n_clusters=k, random_state=0)
            km.fit(df[['sepal length (cm)', 'sepal width (cm)']])
            sse.append(km.inertia_)

        temp_dir = tempfile.gettempdir()
        elbow_plot_path = os.path.join(temp_dir, 'elbow_plot.png')

        plt.figure()
        plt.plot(range(1, 10), sse)
        plt.xlabel('Número de Clusters')
        plt.ylabel('Soma dos Erros Quadrados (SSE)')
        plt.title('Método do Cotovelo')
        plt.savefig(elbow_plot_path)
        plt.close()

        silhouette_avg = silhouette_score(df[['sepal length (cm)', 'sepal width (cm)']], df['Cluster'])
        print(f'Silhouette Score para K={K}: {silhouette_avg}')

        return KMeansResult(silhouette_score=silhouette_avg, elbow_plot_path=elbow_plot_path)

    def _calculate_fpc(self, u) -> float:
        n, c = u.shape
        fpc = np.sum(u**2) / n
        return fpc

    def train_cmeans(self) -> CMeansResult:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df = df[['sepal length (cm)', 'sepal width (cm)']]

        scaler = MinMaxScaler()
        df[['sepal length (cm)', 'sepal width (cm)']] = scaler.fit_transform(
            df[['sepal length (cm)', 'sepal width (cm)']]
        )

        data = df.values.T
        c = 3
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, c=c, m=2.0, error=0.005, maxiter=1000)

        fpc = self._calculate_fpc(u.T)
        print(f"Índice de Partição Fuzzy (FPC): {fpc}")

        temp_dir = tempfile.gettempdir()
        cmeans_plot_path = os.path.join(temp_dir, 'fuzzy_cmeans_plot.png')

        plt.figure()
        cluster_membership = np.argmax(u, axis=0)
        for j in range(c):
            plt.scatter(df.iloc[cluster_membership == j, 0],
                        df.iloc[cluster_membership == j, 1], label=f"Cluster {j+1}")

        for pt in cntr:
            plt.plot(pt[0], pt[1], 'rs', markersize=10, label="Centro do Cluster")

        plt.xlabel('Comprimento da Sépala')
        plt.ylabel('Largura da Sépala')
        plt.title('Clusters Fuzzy C-Means - Iris Dataset')
        plt.legend()
        plt.savefig(cmeans_plot_path)
        plt.close()

        return CMeansResult(fpc=fpc, cmeans_plot_path=cmeans_plot_path)
