from flask import Flask, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import tempfile
import os
import io

app = Flask(__name__)
CORS(app)


results = None

plt.switch_backend('Agg')

def train_kmeans():
    global results

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df = df[['sepal length (cm)', 'sepal width (cm)']]

    scaler = MinMaxScaler()
    df[['sepal length (cm)', 'sepal width (cm)']] = scaler.fit_transform(df[['sepal length (cm)', 'sepal width (cm)']])

    K = 3
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(df)
    df['Cluster'] = kmeans.labels_

    sse = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df[['sepal length (cm)', 'sepal width (cm)']])
        sse.append(kmeans.inertia_)

    temp_dir = tempfile.gettempdir()
    elbow_plot_path = os.path.join(temp_dir, 'elbow_plot.png')

    plt.figure()
    plt.plot(range(1, 10), sse)
    plt.xlabel('Número de Clusters')
    plt.ylabel('Soma dos Erros Quadrados (SSE)')
    plt.title('Método do Cotovelo')
    plt.savefig(elbow_plot_path)

    silhouette_avg = silhouette_score(df[['sepal length (cm)', 'sepal width (cm)']], df['Cluster'])
    print(f'Silhouette Score para K={K}: {silhouette_avg}')

    results = {
        'silhouette_score': silhouette_avg,
        'elbow_plot_path': elbow_plot_path
    }

    return jsonify({'silhouette_score': silhouette_avg})

@app.route('/k/run', methods=['POST'])
def run():
    return train_kmeans()
    

@app.route('/k/image', methods=['GET'])
def get_kmeans_image():
    if results is None or 'elbow_plot_path' not in results:
        return jsonify({"error": "Nenhum resultado disponível. Execute o treinamento do K-means primeiro."}), 400

    elbow_plot_path = results['elbow_plot_path']

    # Verificar se a imagem ainda existe
    if os.path.exists(elbow_plot_path):
        return send_file(elbow_plot_path, mimetype='image/png')
    else:
        return jsonify({"error": "Imagem não encontrada. Execute o treinamento do K-means novamente."}), 404

if __name__ == '__main__':
    app.run(debug=True)
