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
CORS(app)  # Configuração do CORS para aceitar qualquer origem e qualquer método

# Variável global para armazenar os resultados
results = None

plt.switch_backend('Agg')

def train_kmeans():
    global results

    # Carregar e processar o dataset predefinido
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df = df[['sepal length (cm)', 'sepal width (cm)']]  # Selecionar apenas duas features para simplificar

    # Pré-processamento dos dados
    scaler = MinMaxScaler()
    df[['sepal length (cm)', 'sepal width (cm)']] = scaler.fit_transform(df[['sepal length (cm)', 'sepal width (cm)']])

    # Definir o número de clusters (K=3) e executar o KMeans
    K = 3
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(df)
    df['Cluster'] = kmeans.labels_

    # Método do Cotovelo para sugerir o número ideal de clusters
    sse = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df[['sepal length (cm)', 'sepal width (cm)']])
        sse.append(kmeans.inertia_)

    # Salvar o gráfico do Método do Cotovelo em um diretório temporário
    temp_dir = tempfile.gettempdir()
    elbow_plot_path = os.path.join(temp_dir, 'elbow_plot.png')

    plt.figure()
    plt.plot(range(1, 10), sse)
    plt.xlabel('Número de Clusters')
    plt.ylabel('Soma dos Erros Quadrados (SSE)')
    plt.title('Método do Cotovelo')
    plt.savefig(elbow_plot_path)

    # Avaliação com Índice de Silhueta para K=3
    silhouette_avg = silhouette_score(df[['sepal length (cm)', 'sepal width (cm)']], df['Cluster'])
    print(f'Silhouette Score para K={K}: {silhouette_avg}')

    # Armazenando resultados
    results = {
        'silhouette_score': silhouette_avg,
        'elbow_plot_path': elbow_plot_path  # Armazena o caminho da imagem temporária
    }

    return jsonify({'silhouette_score': silhouette_avg})

# POST para realizar treinamentos e salvar a imagem no diretório temporário
@app.route('/k/run', methods=['POST'])
def run():
    return train_kmeans()
    

# GET para obter a imagem do método do cotovelo
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

# Inicializa servidor
if __name__ == '__main__':
    app.run(debug=True)
