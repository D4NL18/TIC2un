from flask import Flask, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Variável global para armazenar o caminho da imagem e os resultados
results = None

plt.switch_backend('Agg')  # Usado para suportar geração de gráficos sem GUI

# Função para calcular o Fuzzy Partition Coefficient (FPC)
def calculate_fpc(u):
    # Número de pontos e clusters
    n, c = u.shape
    # FPC = 1/n * soma dos quadrados dos elementos de u
    fpc = np.sum(u**2) / n
    return fpc

def train_fuzzy_cmeans():
    global results

    # Carregar e processar o conjunto de dados Iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df = df[['sepal length (cm)', 'sepal width (cm)']]  # Usar apenas duas features

    # Pré-processamento dos dados - Normalização
    scaler = MinMaxScaler()
    df[['sepal length (cm)', 'sepal width (cm)']] = scaler.fit_transform(df[['sepal length (cm)', 'sepal width (cm)']])

    # Converter DataFrame para matriz para compatibilidade com skfuzzy
    data = df.values.T

    # Configuração do Fuzzy C-Means
    c = 3  # Número de clusters
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, c=c, m=2.0, error=0.005, maxiter=1000)

    # Calcular o Índice de Partição Fuzzy (FPC)
    fpc = calculate_fpc(u)
    print(f"Índice de Partição Fuzzy (FPC): {fpc}")

    # Gerar gráfico dos clusters e salvar em um diretório temporário
    temp_dir = tempfile.gettempdir()
    cmeans_plot_path = os.path.join(temp_dir, 'fuzzy_cmeans_plot.png')

    # Plotar clusters
    plt.figure()
    cluster_membership = np.argmax(u, axis=0)
    for j in range(c):
        plt.scatter(df.iloc[cluster_membership == j, 0],
                    df.iloc[cluster_membership == j, 1], label=f"Cluster {j+1}")

    # Plotar os centros dos clusters
    for pt in cntr:
        plt.plot(pt[0], pt[1], 'rs', markersize=10, label="Centro do Cluster")

    plt.xlabel('Comprimento da Sépala')
    plt.ylabel('Largura da Sépala')
    plt.title('Clusters Fuzzy C-Means - Iris Dataset')
    plt.legend()
    plt.savefig(cmeans_plot_path)

    # Armazenar resultados
    results = {
        'fpc': fpc,
        'cmeans_plot_path': cmeans_plot_path
    }

    return jsonify({
        'message': 'Fuzzy C-Means trained successfully',
        'fpc': fpc
    })

# POST para realizar o treinamento e salvar a imagem
@app.route('/c/run', methods=['POST'])
def run():
    return train_fuzzy_cmeans()

# GET para obter a imagem dos clusters
@app.route('/c/image', methods=['GET'])
def get_cmeans_image():
    if results is None or 'cmeans_plot_path' not in results:
        return jsonify({"error": "Nenhum resultado disponível. Execute o treinamento do Fuzzy C-Means primeiro."}), 400

    cmeans_plot_path = results['cmeans_plot_path']

    # Verificar se a imagem ainda existe
    if os.path.exists(cmeans_plot_path):
        return send_file(cmeans_plot_path, mimetype='image/png')
    else:
        return jsonify({"error": "Imagem não encontrada. Execute o treinamento do Fuzzy C-Means novamente."}), 404

# Inicializar servidor
if __name__ == '__main__':
    app.run(debug=True)
