from flask import Flask, jsonify, send_file
from flask_cors import CORS
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import tempfile

app = Flask(__name__)
CORS(app)

#Calcula distância euclidiana
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

#Encontra o neurônio vencedor
def find_bmu(som, sample):
    bmu_idx = np.argmin([euclidean_distance(neuron, sample) for neuron in som.reshape(-1, som.shape[2])])
    return np.unravel_index(bmu_idx, som.shape[:2])

#Atualiza os pesos dos neurônios
def update_weights(som, sample, bmu_idx, iteration, max_iterations, learning_rate, radius):
    learning_rate_decay = learning_rate * np.exp(-iteration / max_iterations)
    radius_decay = radius * np.exp(-iteration / max_iterations)

    for i in range(som.shape[0]):
        for j in range(som.shape[1]):
            distance_to_bmu = euclidean_distance(np.array([i, j]), np.array(bmu_idx))
            if distance_to_bmu <= radius_decay:
                influence = np.exp(-distance_to_bmu**2 / (2 * (radius_decay ** 2)))
                som[i, j, :] += influence * learning_rate_decay * (sample - som[i, j, :])

#Treinar SOM
def train_som(X, som, max_iterations=1000, learning_rate=0.5, radius=3):
    for iteration in range(max_iterations):
        for sample in X:
            bmu_idx = find_bmu(som, sample)
            update_weights(som, sample, bmu_idx, iteration, max_iterations, learning_rate, radius)

@app.route('/train-som', methods=['POST'])
def train_som_endpoint():
    #Carregar dataset e normalização dos dados
    iris = load_iris()
    X = iris.data
    y = iris.target
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    #Inicializar SOM
    som_grid_size = 7
    som = np.random.rand(som_grid_size, som_grid_size, X.shape[1])

    #Treinar SOM
    train_som(X_scaled, som, max_iterations=500, learning_rate=0.5, radius=3)

    #Plot
    plt.figure(figsize=(10, 10))
    colors = ['r', 'g', 'b']
    markers = ['o', 's', 'D']

    for i, sample in enumerate(X_scaled):
        bmu_idx = find_bmu(som, sample)
        plt.plot(bmu_idx[0] + 0.5, bmu_idx[1] + 0.5, markers[y[i]], markerfacecolor='None',
                 markeredgecolor=colors[y[i]], markersize=12, markeredgewidth=2)

    plt.xticks(np.arange(som_grid_size + 1))
    plt.yticks(np.arange(som_grid_size + 1))
    plt.grid()

    #Salvar gráfico
    try:
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, 'som_plot.png')  # Salvar no diretório temporário
        plt.savefig(image_path)  # Tente salvar a imagem
    except Exception as e:
        print(f"Erro ao salvar a imagem: {e}")
        return jsonify({"error": "Falha ao salvar a imagem."}), 500

    return jsonify({"message": "SOM trained successfully", "image_path": image_path})

@app.route('/get-image', methods=['GET'])
def get_image():
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, 'som_plot.png')
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    return jsonify({"error": "Image not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
