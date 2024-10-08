from flask import Flask, jsonify, send_file
from flask_cors import CORS
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import tempfile
from collections import Counter

app = Flask(__name__)
CORS(app)

# Calcula distância euclidiana
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# Encontra o neurônio vencedor
def find_bmu(som, sample):
    bmu_idx = np.argmin([euclidean_distance(neuron, sample) for neuron in som.reshape(-1, som.shape[2])])
    return np.unravel_index(bmu_idx, som.shape[:2])

# Atualiza os pesos dos neurônios
def update_weights(som, sample, bmu_idx, iteration, max_iterations, learning_rate, radius):
    learning_rate_decay = learning_rate * np.exp(-iteration / max_iterations)
    radius_decay = radius * np.exp(-iteration / max_iterations)

    for i in range(som.shape[0]):
        for j in range(som.shape[1]):
            distance_to_bmu = euclidean_distance(np.array([i, j]), np.array(bmu_idx))
            if distance_to_bmu <= radius_decay:
                influence = np.exp(-distance_to_bmu**2 / (2 * (radius_decay ** 2)))
                som[i, j, :] += influence * learning_rate_decay * (sample - som[i, j, :])

# Atribui rótulos aos neurônios com base nas amostras associadas
def assign_labels_to_neurons(som, X_scaled, y):
    som_labels = np.zeros((som.shape[0], som.shape[1]), dtype=int)
    som_bmu_count = np.zeros((som.shape[0], som.shape[1]), dtype=int)

    for i, sample in enumerate(X_scaled):
        bmu_idx = find_bmu(som, sample)
        som_labels[bmu_idx] += y[i]
        som_bmu_count[bmu_idx] += 1

    # Normalizar os rótulos pela maioria
    for i in range(som.shape[0]):
        for j in range(som.shape[1]):
            if som_bmu_count[i, j] > 0:
                neuron_label_counts = Counter([y[k] for k, sample in enumerate(X_scaled) if find_bmu(som, sample) == (i, j)])
                som_labels[i, j] = neuron_label_counts.most_common(1)[0][0]  # Atribuir rótulo da classe majoritária

    return som_labels

# Calcula a acurácia do SOM
def calculate_accuracy(som, X_scaled, y, som_labels):
    correct = 0

    for i, sample in enumerate(X_scaled):
        bmu_idx = find_bmu(som, sample)
        predicted_label = som_labels[bmu_idx]
        true_label = y[i]
        if predicted_label == true_label:
            correct += 1

    accuracy = correct / len(y)
    return accuracy

# Treinar SOM
def train_som(X, som, max_iterations=1000, learning_rate=0.5, radius=3):
    for iteration in range(max_iterations):
        for sample in X:
            bmu_idx = find_bmu(som, sample)
            update_weights(som, sample, bmu_idx, iteration, max_iterations, learning_rate, radius)

@app.route('/train-som', methods=['POST'])
def train_som_endpoint():
    # Carregar dataset e normalização dos dados
    iris = load_iris()
    X = iris.data
    y = iris.target
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Inicializar SOM
    som_grid_size = 7
    som = np.random.rand(som_grid_size, som_grid_size, X.shape[1])

    # Treinar SOM
    train_som(X_scaled, som, max_iterations=500, learning_rate=0.5, radius=3)

    # Atribuir rótulos aos neurônios
    som_labels = assign_labels_to_neurons(som, X_scaled, y)

    # Calcular acurácia
    accuracy = calculate_accuracy(som, X_scaled, y, som_labels)

    # Plot
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

    # Adicionar legenda para cada classe de flor
    for i, flower_type in enumerate(["Setosa", "Versicolor", "Virginica"]):
        plt.plot([], [], color=colors[i], marker=markers[i], label=flower_type,
                 linestyle='None', markerfacecolor='None', markeredgewidth=2)

    plt.legend()

    # Salvar gráfico
    try:
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, 'som_plot.png')
        plt.savefig(image_path)
    except Exception as e:
        print(f"Erro ao salvar a imagem: {e}")
        return jsonify({"error": "Falha ao salvar a imagem."}), 500

    return jsonify({"message": "SOM trained successfully", "image_path": image_path, "accuracy": accuracy})

@app.route('/get-image', methods=['GET'])
def get_image():
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, 'som_plot.png')
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    return jsonify({"error": "Image not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
