import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from models.som_model import SOMResult

plt.switch_backend('Agg')


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def find_bmu(som, sample):
    bmu_idx = np.argmin([euclidean_distance(neuron, sample) for neuron in som.reshape(-1, som.shape[2])])
    return np.unravel_index(bmu_idx, som.shape[:2])


def update_weights(som, sample, bmu_idx, iteration, max_iterations, learning_rate, radius):
    learning_rate_decay = learning_rate * np.exp(-iteration / max_iterations)
    radius_decay = radius * np.exp(-iteration / max_iterations)

    for i in range(som.shape[0]):
        for j in range(som.shape[1]):
            distance_to_bmu = euclidean_distance(np.array([i, j]), np.array(bmu_idx))
            if distance_to_bmu <= radius_decay:
                influence = np.exp(-distance_to_bmu**2 / (2 * (radius_decay ** 2)))
                som[i, j, :] += influence * learning_rate_decay * (sample - som[i, j, :])


def assign_labels_to_neurons(som, X_scaled, y):
    som_labels = np.zeros((som.shape[0], som.shape[1]), dtype=int)
    som_bmu_count = np.zeros((som.shape[0], som.shape[1]), dtype=int)

    for i, sample in enumerate(X_scaled):
        bmu_idx = find_bmu(som, sample)
        som_labels[bmu_idx] += y[i]
        som_bmu_count[bmu_idx] += 1

    for i in range(som.shape[0]):
        for j in range(som.shape[1]):
            if som_bmu_count[i, j] > 0:
                neuron_label_counts = Counter([y[k] for k, sample in enumerate(X_scaled) if find_bmu(som, sample) == (i, j)])
                som_labels[i, j] = neuron_label_counts.most_common(1)[0][0]

    return som_labels


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


class SOMService:
    def train(self) -> SOMResult:
        iris = load_iris()
        X = iris.data
        y = iris.target
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        som_grid_size = 7
        som = np.random.rand(som_grid_size, som_grid_size, X.shape[1])

        accuracy_manual, manual_image_path = self._train_manual(X_scaled, y, som)
        minisom_image_path, accuracy_minisom = self._train_minisom(X_scaled, y)

        return SOMResult(
            manual_accuracy=accuracy_manual,
            minisom_accuracy=accuracy_minisom,
            manual_image_path=manual_image_path,
            minisom_image_path=minisom_image_path
        )

    def _train_manual(self, X_scaled, y, som, max_iterations=500, learning_rate=0.5, radius=3):
        for iteration in range(max_iterations):
            for sample in X_scaled:
                bmu_idx = find_bmu(som, sample)
                update_weights(som, sample, bmu_idx, iteration, max_iterations, learning_rate, radius)

        som_labels = assign_labels_to_neurons(som, X_scaled, y)
        accuracy = calculate_accuracy(som, X_scaled, y, som_labels)

        plt.figure(figsize=(10, 10))
        colors = ['r', 'g', 'b']
        markers = ['o', 's', 'D']

        for i, sample in enumerate(X_scaled):
            bmu_idx = find_bmu(som, sample)
            plt.plot(bmu_idx[0] + 0.5, bmu_idx[1] + 0.5, markers[y[i]], markerfacecolor='None',
                     markeredgecolor=colors[y[i]], markersize=12, markeredgewidth=2)

        plt.xticks(np.arange(som.shape[0] + 1))
        plt.yticks(np.arange(som.shape[1] + 1))
        plt.grid()

        for i, flower_type in enumerate(["Setosa", "Versicolor", "Virginica"]):
            plt.plot([], [], color=colors[i], marker=markers[i], label=flower_type,
                     linestyle='None', markerfacecolor='None', markeredgewidth=2)

        plt.legend()

        temp_dir = tempfile.gettempdir()
        manual_som_image_path = os.path.join(temp_dir, 'manual_som_plot.png')
        plt.savefig(manual_som_image_path)
        plt.close()

        return accuracy, manual_som_image_path

    def _train_minisom(self, X_scaled, y):
        try:
            som = MiniSom(x=7, y=7, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
            som.random_weights_init(X_scaled)
            som.train_random(X_scaled, num_iteration=500)

            temp_dir = tempfile.gettempdir()
            minisom_image_path = os.path.join(temp_dir, 'minisom_som_plot.png')

            plt.figure(figsize=(10, 10))
            colors = ['r', 'g', 'b']
            markers = ['o', 's', 'D']

            for i, x in enumerate(X_scaled):
                winner = som.winner(x)
                plt.plot(winner[0] + 0.5, winner[1] + 0.5, markers[y[i]], markerfacecolor='None',
                         markeredgecolor=colors[y[i]], markersize=12, markeredgewidth=2)

            plt.xticks(np.arange(8))
            plt.yticks(np.arange(8))
            plt.grid()

            for i, flower_type in enumerate(["Setosa", "Versicolor", "Virginica"]):
                plt.plot([], [], color=colors[i], marker=markers[i], label=flower_type,
                         linestyle='None', markerfacecolor='None', markeredgewidth=2)

            plt.legend()
            plt.savefig(minisom_image_path)
            plt.close()

            som_labels = assign_labels_to_neurons(som.get_weights(), X_scaled, y)
            accuracy_minisom = calculate_accuracy(som.get_weights(), X_scaled, y, som_labels)

            return minisom_image_path, accuracy_minisom
        except Exception as e:
            print(f"Error in training MiniSom: {e}")
            return None, None
