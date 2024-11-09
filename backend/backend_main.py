from flask import Flask, jsonify, send_file
from flask_cors import CORS
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import tempfile
from collections import Counter
from minisom import MiniSom

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# ====================================================== SOM ===========================================================

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

# Atribui labels
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

# Accuracy
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

# Treinar SOM manual
def train_som(X_scaled, y, som, max_iterations=1000, learning_rate=0.5, radius=3):
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

    # Salvar imagem
    temp_dir = tempfile.gettempdir()
    manual_som_image_path = os.path.join(temp_dir, 'manual_som_plot.png')
    plt.savefig(manual_som_image_path)
    plt.close()

    return accuracy, manual_som_image_path

# Treinar SOM usando MiniSom
def train_minisom(X_scaled, y):
    try:
        # Inicializa MiniSom
        som = MiniSom(x=7, y=7, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
        som.random_weights_init(X_scaled)
        som.train_random(X_scaled, num_iteration=500)

        # Criação do gráfico MiniSom
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
        
        # Adiciona legenda para cada classe de flor
        for i, flower_type in enumerate(["Setosa", "Versicolor", "Virginica"]):
            plt.plot([], [], color=colors[i], marker=markers[i], label=flower_type,
                     linestyle='None', markerfacecolor='None', markeredgewidth=2)
        
        plt.legend()
        plt.savefig(minisom_image_path)
        plt.close()

        # Calcula a precisão do MiniSom
        som_labels = assign_labels_to_neurons(som.get_weights(), X_scaled, y)
        accuracy_minisom = calculate_accuracy(som.get_weights(), X_scaled, y, som_labels)

        return minisom_image_path, accuracy_minisom
    except Exception as e:
        print(f"Error in training MiniSom: {e}")
        return None, None

# Armazena a precisão dos SOMs
accuracy_store = {}


# ====================================================== SVM ===========================================================

# Passo 1: Imports
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import io
import os

# Passo 7: Importar o Modelo SVM
from sklearn.svm import SVC


results = None

plt.switch_backend('Agg')

def run_svm():
    global results

    # Passo 2: Escolher a Base de Dados
    iris = load_iris()

    # Passo 3: Carregar os Dados
    X = iris.data
    y = iris.target

    # Passo 4: Analisar os Dados
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y
    print(df.head())
    print(df.info())

    # Passo 5: Pré-processamento dos Dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Passo 6: Separar os Dados em Conjuntos de Treinamento e Teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Passo 8: Criar uma Instância do Modelo
    svm_model = SVC()

    # Passo 9: Treinar o Modelo
    svm_model.fit(X_train, y_train)

    # Passo 10: Fazer Previsões
    y_pred = svm_model.predict(X_test)

    # Passo 11: Avaliar o Modelo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)

    # Passo 12: Visualizar a Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')

    # Salvar imagem para enviar no get
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close() 

    # Passo 13: Ajustar Hiperparâmetros
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto'] 
    }
    grid_search = GridSearchCV(svm_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_svm_model = grid_search.best_estimator_

    # Passo 14: Documentar e Salvar o Modelo
    backup_dir = 'backend/backup_svm'

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    model_filename = os.path.join(backup_dir, 'best_svm_model.joblib')
    joblib.dump(best_svm_model, model_filename)
    print(f"Modelo treinado salvo como {model_filename}")

    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': classification_rep,
        'image_data': img_bytes.getvalue()
    }



# ====================================================== DL ===========================================================

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import matplotlib.pyplot as plt
import threading

iris = load_iris()
X = iris.data 
y = iris.target

confusion_matrix_tf = None
confusion_matrix_pt = None
accuracy_tf = None
accuracy_pt = None

# Preparação dos dados
def prepare_data(X, y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Train TensorFlow
def train_tf(X_train, y_train, X_test, y_test):
    global confusion_matrix_tf, accuracy_tf
    
    # Definir modelo
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(10, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, batch_size=5, validation_split=0.2)

    # Avaliação
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    accuracy_tf = test_accuracy

    # Previsões e matriz de confusão
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    confusion_matrix_tf = confusion_matrix(y_test, y_pred_classes)

# Treinamento PyTorch
def train_pt(X_train, y_train, X_test, y_test):
    global confusion_matrix_pt, accuracy_pt

    # Convertendo para tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    class IrisModel(nn.Module):
        def __init__(self):
            super(IrisModel, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, 3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = IrisModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Treinamento
    for epoch in range(100):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Avaliação
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        accuracy_pt = (predicted == y_test_tensor).float().mean().item()
        confusion_matrix_pt = confusion_matrix(y_test, predicted.numpy())

# Plot matriz confusão
def create_confusion_matrix_image(conf_matrix):
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')

    thresh = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), 
                 ha='center', va='center',
                 color='white' if conf_matrix[i, j] > thresh else 'black')

    # Salvar a imagem em um arquivo temporário
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, 'confusion_matrix.png')
    plt.savefig(image_path)
    plt.close()
    return image_path


# ====================================================== CNN Tensorflow ===========================================================

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow import keras
from keras import datasets, models, layers, applications, utils
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

# Diretórios
model_dir = 'backend/Roteiros_Individuais/CNN_tf/models'
image_dir = 'Roteiros_Individuais/CNN_tf/images'
os.makedirs(image_dir, exist_ok=True)

# Carregar modelo já treinado no CNN_treino.py
def load_model():
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_path = os.path.join(model_dir, 'melhor_modelo_cnn_pesos.weights.h5')
    model.load_weights(model_path)
    return model

# ====================================================== CNN Finetunning ===========================================================

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow import keras
from keras import datasets, models, layers, applications, utils
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Diretórios
model_dir = 'backend/Roteiros_Individuais/CNN_finetunning/models'
image_dir = 'Roteiros_Individuais/CNN_finetunning/image'
os.makedirs(image_dir, exist_ok=True)

# Carregar modelo treinado com fine-tuning
def load_model():
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    # Congelar camadas iniciais e liberar camadas finais para fine-tuning
    for layer in base_model.layers[:15]:
        layer.trainable = False
    for layer in base_model.layers[15:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Carregar pesos salvos do modelo
    model_path = os.path.join(model_dir, 'fine_tuned_model.weights.h5')
    model.load_weights(model_path)
    return model

# ===================================== K-Means ======================================================
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

    # Carregar e processar dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df = df[['sepal length (cm)', 'sepal width (cm)']]

    # Pré-processamento dos dados
    scaler = MinMaxScaler()
    df[['sepal length (cm)', 'sepal width (cm)']] = scaler.fit_transform(df[['sepal length (cm)', 'sepal width (cm)']])

    # Definir o número de clusters
    K = 3
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(df)
    df['Cluster'] = kmeans.labels_

    # Método do Cotovelo
    sse = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df[['sepal length (cm)', 'sepal width (cm)']])
        sse.append(kmeans.inertia_)

    # Salvar o gráfico em um diretório temporário
    temp_dir = tempfile.gettempdir()
    elbow_plot_path = os.path.join(temp_dir, 'elbow_plot.png')

    plt.figure()
    plt.plot(range(1, 10), sse)
    plt.xlabel('Número de Clusters')
    plt.ylabel('Soma dos Erros Quadrados (SSE)')
    plt.title('Método do Cotovelo')
    plt.savefig(elbow_plot_path)

    # Avaliação Índice de Silhueta
    silhouette_avg = silhouette_score(df[['sepal length (cm)', 'sepal width (cm)']], df['Cluster'])
    print(f'Silhouette Score para K={K}: {silhouette_avg}')

    # Armazenando resultados
    results = {
        'silhouette_score': silhouette_avg,
        'elbow_plot_path': elbow_plot_path
    }

    return jsonify({'silhouette_score': silhouette_avg})

# ===================================== C-Means ======================================================

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

results = None

plt.switch_backend('Agg')

# Função para calcular o Fuzzy Partition Coefficient (FPC)
def calculate_fpc(u):
    # Número de pontos e clusters
    n, c = u.shape
    # FPC = 1/n * soma dos quadrados dos elementos de u
    fpc = np.sum(u**2) / n
    return fpc

def train_fuzzy_cmeans():
    global results

    # Carregar dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df = df[['sepal length (cm)', 'sepal width (cm)']]

    # Pré-processamento dos dados
    scaler = MinMaxScaler()
    df[['sepal length (cm)', 'sepal width (cm)']] = scaler.fit_transform(df[['sepal length (cm)', 'sepal width (cm)']])

    # Converter DataFrame para matriz para compatibilidade com skfuzzy
    data = df.values.T

    c = 3 
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

    results = {
        'fpc': fpc,
        'cmeans_plot_path': cmeans_plot_path
    }

    return jsonify({
        'message': 'Fuzzy C-Means trained successfully',
        'fpc': fpc
    })

# ===================================== Neuro Fuzzy ======================================================

from flask import Flask, jsonify, send_file
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tempfile
import os
import pickle

app = Flask(__name__)
CORS(app)

plt.switch_backend('Agg')

# Função de pertinência Gaussiana
def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

# Classe ANFIS
class ANFIS:
    def __init__(self, n_inputs, n_rules, learning_rate=0.02):
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.means = np.random.normal(0.5, 0.1, (n_rules, n_inputs))
        self.sigmas = np.full((n_rules, n_inputs), 0.1)
        self.weights = np.random.rand(n_rules)
        self.learning_rate = learning_rate
        self.mean_sigma_learning_rate = learning_rate * 0.27

    # Passo forward
    def forward(self, X):
        firing_strengths = np.zeros((X.shape[0], self.n_rules))
        for i in range(self.n_rules):
            mu = gaussmf(X[:, 0], self.means[i, 0], self.sigmas[i, 0]) * \
                 gaussmf(X[:, 1], self.means[i, 1], self.sigmas[i, 1])
            firing_strengths[:, i] = mu
        output = np.dot(firing_strengths, self.weights)
        return output, firing_strengths

    # Função de treinamento
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            output, firing_strengths = self.forward(X)
            error = y - output
            self.weights += self.learning_rate * np.dot(firing_strengths.T, error)
            for i in range(self.n_rules):
                for j in range(self.n_inputs):
                    grad_mean = np.sum(error * firing_strengths[:, i] * (X[:, j] - self.means[i, j]) / (self.sigmas[i, j] ** 2))
                    grad_sigma = np.sum(error * firing_strengths[:, i] * ((X[:, j] - self.means[i, j]) ** 2) / (self.sigmas[i, j] ** 3))
                    self.means[i, j] += self.mean_sigma_learning_rate * grad_mean
                    self.sigmas[i, j] += self.mean_sigma_learning_rate * grad_sigma

    # Função para medir precisão (accuracy)
    def calculate_accuracy(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return mse, mae

    # Função para salvar pesos
    def save_weights(self, file_path="anfis_weights.pkl"):
        with open(file_path, 'wb') as file:
            pickle.dump(self.weights, file)

    # Função para carregar pesos
    def load_weights(self, file_path="anfis_weights.pkl"):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.weights = pickle.load(file)

# Configuração e normalização dos dados
def prepare_data():
    temperaturas = np.random.uniform(20, 40, 100)
    umidades = np.random.uniform(30, 90, 100)
    velocidades = 2.5 * temperaturas + 0.5 * umidades

    scaler_X = MinMaxScaler()
    X = np.column_stack((temperaturas, umidades))
    X = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(velocidades.reshape(-1, 1)).flatten()

    return X, y, velocidades, scaler_y

# ====================================================== Fuzzy ===========================================================

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import tempfile
import os

app = Flask(__name__)
CORS(app)

plt.switch_backend('Agg')

def create_fuzzy_system(temperature_input, humidity_input):

    temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

    # Definindo as funções para a temperatura
    temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
    temperature['medium'] = fuzz.trimf(temperature.universe, [15, 25, 35])
    temperature['high'] = fuzz.trimf(temperature.universe, [30, 40, 40])

    # Definindo as funções para a umidade
    humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 40])
    humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
    humidity['high'] = fuzz.trimf(humidity.universe, [60, 100, 100])

    # Definindo as funções para a velocidade do ventilador
    fan_speed['off'] = fuzz.trimf(fan_speed.universe, [0, 0, 30])
    fan_speed['low'] = fuzz.trimf(fan_speed.universe, [20, 40, 60])
    fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [50, 70, 90])
    fan_speed['high'] = fuzz.trimf(fan_speed.universe, [70, 100, 100])

    # Criando as regras fuzzy
    rule1 = ctrl.Rule(temperature['low'] & humidity['low'], fan_speed['off'])
    rule2 = ctrl.Rule(temperature['low'] & humidity['medium'], fan_speed['low'])
    rule3 = ctrl.Rule(temperature['low'] & humidity['high'], fan_speed['medium'])

    rule4 = ctrl.Rule(temperature['medium'] & humidity['low'], fan_speed['low'])
    rule5 = ctrl.Rule(temperature['medium'] & humidity['medium'], fan_speed['medium'])
    rule6 = ctrl.Rule(temperature['medium'] & humidity['high'], fan_speed['high'])

    rule7 = ctrl.Rule(temperature['high'] & humidity['low'], fan_speed['medium'])
    rule8 = ctrl.Rule(temperature['high'] & humidity['medium'], fan_speed['high'])
    rule9 = ctrl.Rule(temperature['high'] & humidity['high'], fan_speed['high'])

    # Criando o sistema de controle fuzzy
    fan_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    fan_simulation = ctrl.ControlSystemSimulation(fan_control)

    # Definindo os inputs
    fan_simulation.input['temperature'] = temperature_input
    fan_simulation.input['humidity'] = humidity_input

    fan_simulation.compute()

    return fan_simulation.output['fan_speed']

# ====================================================== HTTP SOM ===========================================================

@app.route('/som/train', methods=['POST'])
def train_som_endpoint():

    iris = load_iris()
    X = iris.data
    y = iris.target
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    som_grid_size = 7
    som = np.random.rand(som_grid_size, som_grid_size, X.shape[1])

    accuracy_manual_som, manual_som_image_path = train_som(X_scaled, y, som, max_iterations=500, learning_rate=0.5, radius=3)
    
    accuracy_store['manual'] = accuracy_manual_som

    minisom_image_path, accuracy_minisom = train_minisom(X_scaled, y)

    accuracy_store['minisom'] = accuracy_minisom

    return jsonify({
        "message": "SOMs trained successfully",
        "manual_som_image_path": manual_som_image_path,
        "minisom_image_path": minisom_image_path,
        "accuracy_manual_som": accuracy_manual_som,
        "accuracy_minisom": accuracy_minisom
    })

@app.route('/som/get-image/<som_type>', methods=['GET'])
def get_image(som_type):
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, f'{som_type}_som_plot.png')
    
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    
    return jsonify({"error": f"Image for {som_type} SOM not found"}), 404

@app.route('/som/get-accuracy/<som_type>', methods=['GET'])
def get_accuracy(som_type):
    accuracy = accuracy_store.get(som_type)
    if accuracy is not None:
        return jsonify({"accuracy": accuracy})
    return jsonify({"error": f"Accuracy for {som_type} SOM not found"}), 404


# ====================================================== HTTP SVM ===========================================================

@app.route('/svm/run', methods=['POST'])
def run_svm_endpoint():
    run_svm()
    return jsonify({"message": "Modelo SVM executado com sucesso!"})

@app.route('/svm/results', methods=['GET'])
def fetch_results():  
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o SVM primeiro."}), 400

    return jsonify({
        'accuracy': results['accuracy'],
        'image_url': f"{request.host_url}image"
    })

@app.route('/svm/image', methods=['GET'])
def serve_image():
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o SVM primeiro."}), 400

    return send_file(io.BytesIO(results['image_data']), mimetype='image/png')


# ====================================================== HTTP DL ===========================================================

@app.route('/dl/train', methods=['POST'])
def train():
    global confusion_matrix_tf, confusion_matrix_pt, accuracy_tf, accuracy_pt
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    tf_thread = threading.Thread(target=train_tf, args=(X_train, y_train, X_test, y_test))
    pt_thread = threading.Thread(target=train_pt, args=(X_train, y_train, X_test, y_test))
    
    tf_thread.start()
    pt_thread.start()
    
    tf_thread.join()
    pt_thread.join()

    return jsonify({"message": "Training completed"}), 200

@app.route('/dl/image/pt', methods=['GET'])
@cross_origin()
def get_confusion_matrix_image():
    if confusion_matrix_pt is None:
        return jsonify({"error": "Model not trained yet"}), 400
    
    image_path = create_confusion_matrix_image(confusion_matrix_pt)
    return send_file(image_path, mimetype='image/png')

@app.route('/dl/image/tf', methods=['GET'])
@cross_origin()
def get_confusion_matrix_tf_image():
    if confusion_matrix_tf is None:
        return jsonify({"error": "Model not trained yet"}), 400
    
    image_path = create_confusion_matrix_image(confusion_matrix_tf)
    return send_file(image_path, mimetype='image/png')

@app.route('/dl/accuracy/tf', methods=['GET'])
def get_accuracy_tf():
    if accuracy_tf is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({"accuracy": accuracy_tf}), 200

@app.route('/dl/accuracy/pt', methods=['GET'])
def get_accuracy_pt():
    if accuracy_pt is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({"accuracy": accuracy_pt}), 200


# ====================================================== HTTP CNN Tensorflow ===========================================================

@app.route('/cnn/predict', methods=['POST'])
def predict_TF():
    global global_accuracy, global_f1_score

    (_, _), (X_test, y_test) = datasets.cifar10.load_data()
    X_test = X_test.astype('float32') / 255.0
    y_true = y_test.reshape(-1)

    model = load_model()
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    global_accuracy = accuracy_score(y_true, y_pred_classes)
    global_f1_score = f1_score(y_true, y_pred_classes, average='weighted')

    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 7))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.title('Matriz de Confusão - CIFAR-10')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if conf_matrix[i, j] > thresh else 'black')

    image_path = os.path.join(image_dir, 'confusion_matrix.png')
    plt.savefig(image_path)
    plt.close()

    return jsonify({"message": "Predição realizada", "accuracy": global_accuracy, "f1_score": global_f1_score})

@app.route('/cnn/image', methods=['GET'])
def get_confusion_matrix_TF():
    image_path = os.path.join(image_dir, 'confusion_matrix.png')
    return send_file(image_path, mimetype='image/png')

@app.route('/cnn/accuracy', methods=['GET'])
def get_metrics_TF():
    return jsonify({"accuracy": global_accuracy, "f1_score": global_f1_score})


# ====================================================== HTTP CNN Finetunning ===========================================================

@app.route('/cnn_finetunning/predict', methods=['POST'])
def predict_FT():
    global global_accuracy, global_f1_score

    (_, _), (X_test, y_test) = datasets.cifar10.load_data()
    X_test = X_test.astype('float32') / 255.0
    y_true = y_test.reshape(-1)

    model = load_model()
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    global_accuracy = accuracy_score(y_true, y_pred_classes)
    global_f1_score = f1_score(y_true, y_pred_classes, average='weighted')

    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 7))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.title('Matriz de Confusão - CIFAR-10')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if conf_matrix[i, j] > thresh else 'black')

    image_path = os.path.join(image_dir, 'confusion_matrix.png')
    plt.savefig(image_path)
    plt.close()

    return jsonify({"message": "Predição realizada", "accuracy": global_accuracy, "f1_score": global_f1_score})

@app.route('/cnn_finetunning/image', methods=['GET'])
def get_confusion_matrix_FT():
    image_path = os.path.join(image_dir, 'confusion_matrix.png')
    return send_file(image_path, mimetype='image/png')

@app.route('/cnn_finetunning/accuracy', methods=['GET'])
def get_metrics_FT():
    return jsonify({"accuracy": global_accuracy, "f1_score": global_f1_score})

# ====================================================== HTTP K-Means ===========================================================


@app.route('/k/run', methods=['POST'])
def run_k():
    return train_kmeans()
    

@app.route('/k/image', methods=['GET'])
def get_kmeans_image():
    if results is None or 'elbow_plot_path' not in results:
        return jsonify({"error": "Nenhum resultado disponível. Execute o treinamento do K-means primeiro."}), 400

    elbow_plot_path = results['elbow_plot_path']

    if os.path.exists(elbow_plot_path):
        return send_file(elbow_plot_path, mimetype='image/png')
    else:
        return jsonify({"error": "Imagem não encontrada. Execute o treinamento do K-means novamente."}), 404


# ====================================================== HTTP C-Means ===========================================================

@app.route('/c/run', methods=['POST'])
def run_c():
    return train_fuzzy_cmeans()

@app.route('/c/image', methods=['GET'])
def get_cmeans_image():
    if results is None or 'cmeans_plot_path' not in results:
        return jsonify({"error": "Nenhum resultado disponível. Execute o treinamento do Fuzzy C-Means primeiro."}), 400

    cmeans_plot_path = results['cmeans_plot_path']

    if os.path.exists(cmeans_plot_path):
        return send_file(cmeans_plot_path, mimetype='image/png')
    else:
        return jsonify({"error": "Imagem não encontrada. Execute o treinamento do Fuzzy C-Means novamente."}), 404


# ====================================================== HTTP Neuro Fuzzy ===========================================================

@app.route('/nf/run', methods=['POST'])
def run_anfis():
    global results
    X, y, velocidades, scaler_y = prepare_data()
    
    anfis_model = ANFIS(n_inputs=2, n_rules=5, learning_rate=0.02)
    anfis_model.train(X, y, epochs=100)
    anfis_model.save_weights()

    anfis_model.load_weights()
    y_pred, _ = anfis_model.forward(X)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mse, mae = anfis_model.calculate_accuracy(velocidades, y_pred)

    results = {'mse': mse, 'mae': mae, 'y_true': velocidades, 'y_pred': y_pred}
    return jsonify({'mse': mse, 'mae': mae})

@app.route('/nf/image', methods=['GET'])
def get_image_nf():
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o treinamento primeiro."}), 400
    
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    temp_dir = tempfile.gettempdir()
    plot_path = os.path.join(temp_dir, 'anfis_prediction.png')
    
    plt.figure()
    plt.plot(y_true, label='Velocidade esperada', color='blue')
    plt.plot(y_pred, label='Velocidade predita', linestyle='--', color='orange')
    plt.xlabel('Amostras')
    plt.ylabel('Velocidade do Ventilador')
    plt.legend()
    plt.title(f'MSE: {results["mse"]:.2f}, MAE: {results["mae"]:.2f}')
    plt.savefig(plot_path)
    plt.close()

    return send_file(plot_path, mimetype='image/png')

# ====================================================== HTTP Fuzzy ===========================================================
@app.route('/f/run', methods=['POST'])
def get_fan_speed():

    data = request.get_json()
    temperature_input = data.get('temperature', 30)
    humidity_input = data.get('humidity', 60)


    fan_speed = create_fuzzy_system(temperature_input, humidity_input)

    return jsonify({"fan_speed": fan_speed})

@app.route('/f/image', methods=['GET'])
def plot_fuzzy_system():

    temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

    temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
    temperature['medium'] = fuzz.trimf(temperature.universe, [15, 25, 35])
    temperature['high'] = fuzz.trimf(temperature.universe, [30, 40, 40])

    humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 40])
    humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
    humidity['high'] = fuzz.trimf(humidity.universe, [60, 100, 100])

    fan_speed['off'] = fuzz.trimf(fan_speed.universe, [0, 0, 30])
    fan_speed['low'] = fuzz.trimf(fan_speed.universe, [20, 40, 60])
    fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [50, 70, 90])
    fan_speed['high'] = fuzz.trimf(fan_speed.universe, [70, 100, 100])

    fan_speed.view()

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(tmpfile.name)
    tmpfile.close()

    return send_file(tmpfile.name, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)