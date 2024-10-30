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
CORS(app)


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
    iris = datasets.load_iris()

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


# ====================================================== CNN ===========================================================

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
model_dir = 'backend/CNN/models'
image_dir = 'CNN/images'
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





# ====================================================== HTTP SOM ===========================================================

#Post para realizar treinamentos
@app.route('/som/train', methods=['POST'])
def train_som_endpoint():
    # Carregar dataset e normalização dos dados
    iris = load_iris()
    X = iris.data
    y = iris.target
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Inicializar SOM manual
    som_grid_size = 7
    som = np.random.rand(som_grid_size, som_grid_size, X.shape[1])

    # Treinar SOM manual
    accuracy_manual_som, manual_som_image_path = train_som(X_scaled, y, som, max_iterations=500, learning_rate=0.5, radius=3)
    
    # Armazenar a precisão do SOM manual
    accuracy_store['manual'] = accuracy_manual_som

    # Treinar MiniSom e obter a imagem e a precisão
    minisom_image_path, accuracy_minisom = train_minisom(X_scaled, y)

    # Armazenar a precisão do MiniSom
    accuracy_store['minisom'] = accuracy_minisom

    return jsonify({
        "message": "SOMs trained successfully",
        "manual_som_image_path": manual_som_image_path,
        "minisom_image_path": minisom_image_path,
        "accuracy_manual_som": accuracy_manual_som,
        "accuracy_minisom": accuracy_minisom
    })

# Get para imagem
@app.route('/som/get-image/<som_type>', methods=['GET'])
def get_image(som_type):
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, f'{som_type}_som_plot.png')
    
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    
    return jsonify({"error": f"Image for {som_type} SOM not found"}), 404

#Get para accuracy
@app.route('/som/get-accuracy/<som_type>', methods=['GET'])
def get_accuracy(som_type):
    accuracy = accuracy_store.get(som_type)
    if accuracy is not None:
        return jsonify({"accuracy": accuracy})
    return jsonify({"error": f"Accuracy for {som_type} SOM not found"}), 404


# ====================================================== HTTP SVM ===========================================================

# Post (executar SVM)
@app.route('/svm/run', methods=['POST'])
def run_svm_endpoint():
    run_svm()
    return jsonify({"message": "Modelo SVM executado com sucesso!"})

# Get (resultados)
@app.route('/svm/results', methods=['GET'])
def fetch_results():  
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o SVM primeiro."}), 400

    return jsonify({
        'accuracy': results['accuracy'],
        'image_url': f"{request.host_url}image"
    })

# Get (imagem)
@app.route('/svm/image', methods=['GET'])
def serve_image():
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o SVM primeiro."}), 400

    return send_file(io.BytesIO(results['image_data']), mimetype='image/png')


# ====================================================== HTTP DL ===========================================================

# Post Treinamentos
@app.route('/dl/train', methods=['POST'])
def train():
    global confusion_matrix_tf, confusion_matrix_pt, accuracy_tf, accuracy_pt
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Treinamento usando threads
    tf_thread = threading.Thread(target=train_tf, args=(X_train, y_train, X_test, y_test))
    pt_thread = threading.Thread(target=train_pt, args=(X_train, y_train, X_test, y_test))
    
    tf_thread.start()
    pt_thread.start()
    
    tf_thread.join()
    pt_thread.join()

    return jsonify({"message": "Training completed"}), 200

# Get imagem PyTorch
@app.route('/dl/image/pt', methods=['GET'])
@cross_origin()  # Aplicando CORS especificamente a este endpoint
def get_confusion_matrix_image():
    if confusion_matrix_pt is None:
        return jsonify({"error": "Model not trained yet"}), 400
    
    image_path = create_confusion_matrix_image(confusion_matrix_pt)
    return send_file(image_path, mimetype='image/png')

# Get imagem TensorFlow
@app.route('/dl/image/tf', methods=['GET'])
@cross_origin()  # Aplicando CORS especificamente a este endpoint
def get_confusion_matrix_tf_image():
    if confusion_matrix_tf is None:
        return jsonify({"error": "Model not trained yet"}), 400
    
    image_path = create_confusion_matrix_image(confusion_matrix_tf)
    return send_file(image_path, mimetype='image/png')

# Get accuracy TensorFlow
@app.route('/dl/accuracy/tf', methods=['GET'])
def get_accuracy_tf():
    if accuracy_tf is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({"accuracy": accuracy_tf}), 200

# Get imagem PyTorch
@app.route('/dl/accuracy/pt', methods=['GET'])
def get_accuracy_pt():
    if accuracy_pt is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({"accuracy": accuracy_pt}), 200


# ====================================================== HTTP CNN ===========================================================


# Post (executar predição)
@app.route('/cnn/predict', methods=['POST'])
def predict():
    global global_accuracy, global_f1_score

    (_, _), (X_test, y_test) = datasets.cifar10.load_data()
    X_test = X_test.astype('float32') / 255.0
    y_true = y_test.reshape(-1)

    model = load_model()
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    global_accuracy = accuracy_score(y_true, y_pred_classes)
    global_f1_score = f1_score(y_true, y_pred_classes, average='weighted')

    #Matriz Confusão
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

# Get matriz confusão
@app.route('/cnn/image', methods=['GET'])
def get_confusion_matrix():
    image_path = os.path.join(image_dir, 'confusion_matrix.png')
    return send_file(image_path, mimetype='image/png')

# Get métricas
@app.route('/cnn/accuracy', methods=['GET'])
def get_metrics():
    return jsonify({"accuracy": global_accuracy, "f1_score": global_f1_score})


# ====================================================== RUN APP ===========================================================


if __name__ == '__main__':
    app.run(debug=True)
