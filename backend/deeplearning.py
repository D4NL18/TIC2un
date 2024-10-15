from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

app = Flask(__name__)
CORS(app)

# Carregando o dataset Iris
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Variáveis globais para armazenar resultados
confusion_matrix_tf = None
confusion_matrix_pt = None
accuracy_tf = None
accuracy_pt = None

# Função para normalizar e preparar os dados
def prepare_data(X, y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Função de treinamento em TensorFlow
def train_tf(X_train, y_train, X_test, y_test):
    global confusion_matrix_tf, accuracy_tf
    
    # Definir o modelo
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

# Função de treinamento em PyTorch
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

# Endpoint para iniciar os treinamentos
@app.route('/train', methods=['POST'])
def train():
    global confusion_matrix_tf, confusion_matrix_pt, accuracy_tf, accuracy_pt
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Treinamento em paralelo (opcional)
    import threading
    tf_thread = threading.Thread(target=train_tf, args=(X_train, y_train, X_test, y_test))
    pt_thread = threading.Thread(target=train_pt, args=(X_train, y_train, X_test, y_test))
    
    tf_thread.start()
    pt_thread.start()
    
    tf_thread.join()
    pt_thread.join()

    return jsonify({"message": "Training completed"}), 200

# Endpoint para obter a matriz de confusão do TensorFlow
@app.route('/confusion-matrix/tf', methods=['GET'])
def get_confusion_matrix_tf():
    if confusion_matrix_tf is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify(confusion_matrix_tf.tolist()), 200

# Endpoint para obter a matriz de confusão do PyTorch
@app.route('/confusion-matrix/pt', methods=['GET'])
def get_confusion_matrix_pt():
    if confusion_matrix_pt is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify(confusion_matrix_pt.tolist()), 200

# Endpoint para obter a precisão do TensorFlow
@app.route('/accuracy/tf', methods=['GET'])
def get_accuracy_tf():
    if accuracy_tf is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({"accuracy": accuracy_tf}), 200

# Endpoint para obter a precisão do PyTorch
@app.route('/accuracy/pt', methods=['GET'])
def get_accuracy_pt():
    if accuracy_pt is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({"accuracy": accuracy_pt}), 200

if __name__ == '__main__':
    app.run(debug=True)
