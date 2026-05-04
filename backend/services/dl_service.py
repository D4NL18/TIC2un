import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import threading
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

from models.dl_model import DLResult

plt.switch_backend('Agg')


class DLService:
    def _prepare_data(self):
        iris = load_iris()
        X = iris.data
        y = iris.target
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def _train_tf(self, X_train, y_train, X_test, y_test, result: DLResult):
        model = keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(10, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=100, batch_size=5, validation_split=0.2)

        _, test_accuracy = model.evaluate(X_test, y_test)
        result.accuracy_tf = test_accuracy

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        result.confusion_matrix_tf = confusion_matrix(y_test, y_pred_classes)

    def _train_pt(self, X_train, y_train, X_test, y_test, result: DLResult):
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

        for epoch in range(100):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            result.accuracy_pt = (predicted == y_test_tensor).float().mean().item()
            result.confusion_matrix_pt = confusion_matrix(y_test, predicted.numpy())

    def train(self) -> DLResult:
        X_train, X_test, y_train, y_test = self._prepare_data()
        result = DLResult()

        tf_thread = threading.Thread(target=self._train_tf, args=(X_train, y_train, X_test, y_test, result))
        pt_thread = threading.Thread(target=self._train_pt, args=(X_train, y_train, X_test, y_test, result))

        tf_thread.start()
        pt_thread.start()
        tf_thread.join()
        pt_thread.join()

        return result

    def create_confusion_matrix_image(self, conf_matrix) -> str:
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

        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, 'confusion_matrix.png')
        plt.savefig(image_path)
        plt.close()
        return image_path
