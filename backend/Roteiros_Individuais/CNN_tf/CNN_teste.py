import os
import tensorflow as tf
from tensorflow import keras
from keras  import datasets
from keras import models
from keras import layers
from keras import applications
from keras import utils 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np

model_dir = 'backend/CNN/models'

# Dataset
(_, _), (X_test, y_test) = datasets.cifar10.load_data()

# Normalização
X_test = X_test.astype('float32') / 255.0

# One-hot encoding
y_test_cat = utils.to_categorical(y_test, 10)

# Reconstruir o modelo CNN do treinamento
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas de classificação
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compilar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Carregar os pesos salvos
model_path = os.path.join(model_dir, 'melhor_modelo_cnn_pesos.weights.h5')
model.load_weights(model_path)

print(f"Pesos carregados com sucesso de: {model_path}")

# Predições
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calcular métricas
y_true = y_test.reshape(-1)
acc = accuracy_score(y_true, y_pred_classes)
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f"Acurácia no conjunto de teste: {acc:.4f}")
print(f"F1 Score no conjunto de teste: {f1:.4f}")

# Matriz Confusão
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Matriz de Confusão:")
print(conf_matrix)
