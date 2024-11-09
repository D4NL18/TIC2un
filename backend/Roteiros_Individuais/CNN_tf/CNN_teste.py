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

(_, _), (X_test, y_test) = datasets.cifar10.load_data()

X_test = X_test.astype('float32') / 255.0

y_test_cat = utils.to_categorical(y_test, 10)

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

print(f"Pesos carregados com sucesso de: {model_path}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = y_test.reshape(-1)
acc = accuracy_score(y_true, y_pred_classes)
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f"Acurácia no conjunto de teste: {acc:.4f}")
print(f"F1 Score no conjunto de teste: {f1:.4f}")

conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Matriz de Confusão:")
print(conf_matrix)
