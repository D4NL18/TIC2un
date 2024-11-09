import os
import tensorflow as tf
from keras import datasets, models, layers, applications, utils
import matplotlib.pyplot as plt

model_dir = 'backend/CNN/models'
image_dir = 'backend/CNN/images'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalização
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encoding
y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)

# Model CNN
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

# Treinar
history = model.fit(X_train, y_train_cat, validation_split=0.2, epochs=10, batch_size=32)

# Salvar os pesos
model_path = os.path.join(model_dir, 'melhor_modelo_cnn_pesos.weights.h5')
model.save_weights(model_path)

print(f"Pesos salvos com sucesso em: {model_path}")