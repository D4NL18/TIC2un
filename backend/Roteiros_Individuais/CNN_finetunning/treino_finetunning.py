import os
import tensorflow as tf
from keras import datasets, models, layers, applications, utils
import matplotlib.pyplot as plt

model_dir = 'backend/Roteiros_Individuais/CNN_finetunning/models'
os.makedirs(model_dir, exist_ok=True)

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

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

history = model.fit(X_train, y_train_cat, validation_split=0.2, epochs=1, batch_size=32)

model_path = os.path.join(model_dir, 'fine_tuned_model.weights.h5')
model.save_weights(model_path)

print(f"Pesos salvos com sucesso em: {model_path}")