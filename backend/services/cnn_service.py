import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers, applications, utils
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from models.cnn_model import CNNResult
from config import MODEL_DIR_CNN, IMAGE_DIR_CNN, MODEL_DIR_CNN_FT, IMAGE_DIR_CNN_FT

plt.switch_backend('Agg')

os.makedirs(IMAGE_DIR_CNN, exist_ok=True)
os.makedirs(IMAGE_DIR_CNN_FT, exist_ok=True)


class CNNService:
    def _load_cnn_model(self):
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
        model_path = os.path.join(MODEL_DIR_CNN, 'melhor_modelo_cnn_pesos.weights.h5')
        model.load_weights(model_path)
        return model

    def _load_finetuned_model(self):
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
        model_path = os.path.join(MODEL_DIR_CNN_FT, 'fine_tuned_model.weights.h5')
        model.load_weights(model_path)
        return model

    def _run_prediction(self, model, image_save_dir) -> CNNResult:
        (_, _), (X_test, y_test) = datasets.cifar10.load_data()
        X_test = X_test.astype('float32') / 255.0
        y_true = y_test.reshape(-1)

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_true, y_pred_classes)
        f1 = f1_score(y_true, y_pred_classes, average='weighted')

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

        image_path = os.path.join(image_save_dir, 'confusion_matrix.png')
        plt.savefig(image_path)
        plt.close()

        return CNNResult(accuracy=acc, f1_score=f1)

    def predict_cnn(self) -> CNNResult:
        model = self._load_cnn_model()
        return self._run_prediction(model, IMAGE_DIR_CNN)

    def predict_finetuned(self) -> CNNResult:
        model = self._load_finetuned_model()
        return self._run_prediction(model, IMAGE_DIR_CNN_FT)
