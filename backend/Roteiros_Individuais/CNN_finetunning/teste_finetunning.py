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
image_dir = 'image'
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

# Rota POST para predição
@app.route('/cnn_finetunning/predict', methods=['POST'])
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

    # Gerar e salvar a matriz de confusão
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

# Rota GET para obter a matriz de confusão
@app.route('/cnn_finetunning/image', methods=['GET'])
def get_confusion_matrix():
    image_path = os.path.join(image_dir, 'confusion_matrix.png')
    return send_file(image_path, mimetype='image/png')

# Rota GET para obter métricas
@app.route('/cnn_finetunning/accuracy', methods=['GET'])
def get_metrics():
    return jsonify({"accuracy": global_accuracy, "f1_score": global_f1_score})

if __name__ == '__main__':
    app.run(debug=True)
