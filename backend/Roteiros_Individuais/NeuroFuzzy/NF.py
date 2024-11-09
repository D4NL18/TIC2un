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

# Endpoint para rodar o treinamento e salvar resultados
@app.route('/nf/run', methods=['POST'])
def run_anfis():
    global results
    X, y, velocidades, scaler_y = prepare_data()
    
    anfis_model = ANFIS(n_inputs=2, n_rules=5, learning_rate=0.02)
    anfis_model.train(X, y, epochs=100)
    anfis_model.save_weights()

    # Carrega os pesos para garantir consistência
    anfis_model.load_weights()
    y_pred, _ = anfis_model.forward(X)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()  # Reverte a normalização

    # Calcula a precisão
    mse, mae = anfis_model.calculate_accuracy(velocidades, y_pred)

    # Armazena resultados globais para outros endpoints
    results = {'mse': mse, 'mae': mae, 'y_true': velocidades, 'y_pred': y_pred}
    return jsonify({'mse': mse, 'mae': mae})

# Endpoint para retornar o gráfico de predição
@app.route('/nf/image', methods=['GET'])
def get_image():
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o treinamento primeiro."}), 400
    
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # Salva o gráfico temporário
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



if __name__ == '__main__':
    app.run(debug=True)
