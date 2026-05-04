import os
import tempfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.fuzzy_model import NeuroFuzzyResult
from config import ANFIS_WEIGHTS_PATH

plt.switch_backend('Agg')


def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


class ANFIS:
    def __init__(self, n_inputs, n_rules, learning_rate=0.02):
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.means = np.random.normal(0.5, 0.1, (n_rules, n_inputs))
        self.sigmas = np.full((n_rules, n_inputs), 0.1)
        self.weights = np.random.rand(n_rules)
        self.learning_rate = learning_rate
        self.mean_sigma_learning_rate = learning_rate * 0.27

    def forward(self, X):
        firing_strengths = np.zeros((X.shape[0], self.n_rules))
        for i in range(self.n_rules):
            mu = gaussmf(X[:, 0], self.means[i, 0], self.sigmas[i, 0]) * \
                 gaussmf(X[:, 1], self.means[i, 1], self.sigmas[i, 1])
            firing_strengths[:, i] = mu
        output = np.dot(firing_strengths, self.weights)
        return output, firing_strengths

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

    def calculate_accuracy(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return mse, mae

    def save_weights(self, file_path=ANFIS_WEIGHTS_PATH):
        with open(file_path, 'wb') as file:
            pickle.dump(self.weights, file)

    def load_weights(self, file_path=ANFIS_WEIGHTS_PATH):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.weights = pickle.load(file)


class FuzzyService:
    def _prepare_neuro_data(self):
        temperaturas = np.random.uniform(20, 40, 100)
        umidades = np.random.uniform(30, 90, 100)
        velocidades = 2.5 * temperaturas + 0.5 * umidades

        scaler_X = MinMaxScaler()
        X = np.column_stack((temperaturas, umidades))
        X = scaler_X.fit_transform(X)

        scaler_y = MinMaxScaler()
        y = scaler_y.fit_transform(velocidades.reshape(-1, 1)).flatten()

        return X, y, velocidades, scaler_y

    def run_neuro_fuzzy(self) -> NeuroFuzzyResult:
        X, y, velocidades, scaler_y = self._prepare_neuro_data()

        anfis_model = ANFIS(n_inputs=2, n_rules=5, learning_rate=0.02)
        anfis_model.train(X, y, epochs=100)
        anfis_model.save_weights()

        anfis_model.load_weights()
        y_pred, _ = anfis_model.forward(X)
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        mse, mae = anfis_model.calculate_accuracy(velocidades, y_pred)

        return NeuroFuzzyResult(mse=mse, mae=mae, y_true=velocidades, y_pred=y_pred)

    def get_neuro_fuzzy_image(self, result: NeuroFuzzyResult) -> str:
        temp_dir = tempfile.gettempdir()
        plot_path = os.path.join(temp_dir, 'anfis_prediction.png')

        plt.figure()
        plt.plot(result.y_true, label='Velocidade esperada', color='blue')
        plt.plot(result.y_pred, label='Velocidade predita', linestyle='--', color='orange')
        plt.xlabel('Amostras')
        plt.ylabel('Velocidade do Ventilador')
        plt.legend()
        plt.title(f'MSE: {result.mse:.2f}, MAE: {result.mae:.2f}')
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    def run_fuzzy(self, temperature_input: float, humidity_input: float) -> float:
        temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
        humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
        fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

        temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
        temperature['medium'] = fuzz.trimf(temperature.universe, [15, 25, 35])
        temperature['high'] = fuzz.trimf(temperature.universe, [30, 40, 40])

        humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 40])
        humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
        humidity['high'] = fuzz.trimf(humidity.universe, [60, 100, 100])

        fan_speed['off'] = fuzz.trimf(fan_speed.universe, [0, 0, 30])
        fan_speed['low'] = fuzz.trimf(fan_speed.universe, [20, 40, 60])
        fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [50, 70, 90])
        fan_speed['high'] = fuzz.trimf(fan_speed.universe, [70, 100, 100])

        rule1 = ctrl.Rule(temperature['low'] & humidity['low'], fan_speed['off'])
        rule2 = ctrl.Rule(temperature['low'] & humidity['medium'], fan_speed['low'])
        rule3 = ctrl.Rule(temperature['low'] & humidity['high'], fan_speed['medium'])
        rule4 = ctrl.Rule(temperature['medium'] & humidity['low'], fan_speed['low'])
        rule5 = ctrl.Rule(temperature['medium'] & humidity['medium'], fan_speed['medium'])
        rule6 = ctrl.Rule(temperature['medium'] & humidity['high'], fan_speed['high'])
        rule7 = ctrl.Rule(temperature['high'] & humidity['low'], fan_speed['medium'])
        rule8 = ctrl.Rule(temperature['high'] & humidity['medium'], fan_speed['high'])
        rule9 = ctrl.Rule(temperature['high'] & humidity['high'], fan_speed['high'])

        fan_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        fan_simulation = ctrl.ControlSystemSimulation(fan_control)

        fan_simulation.input['temperature'] = temperature_input
        fan_simulation.input['humidity'] = humidity_input
        fan_simulation.compute()

        return fan_simulation.output['fan_speed']

    def get_fuzzy_image(self) -> str:
        temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
        humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
        fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

        temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
        temperature['medium'] = fuzz.trimf(temperature.universe, [15, 25, 35])
        temperature['high'] = fuzz.trimf(temperature.universe, [30, 40, 40])

        humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 40])
        humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
        humidity['high'] = fuzz.trimf(humidity.universe, [60, 100, 100])

        fan_speed['off'] = fuzz.trimf(fan_speed.universe, [0, 0, 30])
        fan_speed['low'] = fuzz.trimf(fan_speed.universe, [20, 40, 60])
        fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [50, 70, 90])
        fan_speed['high'] = fuzz.trimf(fan_speed.universe, [70, 100, 100])

        fan_speed.view()

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(tmpfile.name)
        tmpfile.close()

        return tmpfile.name
