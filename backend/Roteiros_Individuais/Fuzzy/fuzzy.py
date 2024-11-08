from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Configuração do backend do Matplotlib para renderização fora da tela
plt.switch_backend('Agg')

# Função para configurar o sistema Fuzzy de controle de ventilador
def create_fuzzy_system(temperature_input, humidity_input):
    # Definindo as variáveis de entrada e saída
    temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

    # Definindo as funções de pertinência para a temperatura
    temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
    temperature['medium'] = fuzz.trimf(temperature.universe, [15, 25, 35])
    temperature['high'] = fuzz.trimf(temperature.universe, [30, 40, 40])

    # Definindo as funções de pertinência para a umidade
    humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 40])
    humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
    humidity['high'] = fuzz.trimf(humidity.universe, [60, 100, 100])

    # Definindo as funções de pertinência para a velocidade do ventilador
    fan_speed['off'] = fuzz.trimf(fan_speed.universe, [0, 0, 30])
    fan_speed['low'] = fuzz.trimf(fan_speed.universe, [20, 40, 60])
    fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [50, 70, 90])
    fan_speed['high'] = fuzz.trimf(fan_speed.universe, [70, 100, 100])

    # Criando as regras fuzzy
    rule1 = ctrl.Rule(temperature['low'] & humidity['low'], fan_speed['off'])
    rule2 = ctrl.Rule(temperature['low'] & humidity['medium'], fan_speed['low'])
    rule3 = ctrl.Rule(temperature['low'] & humidity['high'], fan_speed['medium'])

    rule4 = ctrl.Rule(temperature['medium'] & humidity['low'], fan_speed['low'])
    rule5 = ctrl.Rule(temperature['medium'] & humidity['medium'], fan_speed['medium'])
    rule6 = ctrl.Rule(temperature['medium'] & humidity['high'], fan_speed['high'])

    rule7 = ctrl.Rule(temperature['high'] & humidity['low'], fan_speed['medium'])
    rule8 = ctrl.Rule(temperature['high'] & humidity['medium'], fan_speed['high'])
    rule9 = ctrl.Rule(temperature['high'] & humidity['high'], fan_speed['high'])

    # Criando o sistema de controle fuzzy
    fan_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    fan_simulation = ctrl.ControlSystemSimulation(fan_control)

    # Definindo as entradas do sistema
    fan_simulation.input['temperature'] = temperature_input  # Temperatura em graus Celsius
    fan_simulation.input['humidity'] = humidity_input       # Umidade em porcentagem

    # Computando a saída
    fan_simulation.compute()

    return fan_simulation.output['fan_speed']

@app.route('/f/run', methods=['POST'])
def get_fan_speed():
    # Pegando os dados de temperatura e umidade do corpo da requisição
    data = request.get_json()
    temperature_input = data.get('temperature', 30)
    humidity_input = data.get('humidity', 60)

    # Calculando a velocidade do ventilador com o sistema Fuzzy
    fan_speed = create_fuzzy_system(temperature_input, humidity_input)

    # Retornando o resultado como JSON
    return jsonify({"fan_speed": fan_speed})

@app.route('/f/image', methods=['GET'])
def plot_fuzzy_system():
    # Gerando o gráfico para visualizar as funções de pertinência
    temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

    # Definindo as funções de pertinência
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

    # Plotando as funções de pertinência
    fan_speed.view()

    # Salvando o gráfico em um arquivo temporário
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(tmpfile.name)
    tmpfile.close()

    # Retornando o arquivo gerado como resposta
    return send_file(tmpfile.name, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
