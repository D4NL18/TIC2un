from flask import Blueprint, jsonify, send_file, request
from repositories import fuzzy_repo
from services.fuzzy_service import FuzzyService

fuzzy_blueprint = Blueprint('fuzzy', __name__)
fuzzy_service = FuzzyService()


@fuzzy_blueprint.route('/nf/run', methods=['POST'])
def run_anfis():
    result = fuzzy_service.run_neuro_fuzzy()
    fuzzy_repo.store_nf(result)
    return jsonify({'mse': result.mse, 'mae': result.mae})


@fuzzy_blueprint.route('/nf/image', methods=['GET'])
def get_image_nf():
    result = fuzzy_repo.get_nf()
    if result is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o treinamento primeiro."}), 400

    plot_path = fuzzy_service.get_neuro_fuzzy_image(result)
    return send_file(plot_path, mimetype='image/png')


@fuzzy_blueprint.route('/f/run', methods=['POST'])
def get_fan_speed():
    data = request.get_json()
    temperature_input = data.get('temperature', 30)
    humidity_input = data.get('humidity', 60)

    fan_speed = fuzzy_service.run_fuzzy(temperature_input, humidity_input)
    return jsonify({"fan_speed": fan_speed})


@fuzzy_blueprint.route('/f/image', methods=['GET'])
def plot_fuzzy_system():
    plot_path = fuzzy_service.get_fuzzy_image()
    return send_file(plot_path, mimetype='image/png')
