import os
import tempfile
from flask import Blueprint, jsonify, send_file
from repositories import som_repo
from services.som_service import SOMService

som_blueprint = Blueprint('som', __name__)
som_service = SOMService()


@som_blueprint.route('/som/train', methods=['POST'])
def train_som_endpoint():
    result = som_service.train()
    som_repo.store(result)

    return jsonify({
        "message": "SOMs trained successfully",
        "manual_som_image_path": result.manual_image_path,
        "minisom_image_path": result.minisom_image_path,
        "accuracy_manual_som": result.manual_accuracy,
        "accuracy_minisom": result.minisom_accuracy
    })


@som_blueprint.route('/som/get-image/<som_type>', methods=['GET'])
def get_image(som_type):
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, f'{som_type}_som_plot.png')

    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')

    return jsonify({"error": f"Image for {som_type} SOM not found"}), 404


@som_blueprint.route('/som/get-accuracy/<som_type>', methods=['GET'])
def get_accuracy(som_type):
    result = som_repo.get()
    if result is None:
        return jsonify({"error": f"Accuracy for {som_type} SOM not found"}), 404

    if som_type == 'manual':
        accuracy = result.manual_accuracy
    elif som_type == 'minisom':
        accuracy = result.minisom_accuracy
    else:
        return jsonify({"error": f"Unknown SOM type: {som_type}"}), 404

    if accuracy is not None:
        return jsonify({"accuracy": accuracy})
    return jsonify({"error": f"Accuracy for {som_type} SOM not found"}), 404
