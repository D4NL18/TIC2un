import os
from flask import Blueprint, jsonify, send_file
from repositories import cnn_repo
from services.cnn_service import CNNService
from config import IMAGE_DIR_CNN, IMAGE_DIR_CNN_FT

cnn_blueprint = Blueprint('cnn', __name__)
cnn_service = CNNService()


@cnn_blueprint.route('/cnn/predict', methods=['POST'])
def predict_cnn():
    result = cnn_service.predict_cnn()
    cnn_repo.store_cnn(result)
    return jsonify({"message": "Predição realizada", "accuracy": result.accuracy, "f1_score": result.f1_score})


@cnn_blueprint.route('/cnn/image', methods=['GET'])
def get_confusion_matrix_cnn():
    image_path = os.path.join(IMAGE_DIR_CNN, 'confusion_matrix.png')
    return send_file(image_path, mimetype='image/png')


@cnn_blueprint.route('/cnn/accuracy', methods=['GET'])
def get_metrics_cnn():
    result = cnn_repo.get_cnn()
    if result is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({"accuracy": result.accuracy, "f1_score": result.f1_score})


@cnn_blueprint.route('/cnn_finetunning/predict', methods=['POST'])
def predict_finetuned():
    result = cnn_service.predict_finetuned()
    cnn_repo.store_ft(result)
    return jsonify({"message": "Predição realizada", "accuracy": result.accuracy, "f1_score": result.f1_score})


@cnn_blueprint.route('/cnn_finetunning/image', methods=['GET'])
def get_confusion_matrix_ft():
    image_path = os.path.join(IMAGE_DIR_CNN_FT, 'confusion_matrix.png')
    return send_file(image_path, mimetype='image/png')


@cnn_blueprint.route('/cnn_finetunning/accuracy', methods=['GET'])
def get_metrics_ft():
    result = cnn_repo.get_ft()
    if result is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({"accuracy": result.accuracy, "f1_score": result.f1_score})
