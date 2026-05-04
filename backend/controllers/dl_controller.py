from flask import Blueprint, jsonify, send_file
from flask_cors import cross_origin
from repositories import dl_repo
from services.dl_service import DLService

dl_blueprint = Blueprint('dl', __name__)
dl_service = DLService()


@dl_blueprint.route('/dl/train', methods=['POST'])
def train_dl():
    result = dl_service.train()
    dl_repo.store(result)
    return jsonify({"message": "Training completed"}), 200


@dl_blueprint.route('/dl/image/pt', methods=['GET'])
@cross_origin()
def get_confusion_matrix_image():
    result = dl_repo.get()
    if result is None or result.confusion_matrix_pt is None:
        return jsonify({"error": "Model not trained yet"}), 400

    image_path = dl_service.create_confusion_matrix_image(result.confusion_matrix_pt)
    return send_file(image_path, mimetype='image/png')


@dl_blueprint.route('/dl/image/tf', methods=['GET'])
@cross_origin()
def get_confusion_matrix_tf_image():
    result = dl_repo.get()
    if result is None or result.confusion_matrix_tf is None:
        return jsonify({"error": "Model not trained yet"}), 400

    image_path = dl_service.create_confusion_matrix_image(result.confusion_matrix_tf)
    return send_file(image_path, mimetype='image/png')


@dl_blueprint.route('/dl/accuracy/tf', methods=['GET'])
def get_accuracy_tf():
    result = dl_repo.get()
    if result is None or result.accuracy_tf is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({"accuracy": result.accuracy_tf}), 200


@dl_blueprint.route('/dl/accuracy/pt', methods=['GET'])
def get_accuracy_pt():
    result = dl_repo.get()
    if result is None or result.accuracy_pt is None:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify({"accuracy": result.accuracy_pt}), 200
