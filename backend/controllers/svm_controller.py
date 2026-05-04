import io
from flask import Blueprint, jsonify, send_file, request
from repositories import svm_repo
from services.svm_service import SVMService

svm_blueprint = Blueprint('svm', __name__)
svm_service = SVMService()


@svm_blueprint.route('/svm/run', methods=['POST'])
def run_svm_endpoint():
    result = svm_service.run()
    svm_repo.store(result)
    return jsonify({"message": "Modelo SVM executado com sucesso!"})


@svm_blueprint.route('/svm/results', methods=['GET'])
def fetch_results():
    result = svm_repo.get()
    if result is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o SVM primeiro."}), 400

    return jsonify({
        'accuracy': result.accuracy,
        'image_url': f"{request.host_url}svm/image"
    })


@svm_blueprint.route('/svm/image', methods=['GET'])
def serve_image():
    result = svm_repo.get()
    if result is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o SVM primeiro."}), 400

    return send_file(io.BytesIO(result.image_data), mimetype='image/png')
