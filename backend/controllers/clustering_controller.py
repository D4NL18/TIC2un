from flask import Blueprint, jsonify, send_file
from repositories import clustering_repo
from services.clustering_service import ClusteringService

clustering_blueprint = Blueprint('clustering', __name__)
clustering_service = ClusteringService()


@clustering_blueprint.route('/k/run', methods=['POST'])
def run_kmeans():
    result = clustering_service.train_kmeans()
    clustering_repo.store_kmeans(result)
    return jsonify({'silhouette_score': result.silhouette_score})


@clustering_blueprint.route('/k/image', methods=['GET'])
def get_kmeans_image():
    result = clustering_repo.get_kmeans()
    if result is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o treinamento do K-means primeiro."}), 400

    import os
    if os.path.exists(result.elbow_plot_path):
        return send_file(result.elbow_plot_path, mimetype='image/png')
    return jsonify({"error": "Imagem não encontrada. Execute o treinamento do K-means novamente."}), 404


@clustering_blueprint.route('/c/run', methods=['POST'])
def run_cmeans():
    result = clustering_service.train_cmeans()
    clustering_repo.store_cmeans(result)
    return jsonify({'message': 'Fuzzy C-Means trained successfully', 'fpc': result.fpc})


@clustering_blueprint.route('/c/image', methods=['GET'])
def get_cmeans_image():
    result = clustering_repo.get_cmeans()
    if result is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o treinamento do Fuzzy C-Means primeiro."}), 400

    import os
    if os.path.exists(result.cmeans_plot_path):
        return send_file(result.cmeans_plot_path, mimetype='image/png')
    return jsonify({"error": "Imagem não encontrada. Execute o treinamento do Fuzzy C-Means novamente."}), 404
