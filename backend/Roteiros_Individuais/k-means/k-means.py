from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Importando o CORS
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import io

app = Flask(__name__)

# Configuração do CORS para aceitar qualquer origem e qualquer método
CORS(app)

# Variável global para armazenar os resultados
results = None

plt.switch_backend('Agg')

@app.route('/k/run', methods=['POST'])
def execute_kmeans():
    global results

    # Carregar e processar dados do corpo do request
    data = request.get_json()  # Recebe o dataset JSON
    df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

    # Pré-processamento dos dados
    scaler = MinMaxScaler()
    df[['Feature1', 'Feature2']] = scaler.fit_transform(df[['Feature1', 'Feature2']])

    # Definir o número de clusters (K=3) e executar o KMeans
    K = 3
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(df[['Feature1', 'Feature2']])
    df['Cluster'] = kmeans.labels_

    # Método do Cotovelo para sugerir o número ideal de clusters
    sse = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df[['Feature1', 'Feature2']])
        sse.append(kmeans.inertia_)

    # Salvar o gráfico do Método do Cotovelo
    plt.figure()
    plt.plot(range(1, 10), sse)
    plt.xlabel('Número de Clusters')
    plt.ylabel('Soma dos Erros Quadrados (SSE)')
    plt.title('Método do Cotovelo')
    plt.savefig('elbow_plot.png')

    # Avaliação com Índice de Silhueta para K=3
    silhouette_avg = silhouette_score(df[['Feature1', 'Feature2']], df['Cluster'])
    print(f'Silhouette Score para K={K}: {silhouette_avg}')

    # Salvando o modelo KMeans
    joblib.dump(kmeans, 'modelo_kmeans.pkl')

    # Armazenando resultados
    results = {
        'silhouette_score': silhouette_avg,
        'image_data': open('elbow_plot.png', 'rb').read()  # Lê a imagem em bytes
    }

    return jsonify({'silhouette_score': silhouette_avg})

@app.route('/k/results', methods=['GET'])
def fetch_results():
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o K-means primeiro."}), 400

    return jsonify({
        'silhouette_score': results['silhouette_score'],
        'image_url': f"{request.host_url}k/image"
    })

@app.route('/k/image', methods=['GET'])
def serve_image():
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o K-means primeiro."}), 400

    return send_file(io.BytesIO(results['image_data']), mimetype='image/png')

# Inicializa servidor
if __name__ == '__main__':
    app.run(debug=True)
