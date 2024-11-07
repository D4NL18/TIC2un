from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# Função para executar o K-means
def execute_kmeans(data, n_clusters):
    # Criação do DataFrame com os dados recebidos
    df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

    # Verifica se o DataFrame tem dados suficientes
    if df.empty or df.shape[0] < 2:
        raise ValueError("Os dados devem conter pelo menos duas amostras para realizar o K-means.")
    
    # Escalonamento dos dados
    scaler = MinMaxScaler()
    df[['Feature1', 'Feature2']] = scaler.fit_transform(df[['Feature1', 'Feature2']])

    # Execução do K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df[['Feature1', 'Feature2']])

    # Retorna os resultados
    return df, kmeans

# Endpoint para executar o K-means
@app.route('/k/run', methods=['POST'])
def run():
    try:
        # Recebe os dados JSON do frontend
        req_data = request.json
        print("Dados recebidos:", req_data)  # Para verificar os dados recebidos no console do Flask
        
        data = req_data.get("data", [])
        n_clusters = req_data.get("n_clusters", 3)

        if not data or n_clusters < 1:
            return jsonify({"error": "Dados ou número de clusters inválidos"}), 400

        # Executa o K-means
        df, kmeans = execute_kmeans(data, n_clusters)
        clusters = df['Cluster'].tolist()

        # Retorna os clusters como JSON
        return jsonify({"clusters": clusters})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Erro ao executar o K-means"}), 500


# Endpoint para gerar o gráfico do método Elbow
@app.route('/k/elbow', methods=['GET'])
def elbow():
    try:
        # Gera dados de exemplo para o gráfico do método Elbow
        X = np.random.rand(100, 2)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        distortions = []
        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)

        # Cria o gráfico do método Elbow
        plt.figure(figsize=(8, 6))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Número de clusters')
        plt.ylabel('Distortion')
        plt.title('Método Elbow para encontrar o número ótimo de clusters')

        # Salva o gráfico em um buffer e converte para base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Retorna o gráfico como uma imagem base64
        return jsonify({"elbow_plot": image_base64})

    except Exception as e:
        return jsonify({"error": "Erro ao gerar o gráfico Elbow"}), 500

if __name__ == '__main__':
    app.run(debug=True)
