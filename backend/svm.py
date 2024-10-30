# Passo 1: Imports
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import io
import os

# Passo 7: Importar o Modelo SVM
from sklearn.svm import SVC

app = Flask(__name__)
CORS(app)

results = None

plt.switch_backend('Agg')

def run_svm():
    global results

    # Passo 2: Escolher a Base de Dados
    iris = datasets.load_iris()

    # Passo 3: Carregar os Dados
    X = iris.data
    y = iris.target

    # Passo 4: Analisar os Dados
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y
    print(df.head())
    print(df.info())

    # Passo 5: Pré-processamento dos Dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Passo 6: Separar os Dados em Conjuntos de Treinamento e Teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Passo 8: Criar uma Instância do Modelo
    svm_model = SVC()

    # Passo 9: Treinar o Modelo
    svm_model.fit(X_train, y_train)

    # Passo 10: Fazer Previsões
    y_pred = svm_model.predict(X_test)

    # Passo 11: Avaliar o Modelo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)

    # Passo 12: Visualizar a Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')

    # Salvar imagem para enviar no get
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close() 

    # Passo 13: Ajustar Hiperparâmetros
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto'] 
    }
    grid_search = GridSearchCV(svm_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_svm_model = grid_search.best_estimator_

    # Passo 14: Documentar e Salvar o Modelo
    backup_dir = 'backend/backup_svm'

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    model_filename = os.path.join(backup_dir, 'best_svm_model.joblib')
    joblib.dump(best_svm_model, model_filename)
    print(f"Modelo treinado salvo como {model_filename}")

    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': classification_rep,
        'image_data': img_bytes.getvalue()
    }

# Post (executar SVM)
@app.route('/svm/run', methods=['POST'])
def run_svm_endpoint():
    run_svm()
    return jsonify({"message": "Modelo SVM executado com sucesso!"})

# Get (resultados)
@app.route('/svm/results', methods=['GET'])
def fetch_results():  
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o SVM primeiro."}), 400

    return jsonify({
        'accuracy': results['accuracy'],
        'image_url': f"{request.host_url}image"
    })

# Get (imagem)
@app.route('/svm/image', methods=['GET'])
def serve_image():
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o SVM primeiro."}), 400

    return send_file(io.BytesIO(results['image_data']), mimetype='image/png')

# Inicializa servidor
if __name__ == '__main__':
    app.run(debug=True)
