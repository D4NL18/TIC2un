# Passo 1: Imports
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Passo 7: Importar o Modelo SVM
from sklearn.svm import SVC

app = Flask(__name__)
CORS(app)

results = None

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
    svm_model = SVC(kernel='linear', C=1.0)

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

    # Salvar a imagem
    img_path = 'confusion_matrix.png'
    plt.savefig(img_path)
    plt.close()  # Fecha a figura para liberar memória

    # Passo 13: Ajustar Hiperparâmetros
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf', 'poly']
    }
    
    grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # Passo 14: Documentar e Salvar o Modelo
    joblib.dump(svm_model, 'svm_model_iris.pkl')

    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': classification_rep,
        'best_params': best_params,
        'image_path': img_path  # Adiciona o caminho da imagem ao resultado
    }

# Post (executar SVM)
@app.route('/run_svm', methods=['POST'])
def run_svm_endpoint():
    run_svm()
    return jsonify({"message": "Modelo SVM executado com sucesso!"})

# Get (Enviar resultados)
@app.route('/results', methods=['GET'])
def get_results():
    if results is None:
        return jsonify({"error": "Nenhum resultado disponível. Execute o SVM primeiro."}), 400

    # Retorna a acurácia e o caminho da imagem da matriz de confusão
    return jsonify({
        'accuracy': results['accuracy'],
        'image_path': results['image_path']  # Envia o caminho da imagem
    })

# Rota para servir a imagem da matriz de confusão
@app.route('/images/<path:filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory('.', filename)

# Inicializa o servidor
if __name__ == '__main__':
    app.run(debug=True)
