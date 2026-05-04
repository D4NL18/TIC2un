import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

from models.svm_model import SVMResult
from config import WEIGHTS_DIR_SVM

plt.switch_backend('Agg')


class SVMService:
    def run(self) -> SVMResult:
        iris = load_iris()
        X = iris.data
        y = iris.target

        df = pd.DataFrame(X, columns=iris.feature_names)
        df['target'] = y
        print(df.head())
        print(df.info())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=iris.target_names, yticklabels=iris.target_names)
        plt.xlabel('Previsto')
        plt.ylabel('Verdadeiro')
        plt.title('Matriz de Confusão')

        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        plt.close()

        param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(svm_model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_svm_model = grid_search.best_estimator_

        if not os.path.exists(WEIGHTS_DIR_SVM):
            os.makedirs(WEIGHTS_DIR_SVM)

        model_filename = os.path.join(WEIGHTS_DIR_SVM, 'best_svm_model.joblib')
        joblib.dump(best_svm_model, model_filename)
        print(f"Modelo treinado salvo como {model_filename}")

        return SVMResult(
            accuracy=accuracy,
            confusion_matrix=conf_matrix.tolist(),
            classification_report=classification_rep,
            image_data=img_bytes.getvalue()
        )
