import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV



class GradientBoostingModel:

    def __init__(self, X_train, X_test, y_train, y_test, classes, class_weights):
        self.__X_train = X_train
        self.__X_test = X_test
        self.__y_train = y_train
        self.__y_test = y_test
        self.__classes = classes
        self.__class_weights = class_weights



    def __fit_predict(self):

        # Pesos por clase (balanceo)
        sample_weight_train = self.__y_train.map(self.__class_weights).values  # vector por instancia (train)

        # CV estratificado
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Estimador base
        gb = GradientBoostingClassifier(random_state=42)

        # Espacio de hiperparámetros
        param_grid = {
            "n_estimators":   [200, 300, 500],
            # "n_estimators":   [200],
            "learning_rate":  [0.05, 0.1],
            # "learning_rate":  [0.05],
            "max_depth":      [2, 3, 4],
            # "max_depth":      [2],
            "subsample":      [0.7, 0.9, 1.0],
            # "subsample":      [0.7],
            "min_samples_leaf": [1, 3, 5],
            # "min_samples_leaf": [1],
            "max_features":     [None, "sqrt"]
        }

        # Grid Search con Cross Validation
        gscv = GridSearchCV(
            estimator=gb,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=skf,
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        # Entrenamiento con pesos por instancia
        print("Comenzando entrenamiento ...\n")
        gscv.fit(self.__X_train, self.__y_train, **{"sample_weight": sample_weight_train})
        print("\nEntrenamiento finalizado.\n")

        print("Mejores hiperparámetros:")
        print(gscv.best_params_)
        print(f"\nMejor F1 macro (CV): {gscv.best_score_:.4f}")

        # Mejor modelo
        best_model = gscv.best_estimator_

        # Evaluación en test
        y_pred = best_model.predict(self.__X_test)

        return y_pred, best_model.feature_importances_



    def __mostrar_resultados(self, y_pred):
        acc = accuracy_score(self.__y_test, y_pred)
        f1m = f1_score(self.__y_test, y_pred, average="macro")

        print("\nResultados en train:")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1 macro: {f1m:.4f}\n")
        print("Classification Report:")
        report = classification_report(self.__y_test, y_pred, digits=3)
        print(report, flush=True)

    

    def __mostrar_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.__y_test, y_pred, labels=self.__classes)
        ConfusionMatrixDisplay(cm, display_labels=self.__classes).plot(cmap="Blues", values_format="d")
        plt.title("Matriz de confusión")
        plt.tight_layout()
        plt.show()



    def __mostrar_feature_importances(self, feature_importances):
        importances = pd.Series(feature_importances, index=self.__X_train.columns).sort_values(ascending=False)
        print("\nPrimeros 10 features por importancia:")
        print(importances.head(10))
        plt.figure(figsize=(8,5))
        importances.head(10).iloc[::-1].plot(kind="barh")
        plt.title("Primeros 10 features por importancia")
        plt.xlabel("Importancia")
        plt.tight_layout()
        plt.show()



    def procesar(self):

        print("\nComenzando Gradient Boosting ...\n")

        y_pred, feature_importances = self.__fit_predict()

        self.__mostrar_resultados(y_pred)

        self.__mostrar_confusion_matrix(y_pred)

        self.__mostrar_feature_importances(feature_importances)

        print("\nGradient Boosting finalizado.\n")




















