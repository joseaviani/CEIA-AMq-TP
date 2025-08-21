
#-----------------------------------------------------------------------


def procesar(X_train, X_test, y_train, y_test, classes, class_weights, X, y):
    print("\n")
    print("XGBOOST")
    # print("Tamaño X_train:", X_train.shape)
    # print("Tamaño X_test:", X_test.shape)
    # print("Tamaño y_train:", y_train.shape)
    # print("Tamaño y_test:", y_test.shape)
    # print("classes:", classes)
    # print("class_weights:", class_weights)


    #-----------------------------------------------------------------------

    procesar_v4(X, y)

    #-----------------------------------------------------------------------


#-----------------------------------------------------------------------


def procesar_v4(X, y):

    # ==========================================
    # XGBoost + GridSearchCV + StratifiedKFold
    # - Multiclase (quality 3..8)
    # - Recalcula class weights por fold (wrapper)
    # ==========================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report,
        confusion_matrix, ConfusionMatrixDisplay
    )
    from sklearn.base import BaseEstimator, ClassifierMixin

    from xgboost import XGBClassifier

    y_orig = y

    # XGBoost multiclase requiere etiquetas 0..K-1
    le = LabelEncoder()
    y = le.fit_transform(y_orig)
    classes_enc = np.unique(y)                 # p.ej. [0..5]
    classes_orig = le.inverse_transform(classes_enc)  # [3..8]

    # Hold-out test (lo que NO participa del GridSearch)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ---------- 2) Wrapper que recalcula sample_weight por fold ----------
    class XGBWithAutoWeights(BaseEstimator, ClassifierMixin):
        """
        Estimador scikit-learn compatible que:
        - Recibe los mismos hiperparámetros que XGBClassifier.
        - En fit(), computa class weights con y de ese fold y entrena XGBClassifier con sample_weight.
        """
        def __init__(self,
                    objective="multi:softprob",
                    num_class=None,
                    random_state=42,
                    tree_method="auto",
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    min_child_weight=1,
                    reg_lambda=1.0,
                    reg_alpha=0.0,
                    eval_metric="mlogloss"):
            self.objective = objective
            self.num_class = num_class
            self.random_state = random_state
            self.tree_method = tree_method
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.subsample = subsample
            self.colsample_bytree = colsample_bytree
            self.min_child_weight = min_child_weight
            self.reg_lambda = reg_lambda
            self.reg_alpha = reg_alpha
            self.eval_metric = eval_metric
            self.est_ = None

        def fit(self, X, y):
            # Recalcular class weights en el y de ESTE fold
            classes = np.unique(y)
            cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
            class_weights = dict(zip(classes, cw))
            sw = pd.Series(y).map(class_weights).values  # sample_weight por instancia

            # Construir y entrenar el XGBClassifier subyacente
            self.est_ = XGBClassifier(
                objective=self.objective,
                num_class=self.num_class if self.num_class is not None else len(classes),
                random_state=self.random_state,
                tree_method=self.tree_method,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                eval_metric=self.eval_metric
            )
            self.est_.fit(X, y, sample_weight=sw, verbose=False)
            return self

        def predict(self, X):
            return self.est_.predict(X)

        def predict_proba(self, X):
            return self.est_.predict_proba(X)

        def get_params(self, deep=True):
            # Necesario para que GridSearchCV pueda ver/ajustar hiperparámetros
            return {
                "objective": self.objective,
                "num_class": self.num_class,
                "random_state": self.random_state,
                "tree_method": self.tree_method,
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "min_child_weight": self.min_child_weight,
                "reg_lambda": self.reg_lambda,
                "reg_alpha": self.reg_alpha,
                "eval_metric": self.eval_metric
            }

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    # ---------- 3) CV + Grid ----------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    est = XGBWithAutoWeights(
        objective="multi:softprob",
        num_class=len(classes_enc),     # 6 clases
        random_state=42,
        tree_method="auto",             # usa 'hist' o 'gpu_hist' si lo soporta tu versión
        eval_metric="mlogloss"
    )

    param_grid = {
        "n_estimators":      [400, 800, 1200],
        "learning_rate":     [0.03, 0.05, 0.1],
        "max_depth":         [3, 4, 5],
        "subsample":         [0.8, 0.9, 1.0],
        "colsample_bytree":  [0.7, 0.8, 1.0],
        "min_child_weight":  [1, 3, 5],
        "reg_lambda":        [1.0, 2.0],
        "reg_alpha":         [0.0, 0.5]
    }

    gscv = GridSearchCV(
        estimator=est,
        param_grid=param_grid,
        scoring="f1_macro",          # métrica robusta al desbalance
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True                   # re-entrena el mejor sobre TODO el set de entrenamiento
    )

    gscv.fit(X_train, y_train)
    print("\n== Mejores hiperparámetros (CV) ==")
    print(gscv.best_params_)
    print(f"Mejor F1 macro (CV): {gscv.best_score_:.4f}")

    best_model = gscv.best_estimator_

    # ---------- 4) Evaluación en test ----------
    y_pred_enc = best_model.predict(X_test)

    # Volver a etiquetas originales (3..8) para reportar
    y_test_orig = le.inverse_transform(y_test)
    y_pred_orig = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test_orig, y_pred_orig)
    f1m = f1_score(y_test_orig, y_pred_orig, average="macro")

    print("\n== Resultados en Test ==")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1m:.4f}\n")
    print("== Classification Report ==")
    print(classification_report(y_test_orig, y_pred_orig, digits=3))

    cm = confusion_matrix(y_test_orig, y_pred_orig, labels=classes_orig)
    ConfusionMatrixDisplay(cm, display_labels=classes_orig).plot(cmap="Blues", values_format="d")
    plt.title("Matriz de confusión - XGBoost + GridSearchCV (mejor modelo)")
    plt.tight_layout()
    plt.show()

    # ---------- 5) Importancias de variables ----------
    # (Las expone el estimador interno)
    feat_imp = pd.Series(best_model.est_.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n== Top 10 features por importancia (XGBoost) ==")
    print(feat_imp.head(10))

    plt.figure(figsize=(8,5))
    feat_imp.head(10).iloc[::-1].plot(kind="barh")
    plt.title("Top 10 importancias de variables - XGBoost (mejor modelo)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.show()


#-----------------------------------------------------------------------