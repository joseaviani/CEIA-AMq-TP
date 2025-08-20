import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)



def procesar(X_train, X_test, y_train, y_test, classes, class_weights):
  print("GRADIENT BOOSTING")
  print("Tamaño X_train:", X_train.shape)
  print("Tamaño X_test:", X_test.shape)
  print("Tamaño y_train:", y_train.shape)
  print("Tamaño y_test:", y_test.shape)
  print("classes:", classes)
  print("class_weights:", class_weights)



    

  # Vector de pesos por instancia (train)
  sample_weight_train = y_train.map(class_weights).values

  # ========== 4) Modelo: Gradient Boosting ==========
  gb = GradientBoostingClassifier(
      random_state=42,
      n_estimators=300,
      learning_rate=0.05,
      max_depth=3,
      subsample=0.9,
      max_features=None
  )

  # Entrenamiento con pesos por instancia
  gb.fit(X_train, y_train, sample_weight=sample_weight_train)

  # ========== 5) Evaluación ==========
  y_pred = gb.predict(X_test)

  acc = accuracy_score(y_test, y_pred)
  f1m = f1_score(y_test, y_pred, average="macro")

  print("== Resultados en Test ==")
  print(f"Accuracy: {acc:.4f}")
  print(f"F1 macro: {f1m:.4f}\n")

  print("== Classification Report ==")
  print(classification_report(y_test, y_pred))

  # Matriz de confusión
  cm = confusion_matrix(y_test, y_pred, labels=classes)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
  disp.plot(values_format="d", cmap="Blues")
  plt.title("Matriz de confusión - Gradient Boosting")
  plt.tight_layout()
  plt.show()

  # ========== 6) Importancias de variables ==========
  importances = pd.Series(gb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
  print("\n== Top 10 features por importancia ==")
  print(importances.head(10))

  plt.figure(figsize=(8,5))
  importances.head(10).iloc[::-1].plot(kind="barh")
  plt.title("Top 10 importancias de variables")
  plt.xlabel("Importancia")
  plt.tight_layout()
  plt.show()



