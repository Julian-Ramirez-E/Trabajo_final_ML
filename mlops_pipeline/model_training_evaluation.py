"""
model_training_evaluation.py
----------------------------
Segunda componente del pipeline MLOps: ENTRENAMIENTO Y EVALUACION DE MODELOS.

Que hace este archivo:
  1. Trae los datos preparados desde ft_engineering.py
  2. Entrena 3 modelos: Regresion Logistica, Random Forest y Gradient Boosting
  3. Los evalua (metricas en prueba + validacion cruzada + tiempo)
  4. Muestra tablas y graficos comparativos
  5. Elige el mejor y lo guarda en modelo_final.joblib

Como el 95 % de los clientes paga a tiempo, medimos todo pensando en el grupo
que mas importa para el negocio: los que NO pagan a tiempo (clase 0).
"""

# ---------------------------------------------------------------
# 1. IMPORTAR LIBRERIAS
# ---------------------------------------------------------------
import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)

# Importamos lo que construimos en ft_engineering.py
from ft_engineering import preparar_datos

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA_ACTUAL, "modelo_final.joblib")
RUTA_GRAFICO = os.path.join(CARPETA_ACTUAL, "comparacion_modelos.png")
RUTA_RESULTADOS = os.path.join(CARPETA_ACTUAL, "resultados_modelos.csv")


# ---------------------------------------------------------------
# 2. FUNCIONES REUTILIZABLES
# ---------------------------------------------------------------
def summarize_classification(y_true, y_pred, y_proba):
    """Calcula las metricas de clasificacion. La clase de interes es 0 (no paga a tiempo)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # pos_label=0: medimos que tan bien detectamos a los que NO pagan
        "precision_mora": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_mora": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_mora": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        # y_proba es la probabilidad de pagar a tiempo (clase 1); el AUC no depende de la clase elegida
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def build_model(nombre, modelo, preprocesador, X_train, y_train, X_test, y_test):
    """Arma el pipeline (preprocesador + modelo), lo entrena y lo evalua."""
    # Pipeline: primero transforma los datos y luego aplica el modelo
    # clone() crea una copia limpia del preprocesador, asi cada modelo tiene el suyo
    pipeline = Pipeline(steps=[
        ("preprocesador", clone(preprocesador)),
        ("modelo", modelo),
    ])

    # Medimos cuanto tarda en entrenar (escalabilidad)
    inicio = time.time()
    pipeline.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio

    # Predicciones sobre el conjunto de prueba
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    resultado = summarize_classification(y_test, y_pred, y_proba)

    # Validacion cruzada (5 pliegues) solo con datos de entrenamiento: mide la CONSISTENCIA
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_cv = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    resultado["cv_auc_media"] = auc_cv.mean()
    resultado["cv_auc_desv"] = auc_cv.std()
    resultado["tiempo_seg"] = tiempo_entrenamiento
    resultado["modelo"] = nombre

    return pipeline, resultado


# ---------------------------------------------------------------
# 3. PROGRAMA PRINCIPAL
# ---------------------------------------------------------------
if __name__ == "__main__":

    # 3.1 Datos preparados
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()
    print("Datos listos -> train:", X_train.shape, "| test:", X_test.shape)

    # 3.2 Modelos a comparar
    # class_weight="balanced" le da mas peso a los malos pagadores (la clase minoritaria)
    modelos = {
        "Regresion Logistica": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                                            class_weight="balanced", random_state=42),
    }

    # 3.3 Entrenar y evaluar cada modelo
    pipelines = {}
    resultados = []
    for nombre, modelo in modelos.items():
        print("Entrenando:", nombre, "...")
        pipeline, resultado = build_model(nombre, modelo, preprocesador, X_train, y_train, X_test, y_test)
        pipelines[nombre] = pipeline
        resultados.append(resultado)

    # 3.4 Tabla resumen
    tabla = pd.DataFrame(resultados).set_index("modelo").round(4)
    print("\n===== TABLA RESUMEN (conjunto de prueba) =====")
    print(tabla.to_string())

    # 3.5 Seleccion del mejor modelo
    # Puntaje = AUC promedio en validacion cruzada - su desviacion (premia rendimiento Y consistencia)
    tabla["puntaje_seleccion"] = tabla["cv_auc_media"] - tabla["cv_auc_desv"]
    mejor_nombre = tabla["puntaje_seleccion"].idxmax()
    mejor_pipeline = pipelines[mejor_nombre]
    print("\nMEJOR MODELO:", mejor_nombre)

    # 3.6 Graficos comparativos
    fig, ejes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Barras con metricas principales
    tabla[["recall_mora", "precision_mora", "f1_mora", "roc_auc"]].plot(kind="bar", ax=ejes[0])
    ejes[0].set_title("Metricas en prueba (clase 'no paga a tiempo')")
    ejes[0].set_ylim(0, 1)
    ejes[0].tick_params(axis="x", rotation=15)

    # (b) Curvas ROC
    for nombre, pipeline in pipelines.items():
        proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ejes[1].plot(fpr, tpr, label=f"{nombre} (AUC={tabla.loc[nombre, 'roc_auc']:.3f})")
    ejes[1].plot([0, 1], [0, 1], "k--", label="Azar")
    ejes[1].set_title("Curvas ROC")
    ejes[1].set_xlabel("Tasa de falsos positivos")
    ejes[1].set_ylabel("Tasa de verdaderos positivos")
    ejes[1].legend()

    # (c) Matriz de confusion del mejor modelo
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, mejor_pipeline.predict(X_test)),
        display_labels=["No paga (0)", "Paga (1)"],
    ).plot(ax=ejes[2], colorbar=False)
    ejes[2].set_title(f"Matriz de confusion - {mejor_nombre}")

    plt.tight_layout()
    plt.savefig(RUTA_GRAFICO, dpi=120)
    plt.close()
    print("Grafico guardado en:", RUTA_GRAFICO)

    # 3.7 Guardar resultados y el mejor modelo
    tabla.to_csv(RUTA_RESULTADOS)
    joblib.dump(mejor_pipeline, RUTA_MODELO)
    print("Modelo guardado en:", RUTA_MODELO)
