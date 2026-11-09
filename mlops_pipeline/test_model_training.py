"""
test_model_training.py
----------------------
Pruebas de las funciones de model_training_evaluation.py y del modelo guardado.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ft_engineering import preparar_datos, limpiar_datos, NUMERICAS, CATEGORICAS
from model_training_evaluation import summarize_classification, build_model, RUTA_MODELO


def test_summarize_classification_con_prediccion_perfecta():
    y_true = pd.Series([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
    r = summarize_classification(y_true, y_pred, y_proba)
    assert r["accuracy"] == 1.0
    assert r["recall_mora"] == 1.0
    assert r["precision_mora"] == 1.0
    assert r["f1_mora"] == 1.0
    assert r["roc_auc"] == 1.0


def test_summarize_classification_recall_parcial():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])      # detecta 1 de los 2 que no pagan
    y_proba = np.array([0.4, 0.6, 0.7, 0.9])
    r = summarize_classification(y_true, y_pred, y_proba)
    assert r["recall_mora"] == 0.5
    assert r["precision_mora"] == 1.0


def test_summarize_classification_no_falla_si_no_predice_la_clase_0():
    y_true = pd.Series([0, 1, 1])
    y_pred = np.array([1, 1, 1])
    r = summarize_classification(y_true, y_pred, np.array([0.6, 0.7, 0.8]))
    assert r["precision_mora"] == 0.0 and r["recall_mora"] == 0.0


def test_build_model_devuelve_pipeline_entrenado_y_metricas():
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()
    modelo = LogisticRegression(class_weight="balanced", max_iter=500)
    pipeline, resultado = build_model("LR de prueba", modelo, preprocesador,
                                      X_train, y_train, X_test, y_test)
    assert resultado["modelo"] == "LR de prueba"
    for clave in ["accuracy", "recall_mora", "f1_mora", "roc_auc", "cv_auc_media", "cv_auc_desv", "tiempo_seg"]:
        assert clave in resultado
    assert 0.5 < resultado["roc_auc"] <= 1.0
    assert len(pipeline.predict(X_test)) == len(X_test)


def test_modelo_guardado_carga_y_predice_probabilidades_validas():
    assert os.path.exists(RUTA_MODELO)
    modelo = joblib.load(RUTA_MODELO)
    X_train, X_test, y_train, y_test, _ = preparar_datos()
    probas = modelo.predict_proba(X_test.head(20))[:, 1]
    assert len(probas) == 20
    assert ((probas >= 0) & (probas <= 1)).all()


def test_script_de_entrenamiento_completo_sin_sobrescribir_archivos(monkeypatch, capsys):
    """Ejecuta model_training_evaluation.py de punta a punta, pero SIN escribir el modelo,
    el grafico ni el CSV (se reemplazan por funciones que no hacen nada)."""
    import runpy
    import matplotlib.pyplot as plt
    import model_training_evaluation

    guardados = []
    monkeypatch.setattr(joblib, "dump", lambda objeto, ruta: guardados.append(ruta))
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)

    runpy.run_path(model_training_evaluation.__file__, run_name="__main__")

    salida = capsys.readouterr().out
    assert "TABLA RESUMEN" in salida
    assert "MEJOR MODELO:" in salida
    assert guardados == [RUTA_MODELO]    # intento guardar el modelo en la ruta correcta
