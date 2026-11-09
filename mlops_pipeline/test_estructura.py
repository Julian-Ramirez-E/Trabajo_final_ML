"""
test_estructura.py
------------------
Verifica que la estructura de carpetas y archivos del proyecto NO haya cambiado.
Jenkins ejecuta esta prueba para validar el repositorio antes de pasar a produccion.
"""
import os

# Raiz del repositorio (una carpeta arriba de mlops_pipeline)
CARPETA_PIPELINE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(CARPETA_PIPELINE)

ARCHIVOS_PIPELINE = [
    "Cargar_datos.ipynb",
    "comprension_eda.ipynb",
    "ft_engineering.py",
    "model_training_evaluation.py",
    "model_deploy.py",
    "model_monitoring.py",
]

ARCHIVOS_RAIZ = [
    "Base_de_datos.csv",
    "requirements.txt",
    ".gitignore",
    "setup.bat",
    "readme.md",
    "Dockerfile",
]


def test_carpeta_mlops_pipeline_existe():
    assert os.path.isdir(CARPETA_PIPELINE)
    assert os.path.basename(CARPETA_PIPELINE) == "mlops_pipeline"


def test_archivos_de_mlops_pipeline_existen():
    faltantes = [a for a in ARCHIVOS_PIPELINE if not os.path.isfile(os.path.join(CARPETA_PIPELINE, a))]
    assert faltantes == [], f"Faltan archivos en mlops_pipeline: {faltantes}"


def test_archivos_de_la_raiz_existen():
    faltantes = [a for a in ARCHIVOS_RAIZ if not os.path.isfile(os.path.join(RAIZ, a))]
    assert faltantes == [], f"Faltan archivos en la raiz: {faltantes}"


def test_modelo_entrenado_existe():
    assert os.path.isfile(os.path.join(CARPETA_PIPELINE, "modelo_final.joblib"))
