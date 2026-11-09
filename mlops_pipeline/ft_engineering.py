"""
ft_engineering.py
-----------------
Primera componente del pipeline MLOps: INGENIERIA DE CARACTERISTICAS.

Que hace este archivo:
  1. Lee Base_de_datos.csv
  2. Limpia los datos (las mismas reglas del EDA)
  3. Divide en entrenamiento (80 %) y prueba (20 %)
  4. Construye el preprocesador (imputacion + escalado + encoding)

Los otros scripts (entrenamiento, monitoreo y API) importan las funciones de aqui,
asi todos usan EXACTAMENTE las mismas reglas de limpieza.
"""

# ---------------------------------------------------------------
# 1. IMPORTAR LIBRERIAS
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------------------
# 2. CONFIGURACION (listas de columnas segun lo decidido en el EDA)
# ---------------------------------------------------------------
# Ruta del CSV: se calcula a partir de la ubicacion de este archivo,
# asi funciona sin importar desde que carpeta ejecutes el script.
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(CARPETA_ACTUAL, "..", "Base_de_datos.csv")

OBJETIVO = "Pago_atiempo"

# Columnas eliminadas en el EDA (y la fecha, que se usa solo para monitoreo)
COLUMNAS_ELIMINAR = [
    "puntaje",                # fuga de datos
    "tipo_credito",           # irrelevante
    "salario_cliente",        # irrelevante
    "cant_creditosvigentes",  # irrelevante
    "saldo_mora_codeudor",    # irrelevante
    "cuota_pactada",          # redundante con capital_prestado
    "fecha_prestamo",         # no es predictor (se usa en el monitoreo)
]

# Columnas que usa el modelo
NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "total_otros_prestamos",
    "puntaje_datacredito", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "creditos_sectorFinanciero", "creditos_sectorCooperativo",
    "creditos_sectorReal", "promedio_ingresos_datacredito",
]
CATEGORICAS = ["tipo_laboral", "tendencia_ingresos"]

# Valores validos de tendencia_ingresos
TENDENCIAS_VALIDAS = ["Creciente", "Decreciente", "Estable"]


# ---------------------------------------------------------------
# 3. FUNCION DE LIMPIEZA
# ---------------------------------------------------------------
def limpiar_datos(df):
    """Aplica las reglas de limpieza del EDA y devuelve solo las columnas del modelo."""
    df = df.copy()  # trabajamos sobre una copia para no modificar el original

    # 3.1 Unificar la forma de escribir los nulos
    valores_nulos = ["", " ", "NA", "N/A", "na", "null", "NULL", "None", "nan", "NaN", "-"]
    df = df.replace(valores_nulos, np.nan)

    # 3.2 Si falta alguna columna del modelo (por ejemplo en la API), la creamos vacia
    for col in NUMERICAS + CATEGORICAS:
        if col not in df.columns:
            df[col] = np.nan

    # 3.3 Asegurar que las numericas sean numeros (si hay texto raro pasa a NaN)
    for col in NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3.4 Reglas de validacion definidas en el EDA
    df.loc[(df["edad_cliente"] < 18) | (df["edad_cliente"] > 90), "edad_cliente"] = np.nan
    df.loc[df["puntaje_datacredito"] <= 0, "puntaje_datacredito"] = np.nan

    # 3.5 tendencia_ingresos: solo valores validos; lo demas -> "Sin_dato"
    # Primero la pasamos a tipo texto: si la columna estaba vacia, pandas la trataria como
    # numerica y no permitiria escribir texto en ella.
    df["tendencia_ingresos"] = df["tendencia_ingresos"].astype("object")
    df.loc[~df["tendencia_ingresos"].isin(TENDENCIAS_VALIDAS), "tendencia_ingresos"] = "Sin_dato"

    # 3.6 Quedarnos solo con las columnas del modelo (y el objetivo si existe)
    columnas = NUMERICAS + CATEGORICAS
    if OBJETIVO in df.columns:
        columnas = columnas + [OBJETIVO]
    return df[columnas]


# ---------------------------------------------------------------
# 4. FUNCION QUE CONSTRUYE EL PREPROCESADOR
# ---------------------------------------------------------------
def construir_preprocesador():
    """Crea el ColumnTransformer: imputacion + escalado (numericas) y One-Hot (categoricas)."""

    # Para las numericas: 1) rellenar nulos con la MEDIA, 2) escalar (media 0, desv. 1)
    pipeline_numerico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="mean")),
        ("escalador", StandardScaler()),
    ])

    # Para las categoricas: convertir cada categoria en columnas 0/1 (One-Hot)
    # handle_unknown="ignore": si llega una categoria nueva, no falla.
    pipeline_categorico = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # ColumnTransformer aplica cada pipeline a su grupo de columnas
    preprocesador = ColumnTransformer(transformers=[
        ("numericas", pipeline_numerico, NUMERICAS),
        ("categoricas", pipeline_categorico, CATEGORICAS),
    ])
    return preprocesador


# ---------------------------------------------------------------
# 5. FUNCION PRINCIPAL: DEVUELVE LOS CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
# ---------------------------------------------------------------
def preparar_datos(ruta_csv=RUTA_CSV):
    """Lee, limpia y divide los datos. Devuelve X_train, X_test, y_train, y_test, preprocesador."""
    df = pd.read_csv(ruta_csv)
    df = limpiar_datos(df)

    # Separar predictores (X) y objetivo (y)
    X = df.drop(columns=[OBJETIVO])
    y = df[OBJETIVO]

    # stratify=y mantiene la misma proporcion de buenos/malos pagadores en train y test
    # (importante porque solo ~5 % son malos pagadores)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocesador = construir_preprocesador()
    return X_train, X_test, y_train, y_test, preprocesador


# ---------------------------------------------------------------
# 6. PRUEBA RAPIDA: se ejecuta solo si corres "python ft_engineering.py"
# ---------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()

    print("Entrenamiento:", X_train.shape, "| Prueba:", X_test.shape)
    print("% buenos pagadores en train:", round(y_train.mean() * 100, 2))
    print("% buenos pagadores en test :", round(y_test.mean() * 100, 2))

    # Ajustamos el preprocesador SOLO con train y transformamos ambos conjuntos
    X_train_proc = preprocesador.fit_transform(X_train)
    X_test_proc = preprocesador.transform(X_test)
    print("Columnas despues de preprocesar:", X_train_proc.shape[1])
    print("Nulos restantes en train:", np.isnan(X_train_proc).sum())
