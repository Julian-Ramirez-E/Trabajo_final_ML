"""
model_deploy.py
---------------
Cuarta componente del pipeline MLOps: DESPLIEGUE DEL MODELO COMO API.

Que hace este archivo:
  1. Carga el modelo entrenado (modelo_final.joblib)
  2. Expone una API con FastAPI para pedir predicciones:
       GET  /health       -> confirma que la API y el modelo estan vivos
       POST /predict      -> recibe uno o varios clientes en JSON y devuelve predicciones
       POST /predict_csv  -> recibe un archivo CSV y devuelve predicciones (por lotes)
  3. Aplica las mismas reglas de limpieza del entrenamiento (ft_engineering.limpiar_datos)

Como ejecutarlo:
  uvicorn model_deploy:app --app-dir mlops_pipeline --reload
  Luego abre en el navegador:  http://127.0.0.1:8000/docs  (documentacion interactiva)
"""

# ---------------------------------------------------------------
# 1. IMPORTAR LIBRERIAS
# ---------------------------------------------------------------
import os
from typing import List, Literal, Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Reutilizamos la limpieza y las listas de columnas del entrenamiento
from ft_engineering import limpiar_datos, NUMERICAS, CATEGORICAS

# ---------------------------------------------------------------
# 2. CONFIGURACION Y CARGA DEL MODELO
# ---------------------------------------------------------------
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
# La ruta puede cambiarse con la variable de entorno RUTA_MODELO (util en Docker)
RUTA_MODELO = os.getenv("RUTA_MODELO", os.path.join(CARPETA_ACTUAL, "modelo_final.joblib"))

# Columnas que el cliente DEBE enviar (las demas pueden faltar y se imputan con la media)
COLUMNAS_OBLIGATORIAS = ["capital_prestado", "plazo_meses", "edad_cliente", "tipo_laboral"]

# Cargamos el modelo una sola vez, al arrancar la API
modelo = None
if os.path.exists(RUTA_MODELO):
    modelo = joblib.load(RUTA_MODELO)

app = FastAPI(
    title="API de prediccion de pago a tiempo",
    description="Predice si un cliente pagara su credito a tiempo. Proyecto Final Machine Learning 2026.",
    version="1.0.0",
)


# ---------------------------------------------------------------
# 3. ESQUEMA DE DATOS DE ENTRADA
# ---------------------------------------------------------------
# FastAPI usa pydantic para VALIDAR lo que envia el usuario: si falta un campo
# obligatorio o el tipo es incorrecto, responde con un error claro (codigo 422).
class Cliente(BaseModel):
    # --- Obligatorios ---
    capital_prestado: float = Field(gt=0, description="Monto prestado (mayor a 0)")
    plazo_meses: int = Field(ge=1, le=60, description="Plazo en meses (1 a 60)")
    edad_cliente: int = Field(description="Edad del cliente (si esta fuera de 18-90 se trata como dato faltante)")
    tipo_laboral: Literal["Empleado", "Independiente"]
    # --- Opcionales (si no se envian, el modelo los rellena con la media de entrenamiento) ---
    total_otros_prestamos: Optional[float] = Field(default=None, ge=0)
    puntaje_datacredito: Optional[float] = None
    huella_consulta: Optional[int] = Field(default=None, ge=0)
    saldo_mora: Optional[float] = Field(default=None, ge=0)
    saldo_total: Optional[float] = Field(default=None, ge=0)
    saldo_principal: Optional[float] = Field(default=None, ge=0)
    creditos_sectorFinanciero: Optional[int] = Field(default=None, ge=0)
    creditos_sectorCooperativo: Optional[int] = Field(default=None, ge=0)
    creditos_sectorReal: Optional[int] = Field(default=None, ge=0)
    promedio_ingresos_datacredito: Optional[float] = Field(default=None, ge=0)
    tendencia_ingresos: Optional[str] = Field(default=None, description="Creciente, Decreciente o Estable")

    # Ejemplo que aparece en la documentacion interactiva (/docs)
    model_config = {
        "json_schema_extra": {
            "example": {
                "capital_prestado": 1852560,
                "plazo_meses": 12,
                "edad_cliente": 32,
                "tipo_laboral": "Empleado",
                "total_otros_prestamos": 1000000,
                "puntaje_datacredito": 795,
                "huella_consulta": 2,
                "saldo_mora": 0,
                "saldo_total": 0,
                "creditos_sectorFinanciero": 0,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": 916148,
                "tendencia_ingresos": "Creciente",
            }
        }
    }


# ---------------------------------------------------------------
# 4. LOGICA DE PREDICCION (compartida por /predict y /predict_csv)
# ---------------------------------------------------------------
def hacer_predicciones(df):
    """Recibe un DataFrame con clientes, lo limpia y devuelve la lista de predicciones."""
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no esta cargado. Ejecuta model_training_evaluation.py")

    # Misma limpieza que en el entrenamiento; nos quedamos solo con las columnas del modelo
    datos = limpiar_datos(df)[NUMERICAS + CATEGORICAS]

    probabilidades = modelo.predict_proba(datos)[:, 1]   # probabilidad de pagar a tiempo (clase 1)
    clases = modelo.predict(datos)                       # 1 = paga a tiempo, 0 = riesgo de mora

    resultados = []
    for prob, clase in zip(probabilidades, clases):
        resultados.append({
            "probabilidad_pago": round(float(prob), 4),
            "probabilidad_mora": round(1 - float(prob), 4),
            "prediccion": int(clase),
            "resultado": "Paga a tiempo" if clase == 1 else "Riesgo de mora",
        })
    return {"total": len(resultados), "predicciones": resultados}


# ---------------------------------------------------------------
# 5. ENDPOINTS
# ---------------------------------------------------------------
@app.get("/")
def inicio():
    """Mensaje de bienvenida."""
    return {"mensaje": "API de prediccion de pago a tiempo", "documentacion": "/docs"}


@app.get("/health")
def salud():
    """Permite comprobar que la API esta viva y que el modelo se cargo."""
    return {"estado": "ok", "modelo_cargado": modelo is not None}


@app.post("/predict")
def predecir(clientes: List[Cliente]):
    """Recibe una LISTA de clientes en JSON (uno o muchos) y devuelve una prediccion por cliente."""
    if len(clientes) == 0:
        raise HTTPException(status_code=400, detail="Envia al menos un cliente.")

    # Convertimos la lista de clientes en una tabla (DataFrame)
    df = pd.DataFrame([cliente.model_dump() for cliente in clientes])
    return hacer_predicciones(df)


@app.post("/predict_csv")
def predecir_csv(archivo: UploadFile = File(...)):
    """Recibe un archivo CSV con clientes (una fila por cliente) y devuelve las predicciones."""
    try:
        df = pd.read_csv(archivo.file)
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer el archivo. Verifica que sea un CSV valido.")

    if df.empty:
        raise HTTPException(status_code=400, detail="El archivo CSV no tiene filas.")

    faltantes = [col for col in COLUMNAS_OBLIGATORIAS if col not in df.columns]
    if faltantes:
        raise HTTPException(status_code=422, detail=f"Faltan columnas obligatorias: {faltantes}")

    return hacer_predicciones(df)


# ---------------------------------------------------------------
# 6. EJECUCION DIRECTA: python mlops_pipeline/model_deploy.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
