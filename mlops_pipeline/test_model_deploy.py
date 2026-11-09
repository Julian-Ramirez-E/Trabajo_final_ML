"""
test_model_deploy.py
--------------------
Pruebas de la API (model_deploy.py) con el cliente de pruebas de FastAPI.
No hace falta levantar ningun servidor.
"""
import pytest
from fastapi.testclient import TestClient

import model_deploy
from model_deploy import app

cliente_api = TestClient(app)

CLIENTE_VALIDO = {
    "capital_prestado": 1852560, "plazo_meses": 12, "edad_cliente": 32, "tipo_laboral": "Empleado",
    "huella_consulta": 2, "puntaje_datacredito": 795, "tendencia_ingresos": "Creciente",
}

CAMPOS_RESPUESTA = {"probabilidad_pago", "probabilidad_mora", "prediccion", "resultado"}


def test_inicio():
    r = cliente_api.get("/")
    assert r.status_code == 200
    assert r.json()["documentacion"] == "/docs"


def test_health_indica_modelo_cargado():
    r = cliente_api.get("/health")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "modelo_cargado": True}


def test_predict_un_cliente():
    r = cliente_api.post("/predict", json=[CLIENTE_VALIDO])
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["total"] == 1
    pred = cuerpo["predicciones"][0]
    assert set(pred) == CAMPOS_RESPUESTA
    assert 0 <= pred["probabilidad_pago"] <= 1
    assert abs(pred["probabilidad_pago"] + pred["probabilidad_mora"] - 1) < 0.001
    assert pred["prediccion"] in (0, 1)
    assert pred["resultado"] in ("Paga a tiempo", "Riesgo de mora")


def test_predict_por_lotes_con_datos_sucios():
    sucio = {"capital_prestado": 5000000, "plazo_meses": 36, "edad_cliente": 122,
             "tipo_laboral": "Independiente", "puntaje_datacredito": -5, "tendencia_ingresos": "8315"}
    r = cliente_api.post("/predict", json=[CLIENTE_VALIDO, sucio, CLIENTE_VALIDO])
    assert r.status_code == 200
    assert r.json()["total"] == 3


def test_predict_lista_vacia_da_400():
    assert cliente_api.post("/predict", json=[]).status_code == 400


@pytest.mark.parametrize("cambio", [
    {"tipo_laboral": "Pensionado"},   # valor no permitido
    {"plazo_meses": 100},             # fuera de rango
    {"plazo_meses": 0},
    {"capital_prestado": -5},         # debe ser mayor a 0
    {"huella_consulta": -1},          # no puede ser negativo
])
def test_predict_datos_invalidos_dan_422(cambio):
    r = cliente_api.post("/predict", json=[{**CLIENTE_VALIDO, **cambio}])
    assert r.status_code == 422


def test_predict_falta_campo_obligatorio_da_422():
    assert cliente_api.post("/predict", json=[{"capital_prestado": 1000}]).status_code == 422


def test_predict_csv_valido():
    csv = ("capital_prestado,plazo_meses,edad_cliente,tipo_laboral\n"
           "1500000,12,30,Empleado\n2500000,24,50,Independiente\n")
    r = cliente_api.post("/predict_csv", files={"archivo": ("clientes.csv", csv.encode(), "text/csv")})
    assert r.status_code == 200
    assert r.json()["total"] == 2


def test_predict_csv_sin_columnas_obligatorias_da_422():
    r = cliente_api.post("/predict_csv", files={"archivo": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422
    assert "Faltan columnas obligatorias" in r.json()["detail"]


def test_predict_csv_sin_filas_da_400():
    encabezado = b"capital_prestado,plazo_meses,edad_cliente,tipo_laboral\n"
    r = cliente_api.post("/predict_csv", files={"archivo": ("x.csv", encabezado, "text/csv")})
    assert r.status_code == 400


def test_predict_csv_ilegible_da_400():
    r = cliente_api.post("/predict_csv", files={"archivo": ("x.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_sin_modelo_cargado_responde_503(monkeypatch):
    monkeypatch.setattr(model_deploy, "modelo", None)   # simulamos que el modelo no esta
    r = cliente_api.post("/predict", json=[CLIENTE_VALIDO])
    assert r.status_code == 503
    assert cliente_api.get("/health").json()["modelo_cargado"] is False


def test_script_de_la_api_arranca_uvicorn(monkeypatch):
    """Ejecutar 'python model_deploy.py' debe llamar a uvicorn (lo reemplazamos para no abrir un servidor)."""
    import runpy
    import uvicorn
    llamadas = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: llamadas.append(kwargs))
    runpy.run_path(model_deploy.__file__, run_name="__main__")
    assert llamadas == [{"host": "127.0.0.1", "port": 8000}]


def test_predict_csv_sin_columnas_opcionales_funciona():
    """Regresion: un CSV minimo (sin tendencia_ingresos ni saldos) no debe romper la API."""
    csv = "capital_prestado,plazo_meses,edad_cliente,tipo_laboral\n1500000,12,30,Empleado\n"
    r = cliente_api.post("/predict_csv", files={"archivo": ("m.csv", csv.encode(), "text/csv")})
    assert r.status_code == 200
