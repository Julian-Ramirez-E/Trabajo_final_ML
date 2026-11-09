"""
test_model_monitoring.py
------------------------
Pruebas de las metricas de data drift y del tablero (model_monitoring.py).
"""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import model_monitoring as mm
from ft_engineering import NUMERICAS, CATEGORICAS


def crear_datos(n=600, desplazamiento=0.0, semilla=0, fecha_inicio="2025-01-01"):
    """Crea datos sinteticos con todas las columnas que usa el monitoreo."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({col: rng.normal(100 + desplazamiento, 10, n) for col in NUMERICAS})
    df["tipo_laboral"] = rng.choice(["Empleado", "Independiente"], n, p=[0.6, 0.4])
    df["tendencia_ingresos"] = rng.choice(["Creciente", "Estable", "Decreciente", "Sin_dato"], n)
    df[mm.VARIABLE_PREDICCION] = rng.uniform(0, 1, n)
    df["fecha_prestamo"] = pd.date_range(fecha_inicio, periods=n, freq="D")
    return df


# ---------------- PSI / JS ----------------
def test_psi_de_distribuciones_iguales_es_cero():
    p = np.array([0.2, 0.3, 0.5])
    assert mm.calcular_psi(p, p) == pytest.approx(0, abs=1e-9)


def test_psi_aumenta_cuando_las_distribuciones_difieren():
    p_ref = np.array([0.5, 0.3, 0.2])
    p_act = np.array([0.1, 0.3, 0.6])
    assert mm.calcular_psi(p_ref, p_act) > mm.PSI_ROJO


def test_psi_no_falla_con_proporciones_en_cero():
    resultado = mm.calcular_psi(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert np.isfinite(resultado)


def test_js_cero_si_son_iguales_y_uno_si_son_totalmente_distintas():
    assert mm.calcular_js(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0, abs=1e-9)
    assert mm.calcular_js(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1, abs=1e-9)


# ---------------- proporciones ----------------
def test_proporciones_numericas_suman_uno():
    rng = np.random.default_rng(1)
    p_ref, p_act = mm.proporciones_numericas(pd.Series(rng.normal(0, 1, 500)), pd.Series(rng.normal(3, 1, 500)))
    assert p_ref.sum() == pytest.approx(1)
    assert p_act.sum() == pytest.approx(1)


def test_proporciones_numericas_variable_constante():
    p_ref, p_act = mm.proporciones_numericas(pd.Series([5.0] * 50), pd.Series([5.0] * 50))
    assert p_ref.sum() == pytest.approx(1) and p_act.sum() == pytest.approx(1)


def test_proporciones_categoricas_incluyen_categorias_nuevas():
    ref = pd.Series(["A", "A", "B", "B"])
    act = pd.Series(["A", "C", "C", "C"])
    p_ref, p_act = mm.proporciones_categoricas(ref, act)
    assert len(p_ref) == len(p_act) == 3          # A, B, C
    assert p_ref.sum() == pytest.approx(1)


# ---------------- Chi-cuadrado ----------------
def test_chi2_detecta_cambio_en_categorias():
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    act = pd.Series(["A"] * 900 + ["B"] * 100)
    chi2, p_valor = mm.calcular_chi2(ref, act)
    assert chi2 > 0 and p_valor < 0.05


def test_chi2_no_detecta_cambio_si_son_iguales():
    serie = pd.Series(["A"] * 500 + ["B"] * 500)
    assert mm.calcular_chi2(serie, serie)[1] > 0.05


# ---------------- semaforo ----------------
@pytest.mark.parametrize("psi, esperado", [
    (0.0, "VERDE"), (0.10, "VERDE"), (0.11, "AMARILLO"), (0.25, "AMARILLO"), (0.26, "ROJO"), (1.0, "ROJO"),
])
def test_nivel_alerta(psi, esperado):
    assert mm.nivel_alerta(psi) == esperado


# ---------------- calcular_drift ----------------
def test_drift_sin_cambios_queda_en_verde():
    tabla = mm.calcular_drift(crear_datos(semilla=1), crear_datos(semilla=2))
    assert (tabla["nivel"] == "VERDE").all()
    assert len(tabla) == len(NUMERICAS) + len(CATEGORICAS) + 1     # +1 por prob_pago


def test_drift_con_cambio_fuerte_genera_alerta_roja():
    tabla = mm.calcular_drift(crear_datos(semilla=1), crear_datos(desplazamiento=30, semilla=2))
    assert (tabla["nivel"] == "ROJO").any()
    fila = tabla[tabla["variable"] == "capital_prestado"].iloc[0]
    assert fila["tipo"] == "numerica" and fila["ks_pvalor"] < 0.05


def test_drift_tabla_tiene_metricas_segun_tipo_de_variable():
    tabla = mm.calcular_drift(crear_datos(semilla=1), crear_datos(semilla=2))
    numerica = tabla[tabla["variable"] == "capital_prestado"].iloc[0]
    categorica = tabla[tabla["variable"] == "tipo_laboral"].iloc[0]
    assert np.isnan(numerica["chi2"]) and not np.isnan(numerica["ks"])
    assert np.isnan(categorica["ks"]) and not np.isnan(categorica["chi2"])


def test_drift_tabla_ordenada_por_psi_descendente():
    tabla = mm.calcular_drift(crear_datos(semilla=1), crear_datos(desplazamiento=30, semilla=2))
    assert tabla["psi"].is_monotonic_decreasing


def test_psi_de_una_variable_numerica_y_una_categorica():
    ref, act = crear_datos(semilla=1), crear_datos(desplazamiento=30, semilla=2)
    assert mm.calcular_psi_variable(ref, act, "capital_prestado") > mm.PSI_ROJO
    assert mm.calcular_psi_variable(ref, act, "tipo_laboral") < mm.PSI_AMARILLO


# ---------------- temporal ----------------
def test_drift_temporal_omite_ventanas_con_pocos_registros():
    ref = crear_datos(semilla=1)
    act = crear_datos(n=130, semilla=2, fecha_inicio="2026-01-01")      # ~4 meses de 30 dias
    temporal = mm.calcular_drift_temporal(ref, act, "M", min_filas=20)
    assert not temporal.empty
    assert set(temporal.columns) == {"periodo", "variable", "psi", "registros"}
    vacio = mm.calcular_drift_temporal(ref, act, "M", min_filas=1000)
    assert vacio.empty


def test_detectar_cambios_bruscos():
    temporal = pd.DataFrame({
        "periodo": ["2026-01", "2026-02", "2026-01", "2026-02"],
        "variable": ["a", "a", "b", "b"],
        "psi": [0.02, 0.50, 0.02, 0.03],
        "registros": [100] * 4,
    })
    mensajes = mm.detectar_cambios_bruscos(temporal)
    assert len(mensajes) == 1 and mensajes[0].startswith("a:")
    assert mm.detectar_cambios_bruscos(pd.DataFrame()) == []


# ---------------- recomendaciones ----------------
def test_recomendaciones_segun_semaforo():
    tabla = pd.DataFrame({"variable": ["x", "y", mm.VARIABLE_PREDICCION], "nivel": ["ROJO", "AMARILLO", "ROJO"]})
    tipos = [t for t, _ in mm.generar_recomendaciones(tabla)]
    assert tipos.count("error") == 2 and "warning" in tipos

    solo_verde = pd.DataFrame({"variable": ["x"], "nivel": ["VERDE"]})
    assert mm.generar_recomendaciones(solo_verde)[0][0] == "success"


# ---------------- carga y division de datos reales ----------------
def test_cargar_datos_monitoreo_agrega_fecha_y_prediccion():
    df = mm.cargar_datos_monitoreo()
    assert {"fecha_prestamo", mm.VARIABLE_PREDICCION}.issubset(df.columns)
    assert df[mm.VARIABLE_PREDICCION].between(0, 1).all()


def test_dividir_referencia_actual_no_pierde_ni_repite_registros():
    df = crear_datos()
    ref, act = mm.dividir_referencia_actual(df, "2025-03-31")
    assert len(ref) + len(act) == len(df)
    assert ref["fecha_prestamo"].max() <= pd.Timestamp("2025-03-31")
    assert act["fecha_prestamo"].min() > pd.Timestamp("2025-03-31")


def test_colorear_fila_pinta_la_columna_nivel():
    fila = pd.Series({"variable": "x", "psi": 0.3, "nivel": "ROJO"})
    estilos = mm.colorear_fila(fila)
    assert mm.COLORES["ROJO"] in estilos[2] and estilos[0] == ""


def test_reporte_consola_imprime_resultados(capsys):
    mm.reporte_consola()
    salida = capsys.readouterr().out
    assert "Referencia:" in salida and "psi" in salida


# ---------------- tablero de Streamlit ----------------
def test_tablero_streamlit_corre_sin_errores():
    at = AppTest.from_file(mm.__file__, default_timeout=180)
    at.run()
    assert not at.exception
    assert len(at.metric) == 5
    # cambiar a variable categorica y a ventana trimestral tampoco debe fallar
    at.selectbox[0].select("tipo_laboral").run()
    at.sidebar.radio[0].set_value("Trimestral").run()
    assert not at.exception


def test_tablero_avisa_si_quedan_pocos_registros():
    at = AppTest.from_file(mm.__file__, default_timeout=180)
    at.run()
    at.sidebar.date_input[0].set_value(pd.Timestamp("2024-11-05").date()).run()
    assert not at.exception
    assert len(at.error) >= 1
