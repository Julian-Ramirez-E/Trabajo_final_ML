"""
test_ft_engineering.py
----------------------
Pruebas de la limpieza de datos y del preprocesamiento (ft_engineering.py).
"""
import numpy as np
import pandas as pd

from ft_engineering import (
    limpiar_datos, construir_preprocesador, preparar_datos,
    NUMERICAS, CATEGORICAS, OBJETIVO, COLUMNAS_ELIMINAR,
)


def crear_cliente(**cambios):
    """Crea un DataFrame de 1 cliente valido; se pueden cambiar valores con **cambios."""
    cliente = {
        "capital_prestado": 2000000, "plazo_meses": 12, "edad_cliente": 35,
        "total_otros_prestamos": 100000, "puntaje_datacredito": 800, "huella_consulta": 3,
        "saldo_mora": 0, "saldo_total": 500000, "saldo_principal": 400000,
        "creditos_sectorFinanciero": 1, "creditos_sectorCooperativo": 0, "creditos_sectorReal": 0,
        "promedio_ingresos_datacredito": 1200000,
        "tipo_laboral": "Empleado", "tendencia_ingresos": "Creciente",
    }
    cliente.update(cambios)
    return pd.DataFrame([cliente])


# ---------------- limpiar_datos ----------------
def test_limpiar_devuelve_solo_columnas_del_modelo():
    df = crear_cliente()
    df["puntaje"] = 99          # columna con fuga de datos: debe desaparecer
    df["salario_cliente"] = 1   # columna irrelevante: debe desaparecer
    resultado = limpiar_datos(df)
    assert list(resultado.columns) == NUMERICAS + CATEGORICAS
    for col in COLUMNAS_ELIMINAR:
        assert col not in resultado.columns


def test_limpiar_conserva_objetivo_si_existe():
    df = crear_cliente()
    df[OBJETIVO] = 1
    assert OBJETIVO in limpiar_datos(df).columns


def test_edad_invalida_se_vuelve_nula():
    assert np.isnan(limpiar_datos(crear_cliente(edad_cliente=122))["edad_cliente"].iloc[0])
    assert np.isnan(limpiar_datos(crear_cliente(edad_cliente=15))["edad_cliente"].iloc[0])
    assert limpiar_datos(crear_cliente(edad_cliente=40))["edad_cliente"].iloc[0] == 40


def test_puntaje_datacredito_no_positivo_se_vuelve_nulo():
    assert np.isnan(limpiar_datos(crear_cliente(puntaje_datacredito=-5))["puntaje_datacredito"].iloc[0])
    assert np.isnan(limpiar_datos(crear_cliente(puntaje_datacredito=0))["puntaje_datacredito"].iloc[0])


def test_tendencia_invalida_pasa_a_sin_dato():
    for valor in ["8315", "xx", None, np.nan, ""]:
        resultado = limpiar_datos(crear_cliente(tendencia_ingresos=valor))
        assert resultado["tendencia_ingresos"].iloc[0] == "Sin_dato"


def test_tendencia_valida_se_conserva():
    for valor in ["Creciente", "Decreciente", "Estable"]:
        assert limpiar_datos(crear_cliente(tendencia_ingresos=valor))["tendencia_ingresos"].iloc[0] == valor


def test_columnas_faltantes_se_crean_vacias():
    df = pd.DataFrame([{"capital_prestado": 1000000, "plazo_meses": 6, "edad_cliente": 30,
                        "tipo_laboral": "Empleado"}])
    resultado = limpiar_datos(df)
    assert list(resultado.columns) == NUMERICAS + CATEGORICAS
    assert resultado["tendencia_ingresos"].iloc[0] == "Sin_dato"
    assert np.isnan(resultado["saldo_total"].iloc[0])


def test_texto_en_columna_numerica_se_vuelve_nulo():
    assert np.isnan(limpiar_datos(crear_cliente(capital_prestado="abc"))["capital_prestado"].iloc[0])


def test_limpiar_no_modifica_el_original():
    df = crear_cliente(edad_cliente=122)
    limpiar_datos(df)
    assert df["edad_cliente"].iloc[0] == 122


# ---------------- preprocesador ----------------
def test_preprocesador_no_deja_nulos_y_codifica_categorias():
    df = pd.concat([crear_cliente(), crear_cliente(edad_cliente=122, tipo_laboral="Independiente",
                                                   tendencia_ingresos="Estable")], ignore_index=True)
    datos = limpiar_datos(df)
    resultado = construir_preprocesador().fit_transform(datos)
    assert not np.isnan(resultado).any()
    # 13 numericas + 2 categorias de tipo_laboral + 2 de tendencia_ingresos = 17 columnas
    assert resultado.shape == (2, 17)


def test_preprocesador_ignora_categorias_desconocidas():
    prep = construir_preprocesador()
    prep.fit(limpiar_datos(crear_cliente()))
    nuevo = limpiar_datos(crear_cliente(tipo_laboral="Pensionado"))
    assert prep.transform(nuevo).shape[0] == 1   # no lanza error


# ---------------- preparar_datos ----------------
def test_preparar_datos_divide_80_20_y_estratifica():
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.20) < 0.01
    assert abs(y_train.mean() - y_test.mean()) < 0.005   # misma proporcion de buenos pagadores
    assert OBJETIVO not in X_train.columns
    assert preprocesador is not None


def test_ft_engineering_como_script_imprime_resumen(capsys):
    """Ejecuta ft_engineering.py como si se corriera 'python ft_engineering.py'."""
    import runpy
    import ft_engineering
    runpy.run_path(ft_engineering.__file__, run_name="__main__")
    salida = capsys.readouterr().out
    assert "Entrenamiento:" in salida
    assert "Nulos restantes en train: 0" in salida
