"""
model_monitoring.py
-------------------
Tercera componente del pipeline MLOps: MONITOREO DEL MODELO (DATA DRIFT).

Que es el data drift?
  Es cuando los datos nuevos "se parecen cada vez menos" a los datos con los que
  se entreno el modelo. Si la poblacion cambia, el modelo puede empezar a fallar.

Que hace este archivo:
  1. Carga los datos junto con las predicciones del modelo (prob_pago).
  2. Divide los datos en REFERENCIA (pasado) y ACTUAL (lo mas reciente).
  3. Calcula metricas de drift: KS, PSI, Jensen-Shannon y Chi-cuadrado.
  4. Genera alertas y recomendaciones automaticas.
  5. Muestra todo en un tablero de Streamlit.

Como usarlo:
  - Tablero visual :  streamlit run mlops_pipeline/model_monitoring.py
  - Reporte en consola:  python mlops_pipeline/model_monitoring.py
"""

# ---------------------------------------------------------------
# 1. IMPORTAR LIBRERIAS
# ---------------------------------------------------------------
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, chi2_contingency

# Funciones y listas de columnas del paso de ingenieria de caracteristicas
from ft_engineering import limpiar_datos, NUMERICAS, CATEGORICAS, RUTA_CSV

# ---------------------------------------------------------------
# 2. CONFIGURACION
# ---------------------------------------------------------------
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA_ACTUAL, "modelo_final.joblib")

# Umbrales del PSI (estandar en riesgo de credito)
PSI_AMARILLO = 0.10   # entre 0.10 y 0.25 -> cambio moderado (precaucion)
PSI_ROJO = 0.25       # mayor a 0.25      -> cambio significativo (alerta)

# Fecha de corte por defecto: hasta esta fecha son datos de REFERENCIA
FECHA_CORTE_DEFECTO = date(2025, 5, 31)

# Colores del semaforo
COLORES = {"VERDE": "#2ecc71", "AMARILLO": "#f1c40f", "ROJO": "#e74c3c"}

# Variable nueva: la probabilidad que predice el modelo (tambien puede tener drift)
VARIABLE_PREDICCION = "prob_pago"


# ---------------------------------------------------------------
# 3. FUNCIONES DE METRICAS DE DRIFT
# ---------------------------------------------------------------
def proporciones_numericas(ref, act, bins=10):
    """Divide una variable numerica en 'bins' rangos (segun la referencia) y calcula
    la proporcion de datos en cada rango, para referencia y actual."""
    # Los limites de los rangos salen de los percentiles de la referencia
    limites = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(limites) < 2:
        return np.array([1.0]), np.array([1.0])  # variable constante
    limites[0], limites[-1] = -np.inf, np.inf     # para que quepan valores nuevos extremos

    conteo_ref, _ = np.histogram(ref, bins=limites)
    conteo_act, _ = np.histogram(act, bins=limites)
    return conteo_ref / conteo_ref.sum(), conteo_act / conteo_act.sum()


def proporciones_categoricas(ref, act):
    """Calcula la proporcion de cada categoria en referencia y actual."""
    categorias = sorted(set(ref.unique()) | set(act.unique()))
    p_ref = ref.value_counts(normalize=True).reindex(categorias, fill_value=0).values
    p_act = act.value_counts(normalize=True).reindex(categorias, fill_value=0).values
    return p_ref, p_act


def calcular_psi(p_ref, p_act):
    """Population Stability Index: suma de (act - ref) * ln(act / ref)."""
    # Evitamos ceros (el logaritmo de 0 no existe)
    p_ref = np.clip(p_ref, 1e-4, None)
    p_act = np.clip(p_act, 1e-4, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def calcular_js(p_ref, p_act):
    """Distancia de Jensen-Shannon (0 = iguales, 1 = totalmente distintas)."""
    return float(jensenshannon(p_ref, p_act, base=2))


def calcular_chi2(ref, act):
    """Prueba Chi-cuadrado para variables categoricas. Devuelve (estadistico, p-valor)."""
    categorias = sorted(set(ref.unique()) | set(act.unique()))
    tabla = pd.DataFrame({
        "ref": ref.value_counts().reindex(categorias, fill_value=0),
        "act": act.value_counts().reindex(categorias, fill_value=0),
    })
    chi2, p_valor, _, _ = chi2_contingency(tabla.values)
    return float(chi2), float(p_valor)


def nivel_alerta(psi):
    """Convierte el PSI en un color de semaforo."""
    if psi > PSI_ROJO:
        return "ROJO"
    if psi > PSI_AMARILLO:
        return "AMARILLO"
    return "VERDE"


def calcular_drift(df_ref, df_act):
    """Calcula todas las metricas de drift, variable por variable. Devuelve una tabla."""
    filas = []

    # --- Variables numericas (incluida la prediccion): KS + PSI + Jensen-Shannon ---
    for col in NUMERICAS + [VARIABLE_PREDICCION]:
        ref = df_ref[col].dropna()
        act = df_act[col].dropna()
        p_ref, p_act = proporciones_numericas(ref, act)
        ks_stat, ks_p = ks_2samp(ref, act)
        psi = calcular_psi(p_ref, p_act)
        filas.append({
            "variable": col, "tipo": "numerica",
            "psi": psi, "ks": ks_stat, "ks_pvalor": ks_p,
            "js": calcular_js(p_ref, p_act),
            "chi2": np.nan, "chi2_pvalor": np.nan,
            "nulos_ref_%": df_ref[col].isna().mean() * 100,
            "nulos_act_%": df_act[col].isna().mean() * 100,
            "nivel": nivel_alerta(psi),
        })

    # --- Variables categoricas: PSI + Jensen-Shannon + Chi-cuadrado ---
    for col in CATEGORICAS:
        ref = df_ref[col].dropna()
        act = df_act[col].dropna()
        p_ref, p_act = proporciones_categoricas(ref, act)
        chi2, chi2_p = calcular_chi2(ref, act)
        psi = calcular_psi(p_ref, p_act)
        filas.append({
            "variable": col, "tipo": "categorica",
            "psi": psi, "ks": np.nan, "ks_pvalor": np.nan,
            "js": calcular_js(p_ref, p_act),
            "chi2": chi2, "chi2_pvalor": chi2_p,
            "nulos_ref_%": df_ref[col].isna().mean() * 100,
            "nulos_act_%": df_act[col].isna().mean() * 100,
            "nivel": nivel_alerta(psi),
        })

    tabla = pd.DataFrame(filas)
    return tabla.sort_values("psi", ascending=False).reset_index(drop=True)


def calcular_psi_variable(df_ref, df_act, col):
    """PSI de una sola variable (se usa en el analisis temporal)."""
    if col in CATEGORICAS:
        p_ref, p_act = proporciones_categoricas(df_ref[col].dropna(), df_act[col].dropna())
    else:
        p_ref, p_act = proporciones_numericas(df_ref[col].dropna(), df_act[col].dropna())
    return calcular_psi(p_ref, p_act)


def calcular_drift_temporal(df_ref, df_act, frecuencia="M", min_filas=100):
    """Calcula el PSI de cada variable para cada periodo (mes o trimestre) vs la referencia."""
    periodos = df_act["fecha_prestamo"].dt.to_period(frecuencia)
    filas = []
    for periodo, grupo in df_act.groupby(periodos):
        if len(grupo) < min_filas:
            continue  # con muy pocos datos el PSI no es confiable
        for col in NUMERICAS + CATEGORICAS + [VARIABLE_PREDICCION]:
            filas.append({
                "periodo": str(periodo), "variable": col,
                "psi": calcular_psi_variable(df_ref, grupo, col), "registros": len(grupo),
            })
    return pd.DataFrame(filas)


def detectar_cambios_bruscos(temporal, salto=0.10):
    """Busca variables cuyo PSI sube mas de 'salto' de un periodo al siguiente."""
    mensajes = []
    if temporal.empty:
        return mensajes
    pivote = temporal.pivot(index="periodo", columns="variable", values="psi").sort_index()
    cambios = pivote.diff()
    for col in cambios.columns:
        for periodo, valor in cambios[col].items():
            if valor > salto:
                mensajes.append(f"{col}: el PSI subio {valor:.2f} en {periodo} respecto al periodo anterior")
    return mensajes


def generar_recomendaciones(tabla):
    """Crea mensajes automaticos segun los semaforos. Devuelve lista de (tipo, mensaje)."""
    mensajes = []
    rojas = tabla[tabla["nivel"] == "ROJO"]["variable"].tolist()
    amarillas = tabla[tabla["nivel"] == "AMARILLO"]["variable"].tolist()

    if rojas:
        mensajes.append(("error", "ALERTA CRITICA: drift significativo (PSI > 0.25) en: " + ", ".join(rojas)
                         + ". Se recomienda revisar estas variables y evaluar el REENTRENAMIENTO del modelo."))
    if VARIABLE_PREDICCION in rojas:
        mensajes.append(("error", "La distribucion de las predicciones cambio mucho: el modelo esta "
                                  "calificando de forma distinta a la poblacion actual. Reentrenar es prioritario."))
    if amarillas:
        mensajes.append(("warning", "Precaucion: cambio moderado (PSI entre 0.10 y 0.25) en: "
                         + ", ".join(amarillas) + ". Se recomienda seguir monitoreando."))
    if not rojas and not amarillas:
        mensajes.append(("success", "Sin alertas: las distribuciones actuales son estables frente a la referencia."))
    return mensajes


# ---------------------------------------------------------------
# 4. CARGA DE DATOS CON PREDICCIONES
# ---------------------------------------------------------------
def cargar_datos_monitoreo():
    """Lee el CSV, lo limpia, y agrega la fecha y la prediccion del modelo."""
    df = pd.read_csv(RUTA_CSV)
    fechas = pd.to_datetime(df["fecha_prestamo"])

    df_limpio = limpiar_datos(df)          # mismas reglas que en el entrenamiento
    df_limpio["fecha_prestamo"] = fechas   # la fecha se necesita para el analisis temporal

    # Pronostico del modelo: probabilidad de pagar a tiempo
    modelo = joblib.load(RUTA_MODELO)
    df_limpio[VARIABLE_PREDICCION] = modelo.predict_proba(df_limpio[NUMERICAS + CATEGORICAS])[:, 1]
    return df_limpio


def dividir_referencia_actual(df, fecha_corte):
    """Separa los datos en referencia (hasta la fecha de corte) y actual (despues)."""
    corte = pd.Timestamp(fecha_corte) + pd.Timedelta(days=1)
    df_ref = df[df["fecha_prestamo"] < corte]
    df_act = df[df["fecha_prestamo"] >= corte]
    return df_ref, df_act


# ---------------------------------------------------------------
# 5. TABLERO DE STREAMLIT
# ---------------------------------------------------------------
def colorear_fila(fila):
    """Pinta la celda 'nivel' de la tabla segun el semaforo."""
    estilos = [""] * len(fila)
    color = COLORES[fila["nivel"]]
    estilos[list(fila.index).index("nivel")] = f"background-color: {color}; color: black; font-weight: bold"
    return estilos


def tablero():
    st.set_page_config(page_title="Monitoreo de Data Drift", layout="wide")
    st.title("Monitoreo de Data Drift - Prediccion de Pago a Tiempo")
    st.caption("Compara los datos historicos (referencia) con los mas recientes (actual) para detectar "
               "cambios en la poblacion que puedan afectar al modelo.")

    # ----- Datos (en cache para que no se recalcule en cada clic) -----
    @st.cache_data
    def datos_cacheados():
        return cargar_datos_monitoreo()
    df = datos_cacheados()

    # ----- Barra lateral con parametros -----
    st.sidebar.header("Parametros")
    fecha_min = df["fecha_prestamo"].min().date()
    fecha_max = df["fecha_prestamo"].max().date()
    fecha_corte = st.sidebar.date_input("Fecha de corte (referencia hasta)", value=FECHA_CORTE_DEFECTO,
                                        min_value=fecha_min, max_value=fecha_max)
    ventana = st.sidebar.radio("Ventana temporal", ["Mensual", "Trimestral"])
    min_filas = st.sidebar.slider("Minimo de registros por ventana", 30, 300, 100)
    st.sidebar.markdown(f"**Umbrales PSI**  \nVerde: menor a {PSI_AMARILLO}  \n"
                        f"Amarillo: {PSI_AMARILLO} a {PSI_ROJO}  \nRojo: mayor a {PSI_ROJO}")

    df_ref, df_act = dividir_referencia_actual(df, fecha_corte)
    if len(df_ref) < 100 or len(df_act) < 100:
        st.error("Con esa fecha de corte quedan muy pocos registros en referencia o actual. Cambia la fecha.")
        return

    tabla = calcular_drift(df_ref, df_act)

    # ----- Indicadores (semaforo) -----
    st.subheader("1. Resumen")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Registros referencia", f"{len(df_ref):,}")
    c2.metric("Registros actual", f"{len(df_act):,}")
    c3.metric("🔴 Variables en rojo", int((tabla["nivel"] == "ROJO").sum()))
    c4.metric("🟡 Variables en amarillo", int((tabla["nivel"] == "AMARILLO").sum()))
    c5.metric("🟢 Variables en verde", int((tabla["nivel"] == "VERDE").sum()))

    # ----- Recomendaciones y alertas -----
    st.subheader("2. Alertas y recomendaciones")
    for tipo, mensaje in generar_recomendaciones(tabla):
        if tipo == "error":
            st.error(mensaje)
        elif tipo == "warning":
            st.warning(mensaje)
        else:
            st.success(mensaje)

    # ----- Tabla de metricas -----
    st.subheader("3. Metricas de drift por variable")
    st.dataframe(tabla.style.apply(colorear_fila, axis=1).format(
        {"psi": "{:.3f}", "ks": "{:.3f}", "ks_pvalor": "{:.4f}", "js": "{:.3f}", "chi2": "{:.1f}",
         "chi2_pvalor": "{:.4f}", "nulos_ref_%": "{:.1f}", "nulos_act_%": "{:.1f}"}, na_rep="-"))
    st.caption("PSI: Population Stability Index (criterio del semaforo). KS: Kolmogorov-Smirnov. "
               "JS: Jensen-Shannon. Chi2: prueba Chi-cuadrado (categoricas). Un p-valor menor a 0.05 indica "
               "diferencia estadisticamente significativa, aunque con muchos datos casi todo sale significativo; "
               "por eso el semaforo se basa en el PSI.")

    # ----- Barras de riesgo -----
    st.subheader("4. Barras de riesgo (PSI)")
    fig_barras = px.bar(tabla.sort_values("psi"), x="psi", y="variable", orientation="h",
                        color="nivel", color_discrete_map=COLORES)
    fig_barras.add_vline(x=PSI_AMARILLO, line_dash="dash", line_color="orange")
    fig_barras.add_vline(x=PSI_ROJO, line_dash="dash", line_color="red")
    fig_barras.update_layout(height=520)
    st.plotly_chart(fig_barras)

    # ----- Distribucion historica vs actual -----
    st.subheader("5. Distribucion historica vs actual")
    variable = st.selectbox("Elige una variable", tabla["variable"].tolist())
    df_graf = pd.concat([df_ref.assign(grupo="Referencia"), df_act.assign(grupo="Actual")])
    if variable in CATEGORICAS:
        prop = (df_graf.groupby("grupo")[variable].value_counts(normalize=True)
                .rename("proporcion").reset_index())
        fig_dist = px.bar(prop, x=variable, y="proporcion", color="grupo", barmode="group")
    else:
        # Recortamos el 1 % extremo para que el grafico sea legible (hay valores atipicos)
        bajo, alto = df_graf[variable].quantile([0.01, 0.99])
        recortado = df_graf[(df_graf[variable] >= bajo) & (df_graf[variable] <= alto)]
        fig_dist = px.histogram(recortado, x=variable, color="grupo", barmode="overlay", opacity=0.6,
                                histnorm="probability density", nbins=40)
    st.plotly_chart(fig_dist)

    # ----- Analisis temporal -----
    st.subheader("6. Evolucion del drift en el tiempo")
    frecuencia = "M" if ventana == "Mensual" else "Q"
    temporal = calcular_drift_temporal(df_ref, df_act, frecuencia, min_filas)
    if temporal.empty:
        st.info("Ninguna ventana alcanza el minimo de registros. Baja el minimo en la barra lateral.")
    else:
        pivote = temporal.pivot(index="variable", columns="periodo", values="psi")
        fig_calor = px.imshow(pivote, aspect="auto", zmin=0, zmax=0.4, text_auto=".2f",
                              color_continuous_scale=[(0, "#2ecc71"), (0.25, "#f1c40f"),
                                                      (0.625, "#e74c3c"), (1, "#7b241c")])
        fig_calor.update_layout(title="PSI por variable y periodo", height=560)
        st.plotly_chart(fig_calor)

        var_tiempo = st.selectbox("Variable para ver su tendencia", pivote.index.tolist(), key="tiempo")
        fig_linea = px.line(temporal[temporal["variable"] == var_tiempo], x="periodo", y="psi", markers=True)
        fig_linea.add_hline(y=PSI_AMARILLO, line_dash="dash", line_color="orange")
        fig_linea.add_hline(y=PSI_ROJO, line_dash="dash", line_color="red")
        st.plotly_chart(fig_linea)

        cambios = detectar_cambios_bruscos(temporal)
        if cambios:
            st.warning("Cambios bruscos detectados:\n\n" + "\n".join("- " + c for c in cambios))
        else:
            st.info("No se detectaron cambios bruscos entre periodos consecutivos.")
        st.caption("Los periodos con pocos registros se omiten porque el PSI se vuelve inestable.")


# ---------------------------------------------------------------
# 6. REPORTE EN CONSOLA (el "job" de monitoreo)
# ---------------------------------------------------------------
def reporte_consola():
    df = cargar_datos_monitoreo()
    df_ref, df_act = dividir_referencia_actual(df, FECHA_CORTE_DEFECTO)
    tabla = calcular_drift(df_ref, df_act)

    print(f"Referencia: {len(df_ref)} registros | Actual: {len(df_act)} registros")
    print(tabla[["variable", "psi", "ks", "js", "chi2_pvalor", "nivel"]].round(3).to_string(index=False))
    print()
    for _, mensaje in generar_recomendaciones(tabla):
        print(mensaje)


# ---------------------------------------------------------------
# 7. PUNTO DE ENTRADA
# ---------------------------------------------------------------
def corre_en_streamlit():
    """True si el archivo lo esta ejecutando 'streamlit run'."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    return get_script_run_ctx(suppress_warning=True) is not None


if corre_en_streamlit():
    tablero()
elif __name__ == "__main__":
    reporte_consola()
