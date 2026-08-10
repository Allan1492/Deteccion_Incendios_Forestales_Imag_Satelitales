"""
Dashboard Interactivo — Detección de Incendios Forestales (Etapa 6)
Proyecto: Reducción de Falsas Alarmas mediante Análisis Multivariable sobre HPC
Curso: TTCT0017 — Computación Paralela y Distribuida, LEAD University

Diseño: consola de instrumentos / panel de misión (rampa térmica).

Cómo correrlo:
    pip install -r requirements.txt
    python app.py
Luego abrir http://127.0.0.1:8050

Datos (solo lectura, no se modifican):
    data/benchmark_consolidado.csv   -> rendimiento CPU vs GPU (Etapa 5)
    data/comparacion_modelos.csv     -> calidad de los modelos tabulares (Etapa 4)
    data/metricas_cnn.json           -> métricas de la CNN de imágenes (Etapa 4)
"""

from pathlib import Path
import json
import os

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx

# ===========================================================================
# 1. Sistema de diseño — "Thermal Console"
# ===========================================================================
BG        = "#07070a"
INK       = "#f3f1ec"
MUTED     = "#8b8892"
FAINT     = "#5a5862"
HAIR      = "#1a1a22"
CARD      = "#0e0e13"
CARD_2    = "#121218"
EMBER     = "#ff5e1a"     # ascua
EMBER_HI  = "#ff8a3d"
EMBER_LO  = "#c23a00"
GOLD      = "#ffcf6b"
SIGNAL    = "#35e08b"     # GPU / señal
COOL      = "#4fa3e0"
CPU_COL   = "#6c6a75"
FONT      = "'Inter', -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
MONO      = "'IBM Plex Mono', 'SFMono-Regular', 'Courier New', monospace"

# rampa térmica (para acentos)
THERMAL   = f"linear-gradient(90deg,{EMBER_LO},{EMBER},{EMBER_HI},{GOLD})"

COLOR_MODELO = {"Random Forest": COOL, "XGBoost": EMBER, "Red Neuronal": SIGNAL}
ORDEN_PCT = ["30%", "60%", "100%"]

BASE_DIR   = Path(__file__).parent
RUTA_DATOS = BASE_DIR / "data" / "benchmark_consolidado.csv"
RUTA_CAL   = BASE_DIR / "data" / "comparacion_modelos.csv"
RUTA_CNN   = BASE_DIR / "data" / "metricas_cnn.json"

# ===========================================================================
# 2. Carga de datos (solo lectura, sin alterar nada)
# ===========================================================================
def es_gpu(backend):
    return any(x in str(backend) for x in ["GPU", "cuda", "cuML"])


def cargar_datos():
    if not RUTA_DATOS.exists():
        return pd.DataFrame()
    df = pd.read_csv(RUTA_DATOS)
    df["dispositivo"] = df["backend"].apply(lambda b: "GPU" if es_gpu(b) else "CPU")
    df["modelo"] = df["modelo"].str.replace(r"\s*\(por epoca\)", "", regex=True)
    return df


def cargar_calidad():
    return pd.read_csv(RUTA_CAL) if RUTA_CAL.exists() else pd.DataFrame()


def cargar_cnn():
    if not RUTA_CNN.exists():
        return {}
    with open(RUTA_CNN, encoding="utf-8") as fh:
        return json.load(fh)


def calcular_speedup(df):
    filas = []
    for (modelo, pct), grupo in df.groupby(["modelo", "dataset_pct"]):
        cpu = grupo[grupo["dispositivo"] == "CPU"]
        gpu = grupo[grupo["dispositivo"] == "GPU"]
        if cpu.empty or gpu.empty:
            continue
        t_cpu, t_gpu = cpu["tiempo_s"].iloc[0], gpu["tiempo_s"].iloc[0]
        filas.append({"modelo": modelo, "dataset_pct": pct,
                      "speedup": t_cpu / t_gpu if t_gpu else None})
    return pd.DataFrame(filas)


def calcular_escalabilidad(df):
    tiempo_base = df[df["dataset_pct"] == "30%"].set_index(["modelo", "dispositivo"])["tiempo_s"]
    f0 = df[df["dataset_pct"] == "30%"]["filas_train"].iloc[0]
    resultado = []
    for pct in ["60%", "100%"]:
        sub = df[df["dataset_pct"] == pct]
        for _, fila in sub.iterrows():
            clave = (fila["modelo"], fila["dispositivo"])
            if clave not in tiempo_base.index:
                continue
            t0 = tiempo_base.loc[clave]
            indice = (fila["tiempo_s"] / t0) / (fila["filas_train"] / f0)
            resultado.append({"modelo": fila["modelo"], "dispositivo": fila["dispositivo"],
                              "dataset_pct": pct, "indice_escalabilidad": indice})
    return pd.DataFrame(resultado)


datos         = cargar_datos()
speedups      = calcular_speedup(datos) if not datos.empty else pd.DataFrame()
escalabilidad = calcular_escalabilidad(datos) if not datos.empty else pd.DataFrame()
calidad       = cargar_calidad()
cnn           = cargar_cnn()

# ===========================================================================
# 3. Figuras (rampa térmica, retícula tenue)
# ===========================================================================
LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK, family="Inter, sans-serif", size=13),
    margin=dict(t=26, l=64, r=24, b=54),
    legend=dict(orientation="h", y=-0.24, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    xaxis=dict(gridcolor=HAIR, zerolinecolor=HAIR, linecolor=HAIR),
    yaxis=dict(gridcolor=HAIR, zerolinecolor=HAIR, linecolor=HAIR),
    hoverlabel=dict(bgcolor=CARD_2, font_size=13, font_family="IBM Plex Mono, monospace",
                    bordercolor=HAIR),
)


def figura_vacia(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_layout(**LAYOUT)
    return fig


def figura_speedup():
    if speedups.empty:
        return figura_vacia("Sin datos de speedup")
    fig = go.Figure()
    for modelo in speedups["modelo"].unique():
        sub = (speedups[speedups["modelo"] == modelo]
               .set_index("dataset_pct").reindex(ORDEN_PCT).reset_index())
        fig.add_trace(go.Scatter(
            x=sub["dataset_pct"], y=sub["speedup"], mode="lines+markers+text",
            text=[f"{v:.1f}×" if pd.notna(v) else "" for v in sub["speedup"]],
            textposition="top center", textfont=dict(size=12, color=INK, family="IBM Plex Mono"),
            name=modelo, line=dict(color=COLOR_MODELO.get(modelo), width=3.5),
            marker=dict(size=11)))
    fig.add_hline(y=1, line_dash="dot", line_color=FAINT,
                  annotation_text="CPU = GPU", annotation_font_color=FAINT)
    fig.update_layout(yaxis_title="Speedup (GPU / CPU)", xaxis_title="Tamaño del dataset", **LAYOUT)
    return fig


def figura_throughput():
    if datos.empty:
        return figura_vacia("Sin datos")
    fig = go.Figure()
    for modelo in datos["modelo"].unique():
        for disp, estilo in [("CPU", "dot"), ("GPU", "solid")]:
            sub = (datos[(datos["modelo"] == modelo) & (datos["dispositivo"] == disp)]
                   .set_index("dataset_pct").reindex(ORDEN_PCT).reset_index())
            fig.add_trace(go.Scatter(
                x=sub["dataset_pct"], y=sub["throughput_reg_s"], mode="lines+markers",
                name=f"{modelo} · {disp}",
                line=dict(color=COLOR_MODELO.get(modelo), dash=estilo, width=2.5),
                marker=dict(size=8)))
    fig.update_layout(yaxis_title="Throughput (registros/s)", yaxis_type="log",
                      xaxis_title="Tamaño del dataset", **LAYOUT)
    return fig


def figura_tiempos(pct):
    if datos.empty:
        return figura_vacia("Sin datos")
    sub = datos[datos["dataset_pct"] == pct]
    fig = go.Figure()
    for disp, color in [("CPU", CPU_COL), ("GPU", EMBER)]:
        parte = sub[sub["dispositivo"] == disp]
        fig.add_trace(go.Bar(x=parte["modelo"], y=parte["tiempo_s"], name=disp,
                             marker_color=color, marker_line_width=0,
                             text=parte["tiempo_s"].round(1), textposition="outside",
                             textfont=dict(color=INK, family="IBM Plex Mono")))
    fig.update_layout(barmode="group", yaxis_title="Tiempo (s) — escala log",
                      yaxis_type="log", **LAYOUT)
    return fig


def figura_recursos(pct, metrica):
    if datos.empty:
        return figura_vacia("Sin datos")
    sub = datos[datos["dataset_pct"] == pct].copy()
    sub["etq"] = sub["modelo"] + " · " + sub["dispositivo"]
    colores = [EMBER if d == "GPU" else CPU_COL for d in sub["dispositivo"]]
    fig = go.Figure(go.Bar(x=sub["etq"], y=sub[metrica], marker_color=colores, marker_line_width=0,
                           text=sub[metrica].round(1), textposition="outside",
                           textfont=dict(color=INK, family="IBM Plex Mono")))
    et = {"ram_pico_mb": "RAM pico (MB)", "gpu_mem_pico_mb": "Memoria GPU pico (MB)",
          "gpu_util_promedio_pct": "Uso promedio de GPU (%) — eficiencia"}
    fig.update_layout(yaxis_title=et.get(metrica, metrica), xaxis_tickangle=-25, **LAYOUT)
    return fig


def figura_escalabilidad():
    if escalabilidad.empty:
        return figura_vacia("Sin datos")
    fig = go.Figure()
    for modelo in escalabilidad["modelo"].unique():
        for disp, estilo in [("CPU", "dot"), ("GPU", "solid")]:
            sub = (escalabilidad[(escalabilidad["modelo"] == modelo) &
                                 (escalabilidad["dispositivo"] == disp)]
                   .set_index("dataset_pct").reindex(["60%", "100%"]).reset_index())
            fig.add_trace(go.Scatter(
                x=sub["dataset_pct"], y=sub["indice_escalabilidad"], mode="lines+markers",
                name=f"{modelo} · {disp}",
                line=dict(color=COLOR_MODELO.get(modelo), dash=estilo, width=2.5),
                marker=dict(size=8)))
    fig.add_hline(y=1, line_dash="dot", line_color=FAINT,
                  annotation_text="Crecimiento lineal (ideal)", annotation_font_color=FAINT)
    fig.update_layout(yaxis_title="Índice de escalabilidad (1.0 = lineal)",
                      xaxis_title="Tamaño del dataset", **LAYOUT)
    return fig


def figura_calidad(metrica):
    if calidad.empty:
        return figura_vacia("Sin datos de calidad")
    df = calidad.copy()
    x, y = list(df["modelo"]), list(df[metrica])
    colores = [COLOR_MODELO.get(m, "#888") for m in x]
    if cnn and metrica in cnn:
        x, y, colores = x + ["CNN (imágenes)"], y + [cnn[metrica]], colores + [GOLD]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colores, marker_line_width=0,
                           text=[f"{v:.3f}" for v in y], textposition="outside",
                           textfont=dict(color=INK, family="IBM Plex Mono")))
    et = {"f1": "F1-score", "roc_auc": "ROC-AUC", "precision": "Precisión",
          "recall": "Recall (exhaustividad)", "accuracy": "Exactitud"}
    fig.update_layout(yaxis_title=et.get(metrica, metrica), **LAYOUT)
    fig.update_yaxes(range=[0, 1.06])
    return fig


def figura_calidad_radar():
    if calidad.empty:
        return figura_vacia("Sin datos")
    metricas = ["precision", "recall", "f1", "roc_auc"]
    etiquetas = ["Precisión", "Recall", "F1", "ROC-AUC"]
    fig = go.Figure()
    for _, fila in calidad.iterrows():
        vals = [fila[m] for m in metricas]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=etiquetas + [etiquetas[0]], fill="toself",
            name=fila["modelo"], line=dict(color=COLOR_MODELO.get(fila["modelo"], "#888"), width=2),
            opacity=0.85))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0.6, 1.0], gridcolor=HAIR, tickfont=dict(color=FAINT, size=10)),
                   angularaxis=dict(gridcolor=HAIR, tickfont=dict(color=MUTED))),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="Inter, sans-serif"),
        legend=dict(orientation="h", y=-0.14, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, l=44, r=44, b=44))
    return fig


# ===========================================================================
# 4. Componentes visuales
# ===========================================================================
def micro(texto, color=MUTED):
    return html.Div(texto, style={"fontSize": "10.5px", "color": color, "textTransform": "uppercase",
                                  "letterSpacing": "0.22em", "fontWeight": "700",
                                  "fontFamily": MONO})


def metric(label, valor, sub=None, color=INK, unidad=None):
    val = [html.Span(valor, style={"fontSize": "46px", "fontWeight": "800", "color": color,
                                   "fontFamily": MONO, "lineHeight": "0.95", "letterSpacing": "-0.02em"})]
    if unidad:
        val.append(html.Span(unidad, style={"fontSize": "18px", "color": MUTED,
                             "fontFamily": MONO, "marginLeft": "6px"}))
    hijos = [
        html.Div(style={"height": "2px", "width": "40px", "background": THERMAL, "marginBottom": "14px"}),
        micro(label),
        html.Div(val, style={"marginTop": "10px"}),
    ]
    if sub:
        hijos.append(html.Div(sub, style={"fontSize": "12px", "color": MUTED, "marginTop": "8px",
                                          "fontFamily": MONO}))
    return html.Div(hijos, style={
        "background": CARD, "borderRadius": "4px", "padding": "22px 22px 24px", "flex": "1",
        "minWidth": "170px", "border": f"1px solid {HAIR}", "position": "relative", "overflow": "hidden"})


def panel(children, titulo=None, sub=None):
    cab = []
    if titulo:
        cab.append(html.Div(titulo, style={"color": INK, "fontSize": "15px", "fontWeight": "700",
                                          "marginBottom": "3px", "letterSpacing": "0.01em"}))
    if sub:
        cab.append(html.Div(sub, style={"color": MUTED, "fontSize": "12.5px", "marginBottom": "16px"}))
    cuerpo = children if isinstance(children, list) else [children]
    return html.Div(cab + cuerpo, style={
        "background": CARD, "borderRadius": "4px", "padding": "22px", "border": f"1px solid {HAIR}"})


def nota(indice, titulo, texto, color=EMBER):
    return html.Div([
        html.Div([
            html.Span(indice, style={"fontFamily": MONO, "fontSize": "11px", "color": color,
                                     "fontWeight": "700", "letterSpacing": "0.1em"}),
            html.Div(style={"flex": "1", "height": "1px", "background": HAIR, "marginLeft": "10px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        html.Div(titulo, style={"color": INK, "fontWeight": "700", "fontSize": "14.5px", "marginBottom": "7px"}),
        html.Div(texto, style={"color": MUTED, "fontSize": "12.5px", "lineHeight": "1.55"}),
    ], style={"background": CARD_2, "borderRadius": "4px", "padding": "18px 20px", "flex": "1",
              "minWidth": "220px", "borderTop": f"2px solid {color}"})


def seccion(indice, titulo, sub=None):
    hijos = [
        html.Div([
            html.Span(indice, style={"fontFamily": MONO, "fontSize": "58px", "fontWeight": "800",
                                     "color": "transparent", "WebkitTextStroke": f"1px {HAIR}",
                                     "lineHeight": "0.8", "marginRight": "18px"}),
            html.Div([
                micro("Sección", EMBER),
                html.Div(titulo, style={"color": INK, "fontSize": "27px", "fontWeight": "800",
                                        "marginTop": "4px", "letterSpacing": "-0.01em"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
    ]
    if sub:
        hijos.append(html.Div(sub, style={"color": MUTED, "fontSize": "13.5px", "marginTop": "10px",
                                          "maxWidth": "680px"}))
    return html.Div(hijos, style={"marginBottom": "22px"})


# ===========================================================================
# 5. Slides
# ===========================================================================
def _mejor(df, col):
    return None if df.empty else df.loc[df[col].idxmax()]


def slide_resumen():
    max_sp = _mejor(speedups, "speedup") if not speedups.empty else None
    mejor_f1 = _mejor(calidad, "f1") if not calidad.empty else None
    cnn_f1 = cnn.get("f1") if cnn else None
    return html.Div([
        html.Div([
            micro("Informe de resultados · Proyecto final", EMBER),
            html.Div([
                html.Span("Reducción de "),
                html.Span("Falsas Alarmas", className="thermal-text"),
                html.Br(),
                html.Span("en la Detección Satelital de Incendios"),
            ], style={"color": INK, "fontSize": "38px", "fontWeight": "800", "marginTop": "14px",
                      "lineHeight": "1.08", "letterSpacing": "-0.02em"}),
            html.Div("Análisis multivariable acelerado por GPU sobre el clúster HPC Kabré (CeNAT). "
                     "52.5 millones de detecciones VIIRS y 42 850 imágenes satelitales.",
                     style={"color": MUTED, "fontSize": "15px", "marginTop": "16px", "maxWidth": "640px"}),
        ], style={"marginBottom": "30px"}),
        html.Div([
            metric("Registros procesados", "52.5", sub="NASA FIRMS · VIIRS", unidad="M"),
            metric("Mayor speedup GPU", f"{max_sp['speedup']:.1f}" if max_sp is not None else "—",
                   sub=f"{max_sp['modelo']} · {max_sp['dataset_pct']}" if max_sp is not None else "",
                   color=EMBER, unidad="×"),
            metric("Mejor F1 tabular", f"{mejor_f1['f1']:.3f}" if mejor_f1 is not None else "—",
                   sub=mejor_f1["modelo"] if mejor_f1 is not None else "", color=SIGNAL),
            metric("F1 de la CNN", f"{cnn_f1:.3f}" if cnn_f1 else "—",
                   sub="Clasificación de imágenes", color=GOLD),
        ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "18px"}),
        panel([
            html.Div([html.Span(className="dot"),
                      html.Span("Ejecutado y verificado en Kabré (CeNAT)",
                                style={"color": SIGNAL, "fontWeight": "700", "letterSpacing": "0.02em"})],
                     style={"marginBottom": "10px"}),
            html.Div("Partición KURA — CPU 40 núcleos — preprocesamiento paralelo   //   "
                     "Partición NUKWA — GPU NVIDIA — modelado   //   Gestor SLURM",
                     style={"color": MUTED, "fontSize": "12.5px", "fontFamily": MONO}),
        ]),
    ])


def slide_problema():
    return html.Div([
        seccion("01", "El problema: falsas alarmas",
                "No toda detección satelital de calor es un incendio real. Reducir los falsos positivos es el objetivo del proyecto."),
        html.Div([
            metric("Detecciones válidas", "87.85", sub="confianza media / alta", color=SIGNAL, unidad="%"),
            metric("Probables falsas alarmas", "12.15", sub="confianza baja (proxy)", color=EMBER, unidad="%"),
            metric("Desbalance de clases", "7:1", sub="por eso no se usa accuracy"),
        ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "16px"}),
        panel(html.Div([
            nota("01 / 03", "El contraste térmico separa las clases",
                 "ΔT = brillo del foco − fondo. Alta confianza: 60.8 K de contraste. "
                 "Baja confianza: solo 28.1 K. Un foco que apenas destaca de su entorno es sospechoso.", EMBER),
            nota("02 / 03", "Enfoque multivariable",
                 "En lugar de un único umbral, se combinan brillo, ΔT, potencia radiativa (FRP), "
                 "geometría del píxel y contexto temporal para decidir mejor.", SIGNAL),
            nota("03 / 03", "Dos ramas independientes",
                 "Clasificación tabular de falsas alarmas (52.5 M registros) y clasificación de "
                 "imágenes con CNN, resueltas en paralelo sobre HPC.", GOLD),
        ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}), titulo="Cómo se aborda"),
    ])


def slide_calidad():
    return html.Div([
        seccion("02", "Calidad de los modelos",
                "Tres modelos tabulares (GPU) para falsas alarmas y una CNN para imágenes. Todos con ROC-AUC entre 0.98 y 0.99."),
        html.Div(dcc.RadioItems(id="sel-metrica-calidad",
                 options=[{"label": " F1", "value": "f1"}, {"label": " ROC-AUC", "value": "roc_auc"},
                          {"label": " Precisión", "value": "precision"}, {"label": " Recall", "value": "recall"}],
                 value="f1", inline=True, labelStyle=R_LABEL, inputStyle=R_INPUT),
                 style={"marginBottom": "14px"}),
        panel(dcc.Graph(id="g-calidad", config={"displayModeBar": False}), titulo="Comparación por métrica"),
        html.Div(style={"height": "14px"}),
        panel(dcc.Graph(figure=figura_calidad_radar(), config={"displayModeBar": False}),
              titulo="Perfil comparativo", sub="Cada modelo en las cuatro métricas clave. Más área es mejor equilibrio."),
    ])


def slide_speedup():
    return html.Div([
        seccion("03", "Aceleración por GPU", "Cuántas veces más rápido entrena cada modelo en GPU frente a CPU."),
        panel(dcc.Graph(figure=figura_speedup(), config={"displayModeBar": False})),
        html.Div(style={"height": "14px"}),
        html.Div([
            nota("A", "Hasta 6.8× más rápido",
                 "XGBoost y Random Forest alcanzan speedups de 6.6 a 6.8× a escala completa.", EMBER),
            nota("B", "No todo escala igual",
                 "La red neuronal gana menos por época: su cuello de botella es la carga de datos, no el cómputo.", SIGNAL),
        ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}),
    ])


def slide_throughput():
    return html.Div([
        seccion("04", "Throughput", "Registros procesados por segundo (escala logarítmica)."),
        panel(dcc.Graph(figure=figura_throughput(), config={"displayModeBar": False})),
    ])


def slide_tiempos():
    return html.Div([
        seccion("05", "Tiempo de entrenamiento", "Comparación CPU vs GPU por modelo y escala de datos."),
        dcc.RadioItems(id="sel-pct-tiempo", options=[{"label": f" {p}", "value": p} for p in ORDEN_PCT],
                       value="100%", inline=True, style={"marginBottom": "14px"},
                       labelStyle=R_LABEL, inputStyle=R_INPUT),
        panel(dcc.Graph(id="g-tiempos", config={"displayModeBar": False})),
    ])


def slide_recursos():
    return html.Div([
        seccion("06", "Uso de recursos y eficiencia",
                "Ser rápido no es lo mismo que usar bien el hardware. La utilización de GPU lo revela."),
        html.Div([
            dcc.RadioItems(id="sel-pct-recursos", options=[{"label": f" {p}", "value": p} for p in ORDEN_PCT],
                           value="100%", inline=True, labelStyle=R_LABEL, inputStyle=R_INPUT),
            dcc.Dropdown(id="sel-metrica-recursos", options=[
                {"label": "Uso de GPU (%) — eficiencia", "value": "gpu_util_promedio_pct"},
                {"label": "RAM pico (MB)", "value": "ram_pico_mb"},
                {"label": "Memoria GPU pico (MB)", "value": "gpu_mem_pico_mb"}],
                value="gpu_util_promedio_pct", clearable=False,
                style={"width": "340px", "marginTop": "10px", "color": "#111"}),
        ], style={"marginBottom": "16px"}),
        panel(dcc.Graph(id="g-recursos", config={"displayModeBar": False})),
        html.Div(style={"height": "14px"}),
        html.Div([
            nota("XGB", "XGBoost — 96 % de GPU", "Aprovecha casi al máximo el acelerador.", SIGNAL),
            nota("RN", "Red neuronal — 38 %", "La GPU espera datos; el cuello está en la CPU que los prepara.", GOLD),
            nota("RF", "Random Forest — 59 %", "Uso intermedio del acelerador.", COOL),
        ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}),
    ])


def slide_escalabilidad():
    return html.Div([
        seccion("07", "Escalabilidad por volumen",
                "Menor que 1.0 escala mejor de lo esperado. Mayor que 1.0 escala peor de lo esperado."),
        panel(dcc.Graph(figure=figura_escalabilidad(), config={"displayModeBar": False})),
    ])


def slide_conclusiones():
    return html.Div([
        seccion("08", "Hallazgos principales", "Lo que este proyecto demuestra sobre cómputo paralelo y distribuido."),
        html.Div([
            nota("01 / 04", "Preprocesamiento: 45× de speedup",
                 "El preprocesamiento paralelo (Polars) alcanza hasta 45× frente a la línea base secuencial, "
                 "con saturación a los 16 hilos (ley de Amdahl).", EMBER),
            nota("02 / 04", "Modelos de alta calidad",
                 "ROC-AUC de 0.98 en falsas alarmas y 0.99 en imágenes (CNN, F1 0.96). La hipótesis del ΔT se confirma.", SIGNAL),
            nota("03 / 04", "La GPU no es magia",
                 "Su eficiencia depende del algoritmo y del flujo de datos: XGBoost 96 % vs. red neuronal 38 % de uso.", GOLD),
            nota("04 / 04", "Límites del paralelismo",
                 "Memoria compartida (Polars) satura; procesos (Dask) fallan por memoria; la GPU depende de la alimentación de datos.", COOL),
        ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}),
    ])


R_LABEL = {"color": INK, "marginRight": "18px", "cursor": "pointer", "fontSize": "13px",
           "fontWeight": "600", "fontFamily": MONO}
R_INPUT = {"marginRight": "6px", "cursor": "pointer", "accentColor": EMBER}

SLIDES = [
    {"id": "resumen",       "pill": "Resumen",       "render": slide_resumen},
    {"id": "problema",      "pill": "El problema",   "render": slide_problema},
    {"id": "calidad",       "pill": "Modelos",       "render": slide_calidad},
    {"id": "speedup",       "pill": "Speedup",       "render": slide_speedup},
    {"id": "throughput",    "pill": "Throughput",    "render": slide_throughput},
    {"id": "tiempos",       "pill": "Tiempos",       "render": slide_tiempos},
    {"id": "recursos",      "pill": "Eficiencia",    "render": slide_recursos},
    {"id": "escalabilidad", "pill": "Escalabilidad", "render": slide_escalabilidad},
    {"id": "conclusiones",  "pill": "Hallazgos",     "render": slide_conclusiones},
]
SLIDE_IDS = [s["id"] for s in SLIDES]

# ===========================================================================
# 6. App
# ===========================================================================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Detección de Incendios — Panel de Resultados"
server = app.server

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
{{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  body {{ margin:0; background:{BG}; font-family:{FONT};
    background-image:
      radial-gradient(900px 480px at 88% -8%, rgba(255,94,26,0.10), transparent 60%),
      linear-gradient(0deg, {HAIR} 1px, transparent 1px),
      linear-gradient(90deg, {HAIR} 1px, transparent 1px);
    background-size: auto, 46px 46px, 46px 46px;
    background-position: center, center, center;
  }}
  ::-webkit-scrollbar {{ width:10px; }} ::-webkit-scrollbar-thumb {{ background:#232128; border-radius:0; }}
  .thermal-text {{ background:{THERMAL}; -webkit-background-clip:text; background-clip:text;
                   -webkit-text-fill-color:transparent; }}
  .pill-btn, .nav-btn {{ transition:transform 120ms ease, border-color 150ms ease, background 150ms ease; }}
  .nav-btn:hover {{ border-color:{EMBER}; }}
  .pill-btn:hover {{ border-color:{EMBER}; }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:{SIGNAL};
          margin-right:9px; box-shadow:0 0 0 0 rgba(53,224,139,.6); animation:pulse 2.2s ease-out infinite; position:relative; top:-1px; }}
  @keyframes pulse {{ 0%{{box-shadow:0 0 0 0 rgba(53,224,139,.5)}} 70%{{box-shadow:0 0 0 9px rgba(53,224,139,0)}} 100%{{box-shadow:0 0 0 0 rgba(53,224,139,0)}} }}
  .scan {{ position:fixed; top:0; left:0; right:0; height:2px; z-index:50;
           background:linear-gradient(90deg, transparent, {EMBER}, transparent);
           opacity:0.55; animation:scan 6s linear infinite; }}
  @keyframes scan {{ 0%{{transform:translateY(0)}} 100%{{transform:translateY(100vh)}} }}
  @media (prefers-reduced-motion: reduce) {{ .dot,.scan{{animation:none}} .scan{{display:none}} }}
</style>
</head>
<body><div class="scan"></div>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body>
</html>
"""

PILL = {"background": "transparent", "color": MUTED, "border": f"1px solid {HAIR}", "borderRadius": "2px",
        "padding": "8px 15px", "fontSize": "11.5px", "fontWeight": "700", "marginRight": "8px",
        "marginBottom": "8px", "cursor": "pointer", "fontFamily": MONO, "letterSpacing": "0.08em",
        "textTransform": "uppercase"}
PILL_ON = {**PILL, "background": CARD_2, "color": INK, "borderColor": EMBER, "borderLeft": f"2px solid {EMBER}"}
NAV = {"background": "transparent", "color": INK, "border": f"1px solid {HAIR}", "borderRadius": "2px",
       "width": "42px", "height": "42px", "cursor": "pointer", "fontSize": "15px"}

app.layout = html.Div([
    dcc.Store(id="idx", data=0),
    html.Div([
        html.Div([
            html.Div(style={"width": "12px", "height": "12px", "background": THERMAL,
                            "marginRight": "12px", "borderRadius": "1px"}),
            html.Div([
                html.Div("DETECCIÓN DE INCENDIOS FORESTALES",
                         style={"color": INK, "fontWeight": "700", "fontSize": "14px",
                                "fontFamily": MONO, "letterSpacing": "0.14em"}),
                html.Div("TTCT0017 · Computación Paralela y Distribuida · LEAD University",
                         style={"color": FAINT, "fontSize": "11px", "marginTop": "3px", "fontFamily": MONO}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Button("<", id="prev", n_clicks=0, className="nav-btn", style=NAV),
            html.Button(">", id="next", n_clicks=0, className="nav-btn", style={**NAV, "marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
              "padding": "20px 34px", "borderBottom": f"1px solid {HAIR}",
              "position": "sticky", "top": "0", "background": "rgba(7,7,10,0.9)",
              "backdropFilter": "blur(10px)", "zIndex": "20"}),
    html.Div([html.Button(s["pill"], id=f"pill-{s['id']}", n_clicks=0, className="pill-btn",
                          style=PILL_ON if i == 0 else PILL) for i, s in enumerate(SLIDES)],
             style={"padding": "18px 34px 4px 34px", "display": "flex", "flexWrap": "wrap",
                    "maxWidth": "1060px", "margin": "0 auto"}),
    html.Div([html.Div(s["render"](), id=f"slide-{s['id']}",
                       style={"display": "block"} if i == 0 else {"display": "none"})
              for i, s in enumerate(SLIDES)],
             style={"padding": "16px 34px 12px 34px", "maxWidth": "1060px", "margin": "0 auto"}),
    html.Div([
        html.Div(id="lbl", style={"fontSize": "10.5px", "color": FAINT, "textTransform": "uppercase",
                                  "letterSpacing": "0.22em", "fontWeight": "700", "fontFamily": MONO,
                                  "whiteSpace": "nowrap"}),
        html.Div(style={"flex": "1", "height": "3px", "background": HAIR, "marginLeft": "16px", "overflow": "hidden"},
                 children=html.Div(id="fill", style={"height": "100%", "background": THERMAL})),
    ], style={"display": "flex", "alignItems": "center", "padding": "14px 34px 30px 34px",
              "maxWidth": "1060px", "margin": "0 auto"}),
], style={"minHeight": "100vh"})


# ===========================================================================
# 7. Callbacks
# ===========================================================================
@app.callback(Output("idx", "data"), Input("prev", "n_clicks"), Input("next", "n_clicks"),
              State("idx", "data"), prevent_initial_call=True)
def navegar(p, n, idx):
    t = ctx.triggered_id
    if t == "prev":
        return max(0, idx - 1)
    if t == "next":
        return min(len(SLIDES) - 1, idx + 1)
    return idx


@app.callback(Output("idx", "data", allow_duplicate=True),
              [Input(f"pill-{sid}", "n_clicks") for sid in SLIDE_IDS], prevent_initial_call=True)
def ir(*_):
    t = ctx.triggered_id
    if t and t.startswith("pill-"):
        return SLIDE_IDS.index(t.replace("pill-", ""))
    return 0


_OUT = ([Output(f"slide-{s}", "style") for s in SLIDE_IDS]
        + [Output(f"pill-{s}", "style") for s in SLIDE_IDS]
        + [Output("fill", "style"), Output("lbl", "children")])


@app.callback(*_OUT, Input("idx", "data"))
def vista(idx):
    idx = idx or 0
    ss = [{"display": "block"} if i == idx else {"display": "none"} for i in range(len(SLIDES))]
    ps = [PILL_ON if i == idx else PILL for i in range(len(SLIDES))]
    pct = int(round((idx + 1) / len(SLIDES) * 100))
    return (*ss, *ps, {"height": "100%", "background": THERMAL, "width": f"{pct}%"},
            f"{idx + 1:02d} / {len(SLIDES):02d}")


@app.callback(Output("g-calidad", "figure"), Input("sel-metrica-calidad", "value"))
def cb_calidad(m):
    return figura_calidad(m)


@app.callback(Output("g-tiempos", "figure"), Input("sel-pct-tiempo", "value"))
def cb_tiempos(p):
    return figura_tiempos(p)


@app.callback(Output("g-recursos", "figure"),
              Input("sel-pct-recursos", "value"), Input("sel-metrica-recursos", "value"))
def cb_recursos(p, m):
    return figura_recursos(p, m)


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=puerto, debug=False)
