"""
Dashboard de Análisis de Rendimiento — Etapa 5 (estilo "slides" oscuro)
Proyecto: Detección de Incendios Forestales con Imágenes Satelitales
Responsable: Robson Sthiffen Calvo Ortega

Cómo correrlo:
    pip install -r requirements.txt
    python app.py
Luego abrí http://127.0.0.1:8050 en el navegador.

Lee un único archivo consolidado: data/benchmark_consolidado.csv
"""

from pathlib import Path
import os

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx

# ---------------------------------------------------------------------------
# 1. Paleta y configuración visual
# ---------------------------------------------------------------------------

BG = "#0b0b0d"
CARD_BG = "#17161a"
CARD_BG_ALT = "#1e1c20"
ACCENT = "#f5820d"
TEXT = "#f2f1ee"
TEXT_MUTED = "#9c9a97"
PILL_INACTIVE_BG = "#1c1b1f"
PILL_ACTIVE_BG = "#ffffff"
PILL_ACTIVE_TEXT = "#0b0b0d"
FONT = "'Inter', -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
FONT_MONO = "'IBM Plex Mono', 'Courier New', monospace"
VERDE_ESTADO = "#3ecf8e"

BASE_DIR = Path(__file__).parent
RUTA_DATOS = BASE_DIR / "data" / "benchmark_consolidado.csv"
ORDEN_PCT = ["30%", "60%", "100%"]
COLOR_MODELO = {"Random Forest": "#4fa3e0", "XGBoost": "#f5820d", "Red Neuronal": "#3ecf8e"}

# ---------------------------------------------------------------------------
# 2. Datos y cálculos (igual que la versión anterior)
# ---------------------------------------------------------------------------

def es_gpu(backend):
    return any(x in backend for x in ["GPU", "cuda", "cuML"])


def cargar_datos():
    if not RUTA_DATOS.exists():
        return pd.DataFrame()
    df = pd.read_csv(RUTA_DATOS)
    df["dispositivo"] = df["backend"].apply(lambda b: "GPU" if es_gpu(b) else "CPU")
    df["modelo"] = df["modelo"].str.replace(r"\s*\(por epoca\)", "", regex=True)
    return df


def calcular_speedup(df):
    filas = []
    for (modelo, pct), grupo in df.groupby(["modelo", "dataset_pct"]):
        cpu = grupo[grupo["dispositivo"] == "CPU"]
        gpu = grupo[grupo["dispositivo"] == "GPU"]
        if cpu.empty or gpu.empty:
            continue
        t_cpu, t_gpu = cpu["tiempo_s"].iloc[0], gpu["tiempo_s"].iloc[0]
        filas.append({"modelo": modelo, "dataset_pct": pct, "speedup": t_cpu / t_gpu if t_gpu else None})
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


datos = cargar_datos()
speedups = calcular_speedup(datos) if not datos.empty else pd.DataFrame()
escalabilidad = calcular_escalabilidad(datos) if not datos.empty else pd.DataFrame()

# ---------------------------------------------------------------------------
# 3. Figuras (tema oscuro)
# ---------------------------------------------------------------------------

LAYOUT_OSCURO = dict(
    template="plotly_dark", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, -apple-system, Segoe UI, Roboto, Arial, sans-serif"),
    margin=dict(t=20, l=60, r=20, b=50),
    legend=dict(orientation="h", y=-0.25, bgcolor="rgba(0,0,0,0)"),
)


def figura_vacia(mensaje):
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(size=14, color=TEXT_MUTED))
    fig.update_layout(**LAYOUT_OSCURO)
    return fig


def figura_speedup():
    if speedups.empty:
        return figura_vacia("No se encontraron datos de speedup")
    fig = go.Figure()
    for modelo in speedups["modelo"].unique():
        sub = speedups[speedups["modelo"] == modelo].set_index("dataset_pct").reindex(ORDEN_PCT).reset_index()
        fig.add_trace(go.Scatter(x=sub["dataset_pct"], y=sub["speedup"], mode="lines+markers",
                                  name=modelo, line=dict(color=COLOR_MODELO.get(modelo), width=3),
                                  marker=dict(size=10)))
    fig.add_hline(y=1, line_dash="dash", line_color=TEXT_MUTED, annotation_text="CPU = GPU")
    fig.update_layout(yaxis_title="Speedup (GPU / CPU)", xaxis_title="Tamaño del dataset", **LAYOUT_OSCURO)
    return fig


def figura_throughput():
    if datos.empty:
        return figura_vacia("No se encontraron datos")
    fig = go.Figure()
    for modelo in datos["modelo"].unique():
        for dispositivo, estilo in [("CPU", "dot"), ("GPU", "solid")]:
            sub = (datos[(datos["modelo"] == modelo) & (datos["dispositivo"] == dispositivo)]
                   .set_index("dataset_pct").reindex(ORDEN_PCT).reset_index())
            fig.add_trace(go.Scatter(x=sub["dataset_pct"], y=sub["throughput_reg_s"], mode="lines+markers",
                                      name=f"{modelo} ({dispositivo})",
                                      line=dict(color=COLOR_MODELO.get(modelo), dash=estilo, width=2),
                                      marker=dict(size=8)))
    fig.update_layout(yaxis_title="Throughput (registros/s)", yaxis_type="log",
                       xaxis_title="Tamaño del dataset", **LAYOUT_OSCURO)
    return fig


def figura_tiempos(dataset_pct):
    if datos.empty:
        return figura_vacia("No se encontraron datos")
    sub = datos[datos["dataset_pct"] == dataset_pct]
    fig = go.Figure()
    for dispositivo, color in [("CPU", "#5c5a57"), ("GPU", ACCENT)]:
        parte = sub[sub["dispositivo"] == dispositivo]
        fig.add_trace(go.Bar(x=parte["modelo"], y=parte["tiempo_s"], name=dispositivo, marker_color=color,
                              text=parte["tiempo_s"].round(1), textposition="outside"))
    fig.update_layout(barmode="group", yaxis_title="Tiempo (s) — escala log", yaxis_type="log", **LAYOUT_OSCURO)
    return fig


def figura_recursos(dataset_pct, metrica):
    if datos.empty:
        return figura_vacia("No se encontraron datos")
    sub = datos[datos["dataset_pct"] == dataset_pct]
    fig = go.Figure(go.Bar(x=sub["modelo"] + " — " + sub["dispositivo"], y=sub[metrica],
                            marker_color=[COLOR_MODELO.get(m, "#888") for m in sub["modelo"]]))
    etiquetas = {"ram_pico_mb": "RAM pico (MB)", "gpu_mem_pico_mb": "Memoria GPU pico (MB)",
                 "gpu_util_promedio_pct": "Uso promedio de GPU (%) — eficiencia"}
    fig.update_layout(yaxis_title=etiquetas.get(metrica, metrica), xaxis_tickangle=-30, **LAYOUT_OSCURO)
    return fig


def figura_escalabilidad():
    if escalabilidad.empty:
        return figura_vacia("No se encontraron datos")
    fig = go.Figure()
    for modelo in escalabilidad["modelo"].unique():
        for dispositivo, estilo in [("CPU", "dot"), ("GPU", "solid")]:
            sub = (escalabilidad[(escalabilidad["modelo"] == modelo) & (escalabilidad["dispositivo"] == dispositivo)]
                   .set_index("dataset_pct").reindex(["60%", "100%"]).reset_index())
            fig.add_trace(go.Scatter(x=sub["dataset_pct"], y=sub["indice_escalabilidad"], mode="lines+markers",
                                      name=f"{modelo} ({dispositivo})",
                                      line=dict(color=COLOR_MODELO.get(modelo), dash=estilo, width=2),
                                      marker=dict(size=8)))
    fig.add_hline(y=1, line_dash="dash", line_color=TEXT_MUTED, annotation_text="Crecimiento lineal (ideal)")
    fig.update_layout(yaxis_title="Índice de escalabilidad (1.0 = lineal)",
                       xaxis_title="Tamaño del dataset (relativo a 30%)", **LAYOUT_OSCURO)
    return fig


# ---------------------------------------------------------------------------
# 4. Definición de "slides"
# ---------------------------------------------------------------------------

def tarjeta_kpi(etiqueta, valor):
    return html.Div([
        html.Div(etiqueta, style={"fontSize": "12px", "color": TEXT_MUTED, "textTransform": "uppercase",
                                   "letterSpacing": "0.06em", "fontWeight": "600"}),
        html.Div(valor, style={"fontSize": "26px", "fontWeight": "600", "color": TEXT, "marginTop": "6px",
                                "fontFamily": FONT_MONO}),
    ], style={"background": CARD_BG_ALT, "borderRadius": "12px", "padding": "20px 22px", "flex": "1",
              "border": "1px solid #242226", "boxShadow": "0 6px 18px rgba(0,0,0,0.25)"})


def tarjeta_panel(children):
    return html.Div(children, style={"background": CARD_BG, "borderRadius": "14px",
                                      "padding": "20px", "border": "1px solid #2a282d",
                                      "boxShadow": "0 8px 24px rgba(0,0,0,0.3)"})


def slide_portada():
    if not speedups.empty:
        fila_max = speedups.loc[speedups["speedup"].idxmax()]
        max_speedup, modelo_max, pct_max = fila_max["speedup"], fila_max["modelo"], fila_max["dataset_pct"]
    else:
        max_speedup, modelo_max, pct_max = 0, "-", "-"
    return html.Div([
        html.Div("Análisis de Rendimiento", style={"color": ACCENT, "fontSize": "36px", "fontWeight": "800"}),
        html.Div("CPU vs GPU · Random Forest, XGBoost y Red Neuronal · 30/60/100% del dataset",
                  style={"color": TEXT_MUTED, "fontSize": "15px", "marginTop": "6px", "marginBottom": "28px"}),
        html.Div([
            tarjeta_kpi("Registros (100%)", "36.8 M"),
            tarjeta_kpi("Modelos comparados", "3"),
            tarjeta_kpi("Mayor speedup GPU", f"{max_speedup:.1f}x ({modelo_max}, {pct_max})"),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "20px"}),
        tarjeta_panel([
            html.Div([html.Span(className="status-dot"), "Hardware verificado"],
                     style={"color": ACCENT, "fontWeight": "700", "marginBottom": "8px"}),
            html.Div("Nodo nukwa-01.cnca · GPU NVIDIA V100 · 24 núcleos CPU — confirmado con sacct/scontrol en Kabré.",
                      style={"color": TEXT, "fontSize": "14px", "fontFamily": FONT_MONO}),
        ]),
    ])


def slide_speedup():
    return html.Div([
        html.Div("Speedup GPU vs CPU", style={"color": ACCENT, "fontSize": "28px", "fontWeight": "800", "marginBottom": "16px"}),
        tarjeta_panel(dcc.Graph(figure=figura_speedup(), config={"displayModeBar": False})),
    ])


def slide_throughput():
    return html.Div([
        html.Div("Throughput (registros/segundo)", style={"color": ACCENT, "fontSize": "28px", "fontWeight": "800", "marginBottom": "6px"}),
        html.Div("Escala logarítmica — GPU puede ser 10-100x más grande que CPU.",
                  style={"color": TEXT_MUTED, "fontSize": "13px", "marginBottom": "16px"}),
        tarjeta_panel(dcc.Graph(figure=figura_throughput(), config={"displayModeBar": False})),
    ])


RADIO_LABEL_STYLE = {"color": TEXT, "marginRight": "18px", "cursor": "pointer", "fontSize": "14px"}
RADIO_INPUT_STYLE = {"marginRight": "6px", "cursor": "pointer"}


def slide_tiempos():
    return html.Div([
        html.Div("Tiempo de entrenamiento", style={"color": ACCENT, "fontSize": "28px", "fontWeight": "800", "marginBottom": "16px"}),
        dcc.RadioItems(id="selector-pct-tiempo", options=[{"label": p, "value": p} for p in ORDEN_PCT],
                       value="100%", inline=True, style={"marginBottom": "14px"},
                       labelStyle=RADIO_LABEL_STYLE, inputStyle=RADIO_INPUT_STYLE),
        tarjeta_panel(dcc.Graph(id="grafico-tiempos", config={"displayModeBar": False})),
    ])


def slide_recursos():
    return html.Div([
        html.Div("Uso de recursos y eficiencia", style={"color": ACCENT, "fontSize": "28px", "fontWeight": "800", "marginBottom": "6px"}),
        html.Div("Eficiencia = % real de uso de GPU (nvidia-smi). Ser rápido no siempre es usar bien el hardware.",
                  style={"color": TEXT_MUTED, "fontSize": "13px", "marginBottom": "14px"}),
        html.Div([
            dcc.RadioItems(id="selector-pct-recursos", options=[{"label": p, "value": p} for p in ORDEN_PCT],
                           value="100%", inline=True, labelStyle=RADIO_LABEL_STYLE, inputStyle=RADIO_INPUT_STYLE),
            dcc.Dropdown(id="selector-metrica-recursos", options=[
                {"label": "RAM pico (MB)", "value": "ram_pico_mb"},
                {"label": "Memoria GPU pico (MB)", "value": "gpu_mem_pico_mb"},
                {"label": "Uso de GPU (%) — eficiencia", "value": "gpu_util_promedio_pct"},
            ], value="gpu_util_promedio_pct", clearable=False, style={"width": "360px", "marginTop": "10px", "color": "#111"}),
        ], style={"marginBottom": "16px"}),
        tarjeta_panel(dcc.Graph(id="grafico-recursos", config={"displayModeBar": False})),
    ])


def slide_escalabilidad():
    return html.Div([
        html.Div("Escalabilidad por volumen de datos", style={"color": ACCENT, "fontSize": "28px", "fontWeight": "800", "marginBottom": "6px"}),
        html.Div("< 1.0 = escala mejor de lo esperado. > 1.0 = escala peor de lo esperado.",
                  style={"color": TEXT_MUTED, "fontSize": "13px", "marginBottom": "16px"}),
        tarjeta_panel(dcc.Graph(figure=figura_escalabilidad(), config={"displayModeBar": False})),
    ])


SLIDES = [
    {"id": "portada", "pill": "Resumen", "render": slide_portada},
    {"id": "speedup", "pill": "Speedup", "render": slide_speedup},
    {"id": "throughput", "pill": "Throughput", "render": slide_throughput},
    {"id": "tiempos", "pill": "Tiempos", "render": slide_tiempos},
    {"id": "recursos", "pill": "Eficiencia", "render": slide_recursos},
    {"id": "escalabilidad", "pill": "Escalabilidad", "render": slide_escalabilidad},
]
SLIDE_IDS = [s["id"] for s in SLIDES]

# ---------------------------------------------------------------------------
# 5. App y layout general
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Rendimiento — Detección de Incendios Forestales"

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>{{%title%}}</title>
{{%favicon%}}
{{%css%}}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  body {{ background: {BG}; margin: 0; font-family: {FONT}; }}
  ::-webkit-scrollbar {{ width: 10px; }}
  ::-webkit-scrollbar-thumb {{ background: #2a282d; border-radius: 6px; }}

  .nav-btn, .pill-btn, .csv-btn {{
    transition: transform 120ms ease, box-shadow 120ms ease, opacity 120ms ease;
  }}
  .nav-btn:hover, .csv-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 14px rgba(0,0,0,0.35); }}
  .nav-btn:active, .csv-btn:active {{ transform: translateY(0px); }}
  .pill-btn:hover {{ box-shadow: 0 2px 10px rgba(0,0,0,0.3); }}

  .nav-btn:focus-visible, .pill-btn:focus-visible, .csv-btn:focus-visible {{
    outline: 2px solid {ACCENT}; outline-offset: 2px;
  }}

  .status-dot {{
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: {VERDE_ESTADO}; margin-right: 8px; position: relative; top: -1px;
    box-shadow: 0 0 0 0 rgba(62,207,142,0.6);
    animation: pulso 2.2s ease-out infinite;
  }}
  @keyframes pulso {{
    0%   {{ box-shadow: 0 0 0 0 rgba(62,207,142,0.55); }}
    70%  {{ box-shadow: 0 0 0 8px rgba(62,207,142,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(62,207,142,0); }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    .status-dot {{ animation: none; }}
    .nav-btn, .pill-btn, .csv-btn {{ transition: none; }}
  }}
</style>
</head>
<body>
{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>
"""

PILL_STYLE = {
    "background": PILL_INACTIVE_BG, "color": TEXT, "border": "none", "borderRadius": "999px",
    "padding": "8px 18px", "fontSize": "13px", "fontWeight": "600", "marginRight": "8px",
    "cursor": "pointer",
}
PILL_STYLE_ACTIVE = {**PILL_STYLE, "background": PILL_ACTIVE_BG, "color": PILL_ACTIVE_TEXT}

NAV_BTN_STYLE = {
    "background": CARD_BG_ALT, "color": TEXT, "border": "1px solid #2a282d", "borderRadius": "50%",
    "width": "38px", "height": "38px", "cursor": "pointer", "fontSize": "16px",
}

app.layout = html.Div([
    dcc.Store(id="slide-idx", data=0),

    # Barra superior
    html.Div([
        html.Div([
            html.Div("Detección de Incendios Forestales", style={"color": TEXT, "fontWeight": "800", "fontSize": "16px"}),
            html.Div("Análisis de Rendimiento · TTCT0017, Computación Paralela y Distribuida",
                      style={"color": TEXT_MUTED, "fontSize": "12px"}),
        ]),
        html.Div([
            html.Button("◀", id="btn-prev", n_clicks=0, className="nav-btn", style=NAV_BTN_STYLE),
            html.Button("▶", id="btn-next", n_clicks=0, className="nav-btn", style={**NAV_BTN_STYLE, "marginLeft": "8px"}),
            html.Button("⬇ CSV", id="btn-csv", n_clicks=0, className="csv-btn",
                        style={**NAV_BTN_STYLE, "width": "auto", "borderRadius": "999px",
                               "padding": "0 16px", "marginLeft": "8px", "color": ACCENT}),
            dcc.Download(id="descarga-csv"),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
              "padding": "18px 28px", "borderBottom": f"1px solid #221f22"}),

    # Píldoras de navegación (declaradas desde el inicio, no se recrean)
    html.Div([
        html.Button(s["pill"], id=f"pill-{s['id']}", n_clicks=0, className="pill-btn",
                     style=PILL_STYLE_ACTIVE if i == 0 else PILL_STYLE)
        for i, s in enumerate(SLIDES)
    ], id="pill-nav", style={"padding": "16px 28px 0px 28px", "display": "flex", "flexWrap": "wrap"}),

    # Todas las slides existen desde el arranque; se muestran/ocultan con CSS
    html.Div([
        html.Div(s["render"](), id=f"slide-{s['id']}",
                  style={"display": "block"} if i == 0 else {"display": "none"})
        for i, s in enumerate(SLIDES)
    ], id="slide-content", style={"padding": "20px 28px 10px 28px", "maxWidth": "980px"}),

    # Barra de progreso
    html.Div([
        html.Div(id="progress-track", style={"flex": "1", "height": "4px", "background": "#221f22",
                                              "borderRadius": "4px", "overflow": "hidden"},
                  children=html.Div(id="progress-fill", style={"height": "100%", "background": ACCENT})),
        html.Div(id="progress-label", style={"color": TEXT_MUTED, "fontSize": "13px", "marginLeft": "14px", "whiteSpace": "nowrap"}),
    ], style={"display": "flex", "alignItems": "center", "padding": "16px 28px 24px 28px"}),

], style={"background": BG, "minHeight": "100vh"})


# ---------------------------------------------------------------------------
# 6. Callbacks de navegación
# ---------------------------------------------------------------------------

@app.callback(
    Output("slide-idx", "data"),
    Input("btn-prev", "n_clicks"), Input("btn-next", "n_clicks"),
    State("slide-idx", "data"),
    prevent_initial_call=True,
)
def navegar(n_prev, n_next, idx):
    trig = ctx.triggered_id
    if trig == "btn-prev":
        return max(0, idx - 1)
    if trig == "btn-next":
        return min(len(SLIDES) - 1, idx + 1)
    return idx


@app.callback(
    Output("slide-idx", "data", allow_duplicate=True),
    [Input(f"pill-{sid}", "n_clicks") for sid in SLIDE_IDS],
    prevent_initial_call=True,
)
def ir_a_pill(*_clicks):
    trig = ctx.triggered_id
    if trig and trig.startswith("pill-"):
        return SLIDE_IDS.index(trig.replace("pill-", ""))
    return 0


_OUTPUTS_VISTA = (
    [Output(f"slide-{sid}", "style") for sid in SLIDE_IDS]
    + [Output(f"pill-{sid}", "style") for sid in SLIDE_IDS]
    + [Output("progress-fill", "style"), Output("progress-label", "children")]
)


@app.callback(*_OUTPUTS_VISTA, Input("slide-idx", "data"))
def actualizar_vista(idx):
    idx = idx or 0
    slide_styles = [{"display": "block"} if i == idx else {"display": "none"} for i in range(len(SLIDES))]
    pill_styles = [PILL_STYLE_ACTIVE if i == idx else PILL_STYLE for i in range(len(SLIDES))]
    pct = int(round((idx + 1) / len(SLIDES) * 100))
    fill_style = {"height": "100%", "background": ACCENT, "width": f"{pct}%"}
    label = f"{idx + 1} / {len(SLIDES)}"
    return (*slide_styles, *pill_styles, fill_style, label)


@app.callback(Output("grafico-tiempos", "figure"), Input("selector-pct-tiempo", "value"))
def actualizar_tiempos(pct):
    return figura_tiempos(pct)


@app.callback(Output("grafico-recursos", "figure"),
              Input("selector-pct-recursos", "value"), Input("selector-metrica-recursos", "value"))
def actualizar_recursos(pct, metrica):
    return figura_recursos(pct, metrica)


@app.callback(Output("descarga-csv", "data"), Input("btn-csv", "n_clicks"), prevent_initial_call=True)
def descargar_csv(n_clicks):
    return dcc.send_file(str(RUTA_DATOS))


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=puerto, debug=False)
