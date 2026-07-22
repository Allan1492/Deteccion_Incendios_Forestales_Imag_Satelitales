#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analizar_resultados.py -- Calcula metricas de rendimiento y genera graficos.

Proyecto: Deteccion de Incendios Forestales con Imagenes Satelitales
Curso   : TTCT0017 - Computacion Paralela y Distribuida (LEAD University)
Etapa 3 : Primeras mediciones de rendimiento (secuencial vs. paralelo)

QUE CALCULA
-----------
  Speedup      S(p) = T_1 / T_p     (T_1 = linea base secuencial en pandas)
  Eficiencia   E(p) = S(p) / p
  Throughput   filas/s y MB/s
  Escalabilidad fuerte : tamano fijo, p creciente
  Escalabilidad debil  : tamano proporcional a p (el tiempo deberia ser plano)
  Memoria pico por configuracion

SALIDAS
-------
  tabla_resumen.csv        Promedios por configuracion con speedup/eficiencia
  tabla_resumen.md         La misma tabla en Markdown (para el informe)
  figuras/*.png            Graficos listos para el informe IEEE

USO
---
    python3 analizar_resultados.py resultados_benchmark.csv
    python3 analizar_resultados.py resultados_benchmark.csv --figuras figuras/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")          # backend headless: obligatorio en el cluster
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Carga y agregacion
# ---------------------------------------------------------------------------

def cargar(ruta):
    if not os.path.exists(ruta):
        print("ERROR: no existe %s" % ruta, file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(ruta)
    print("Corridas leidas: %d" % len(df))
    return df


def promediar(df):
    """Promedia las repeticiones de cada configuracion.

    Se usa la MEDIANA del tiempo en vez de la media porque es mas robusta
    frente a corridas atipicas (por ejemplo, la primera lectura que aun no
    tiene el archivo en cache del sistema operativo).
    """
    claves = ["backend", "operacion", "hilos", "escalabilidad", "frac"]
    agg = df.groupby(claves, as_index=False).agg(
        tiempo_s=("tiempo_s", "median"),
        tiempo_min=("tiempo_s", "min"),
        tiempo_max=("tiempo_s", "max"),
        filas=("filas_resultado", "median"),
        throughput_filas_s=("throughput_filas_s", "median"),
        throughput_mb_s=("throughput_mb_s", "median"),
        memoria_pico_mb=("memoria_pico_mb", "max"),
        n_corridas=("tiempo_s", "size"),
    )
    return agg


def agregar_speedup(agg):
    """Anade DOS speedups distintos, porque miden cosas diferentes.

    DISTINCION METODOLOGICA IMPORTANTE
    ----------------------------------
    1) speedup_global = T_pandas / T_p
       Compara contra la linea base SECUENCIAL (pandas, 1 hilo). Mide la
       ganancia total de cambiar de tecnologia Y paralelizar. Es la cifra
       que responde "cuanto ganamos en total".

    2) speedup_paralelo = T_backend(1 hilo) / T_backend(p hilos)
       Compara cada motor CONSIGO MISMO. Aisla el efecto del paralelismo
       puro, sin mezclarlo con que Polars ya es mas rapido que pandas
       de por si.

    La EFICIENCIA se calcula con speedup_paralelo, nunca con el global:
    dividir un speedup entre motores distintos por el numero de hilos
    produce eficiencias absurdas (>100%), porque atribuye al paralelismo
    una ganancia que en realidad viene del motor.
    """
    fuerte = agg[agg["escalabilidad"] == "fuerte"].copy()

    # --- Base secuencial global (pandas) por operacion ---------------------
    base_global = {}
    for op in fuerte["operacion"].unique():
        sub = fuerte[(fuerte["operacion"] == op) & (fuerte["backend"] == "pandas")]
        origen = "pandas-1hilo"
        if sub.empty:   # respaldo si no se corrio pandas
            sub = fuerte[(fuerte["operacion"] == op) &
                         (fuerte["backend"] == "polars") & (fuerte["hilos"] == 1)]
            origen = "polars-1hilo"
        if not sub.empty:
            base_global[op] = (float(sub["tiempo_s"].iloc[0]), origen)

    # --- Base intra-backend (mismo motor con 1 hilo) -----------------------
    base_backend = {}
    for (op, backend) in fuerte[["operacion", "backend"]].drop_duplicates().values:
        sub = fuerte[(fuerte["operacion"] == op) &
                     (fuerte["backend"] == backend) & (fuerte["hilos"] == 1)]
        if not sub.empty:
            base_backend[(op, backend)] = float(sub["tiempo_s"].iloc[0])

    def _sp_global(f):
        if f["escalabilidad"] != "fuerte" or f["tiempo_s"] <= 0:
            return float("nan")
        b = base_global.get(f["operacion"])
        return b[0] / f["tiempo_s"] if b else float("nan")

    def _sp_paralelo(f):
        if f["escalabilidad"] != "fuerte" or f["tiempo_s"] <= 0:
            return float("nan")
        t1 = base_backend.get((f["operacion"], f["backend"]))
        return t1 / f["tiempo_s"] if t1 else float("nan")

    agg["speedup_global"] = agg.apply(_sp_global, axis=1)
    agg["speedup_paralelo"] = agg.apply(_sp_paralelo, axis=1)
    agg["base_usada"] = agg["operacion"].map(
        lambda o: base_global[o][1] if o in base_global else "")

    # Eficiencia: SOLO con el speedup intra-backend y solo para motores
    # paralelos (pandas es la referencia secuencial, no aplica).
    agg["eficiencia"] = agg.apply(
        lambda f: (f["speedup_paralelo"] / f["hilos"])
        if (f["escalabilidad"] == "fuerte" and f["hilos"] > 0
            and f["backend"] != "pandas" and pd.notna(f["speedup_paralelo"]))
        else float("nan"),
        axis=1)

    # Alias por compatibilidad con el resto del script
    agg["speedup"] = agg["speedup_paralelo"]
    return agg


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------

COLORES = {"pandas": "#7f7f7f", "polars": "#1f77b4",
           "dask": "#ff7f0e", "cudf": "#2ca02c"}


def _guardar(fig, carpeta, nombre):
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  figura -> %s" % ruta)


def grafico_speedup(agg, carpeta):
    fuerte = agg[(agg["escalabilidad"] == "fuerte") &
                 (agg["backend"].isin(["polars", "dask"]))]
    if fuerte.empty:
        return
    for op in sorted(fuerte["operacion"].unique()):
        sub = fuerte[fuerte["operacion"] == op]
        fig, ax = plt.subplots(figsize=(6, 4))
        for backend in sorted(sub["backend"].unique()):
            s = sub[sub["backend"] == backend].sort_values("hilos")
            ax.plot(s["hilos"], s["speedup"], marker="o",
                    label=backend, color=COLORES.get(backend))
        maxp = int(sub["hilos"].max())
        ideal = list(range(1, maxp + 1))
        ax.plot(ideal, ideal, "k--", alpha=0.4, label="ideal (lineal)")
        ax.set_xlabel("Numero de hilos / workers (p)")
        ax.set_ylabel("Speedup  S(p) = $T_1/T_p$")
        ax.set_title("Escalabilidad fuerte -- %s" % op)
        ax.legend()
        ax.grid(alpha=0.3)
        _guardar(fig, carpeta, "speedup_%s.png" % op)


def grafico_eficiencia(agg, carpeta):
    fuerte = agg[(agg["escalabilidad"] == "fuerte") &
                 (agg["backend"].isin(["polars", "dask"]))]
    if fuerte.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for backend in sorted(fuerte["backend"].unique()):
        s = (fuerte[fuerte["backend"] == backend]
             .groupby("hilos", as_index=False)["eficiencia"].mean()
             .sort_values("hilos"))
        ax.plot(s["hilos"], s["eficiencia"] * 100, marker="s",
                label=backend, color=COLORES.get(backend))
    ax.axhline(100, color="k", linestyle="--", alpha=0.4, label="ideal (100%)")
    ax.set_xlabel("Numero de hilos / workers (p)")
    ax.set_ylabel("Eficiencia E(p) [%]")
    ax.set_title("Eficiencia paralela (promedio de operaciones)")
    ax.legend()
    ax.grid(alpha=0.3)
    _guardar(fig, carpeta, "eficiencia.png")


def grafico_tiempos(agg, carpeta):
    fuerte = agg[agg["escalabilidad"] == "fuerte"]
    if fuerte.empty:
        return
    ops = sorted(fuerte["operacion"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    ancho = 0.8 / max(len(ops), 1)
    backends = sorted(fuerte["backend"].unique())
    x = range(len(backends))
    for i, op in enumerate(ops):
        alturas = []
        for b in backends:
            s = fuerte[(fuerte["backend"] == b) & (fuerte["operacion"] == op)]
            # para backends paralelos se toma la mejor configuracion
            alturas.append(s["tiempo_s"].min() if not s.empty else 0)
        ax.bar([xi + i * ancho for xi in x], alturas, ancho, label=op)
    ax.set_xticks([xi + 0.4 - ancho / 2 for xi in x])
    ax.set_xticklabels(backends)
    ax.set_ylabel("Tiempo (s) -- mejor configuracion")
    ax.set_title("Tiempo por operacion y backend")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _guardar(fig, carpeta, "tiempos_por_backend.png")


def grafico_throughput(agg, carpeta):
    fuerte = agg[(agg["escalabilidad"] == "fuerte") &
                 (agg["backend"].isin(["polars", "dask"]))]
    if fuerte.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for backend in sorted(fuerte["backend"].unique()):
        s = (fuerte[fuerte["backend"] == backend]
             .groupby("hilos", as_index=False)["throughput_filas_s"].mean()
             .sort_values("hilos"))
        ax.plot(s["hilos"], s["throughput_filas_s"] / 1e6, marker="^",
                label=backend, color=COLORES.get(backend))
    ax.set_xlabel("Numero de hilos / workers (p)")
    ax.set_ylabel("Throughput (millones de filas/s)")
    ax.set_title("Throughput segun el paralelismo")
    ax.legend()
    ax.grid(alpha=0.3)
    _guardar(fig, carpeta, "throughput.png")


def grafico_escalabilidad_debil(agg, carpeta):
    debil = agg[agg["escalabilidad"] == "debil"]
    if debil.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for op in sorted(debil["operacion"].unique()):
        s = debil[debil["operacion"] == op].sort_values("hilos")
        ax.plot(s["hilos"], s["tiempo_s"], marker="o", label=op)
    ax.set_xlabel("Numero de hilos (p), con carga proporcional a p")
    ax.set_ylabel("Tiempo (s)")
    ax.set_title("Escalabilidad debil (ideal = linea plana)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _guardar(fig, carpeta, "escalabilidad_debil.png")


def grafico_memoria(agg, carpeta):
    fuerte = agg[agg["escalabilidad"] == "fuerte"]
    if fuerte.empty:
        return
    s = fuerte.groupby("backend", as_index=False)["memoria_pico_mb"].max()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(s["backend"], s["memoria_pico_mb"],
           color=[COLORES.get(b, "#888") for b in s["backend"]])
    ax.set_ylabel("Memoria pico (MB)")
    ax.set_title("Uso maximo de memoria por backend")
    ax.grid(alpha=0.3, axis="y")
    _guardar(fig, carpeta, "memoria.png")


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def escribir_tablas(agg, prefijo):
    cols = ["backend", "operacion", "hilos", "escalabilidad", "frac",
            "tiempo_s", "speedup_global", "speedup_paralelo", "eficiencia",
            "throughput_filas_s", "throughput_mb_s", "memoria_pico_mb",
            "n_corridas"]
    t = agg[cols].sort_values(["escalabilidad", "operacion", "backend", "hilos"])
    t.to_csv("%s.csv" % prefijo, index=False)
    print("Tabla -> %s.csv" % prefijo)

    try:
        with open("%s.md" % prefijo, "w", encoding="utf-8") as fh:
            fh.write("# Resultados del benchmark -- Etapa 3\n\n")
            fh.write(t.round(4).to_markdown(index=False))
            fh.write("\n")
        print("Tabla -> %s.md" % prefijo)
    except Exception:
        pass  # to_markdown requiere tabulate; no es critico


def resumen_consola(agg):
    print("")
    print("=" * 70)
    print("RESUMEN DE HALLAZGOS")
    print("=" * 70)

    fuerte = agg[agg["escalabilidad"] == "fuerte"]
    if fuerte.empty:
        return

    par = fuerte[fuerte["backend"].isin(["polars", "dask", "cudf"])]

    # 1) Ganancia TOTAL frente a la linea base secuencial (pandas)
    if not par.empty and par["speedup_global"].notna().any():
        mejor = par.loc[par["speedup_global"].idxmax()]
        print("Mejor speedup GLOBAL (vs %s): %.2fx"
              % (mejor.get("base_usada", "pandas"), mejor["speedup_global"]))
        print("  backend=%s  operacion=%s  hilos=%d"
              % (mejor["backend"], mejor["operacion"], mejor["hilos"]))
        print("  Nota: incluye la ganancia del motor Y del paralelismo.")

    # 2) Ganancia atribuible SOLO al paralelismo
    if not par.empty and par["speedup_paralelo"].notna().any():
        mejor2 = par.loc[par["speedup_paralelo"].idxmax()]
        print("")
        print("Mejor speedup PARALELO (mismo motor, 1 hilo -> p): %.2fx"
              % mejor2["speedup_paralelo"])
        print("  backend=%s  operacion=%s  hilos=%d"
              % (mejor2["backend"], mejor2["operacion"], mejor2["hilos"]))
        print("  Esta es la cifra que mide el paralelismo en sentido estricto.")
    print("")

    for backend in ["polars", "dask"]:
        s = fuerte[fuerte["backend"] == backend]
        if s.empty:
            continue
        maxp = s["hilos"].max()
        ef = s[s["hilos"] == maxp]["eficiencia"].mean()
        if pd.notna(ef):
            print("Eficiencia de %s con %d hilos: %.1f%%"
                  % (backend, maxp, ef * 100))
            if ef < 0.5:
                print("  -> Eficiencia baja: indica saturacion de memoria/IO")
                print("     o sobrecarga de coordinacion. Discutirlo en el")
                print("     informe con la ley de Amdahl.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV generado por bench_run.py")
    ap.add_argument("--figuras", default="figuras",
                    help="Carpeta de salida para los graficos")
    ap.add_argument("--tabla", default="tabla_resumen",
                    help="Prefijo de los archivos de tabla")
    args = ap.parse_args()

    df = cargar(args.csv)
    agg = agregar_speedup(promediar(df))

    escribir_tablas(agg, args.tabla)

    print("")
    print("Generando figuras en '%s/'..." % args.figuras)
    grafico_speedup(agg, args.figuras)
    grafico_eficiencia(agg, args.figuras)
    grafico_tiempos(agg, args.figuras)
    grafico_throughput(agg, args.figuras)
    grafico_escalabilidad_debil(agg, args.figuras)
    grafico_memoria(agg, args.figuras)

    resumen_consola(agg)


if __name__ == "__main__":
    main()
