#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_run.py — Ejecuta UNA configuración de benchmark y guarda una fila de
resultados en un CSV.

Proyecto: Deteccion de Incendios Forestales con Imagenes Satelitales
Curso   : TTCT0017 - Computacion Paralela y Distribuida (LEAD University)
Etapa 3 : Primeras mediciones de rendimiento (secuencial vs. paralelo)
Autor   : Esteban Gutierrez Saborio

POR QUE UN SCRIPT Y NO UN NOTEBOOK
----------------------------------
El numero de hilos de Polars se fija con la variable de entorno
POLARS_MAX_THREADS y solo tiene efecto ANTES de importar polars. En un
notebook no se puede cambiar despues del primer import, asi que cada
configuracion debe correr en un proceso nuevo. Por eso este script mide una
sola configuracion y el driver (run_all.sh) lo invoca varias veces.

USO
---
  python bench_run.py --backend polars --threads 4 --op agregacion \
                      --datos /ruta/incendios.parquet --salida resultados.csv

BACKENDS
--------
  pandas : linea base SECUENCIAL real (un solo hilo)
  polars : motor columnar multihilo (se escala con --threads)
  dask   : paralelismo por tareas / distribuido (se escala con --threads)
  cudf   : aceleracion por GPU (opcional, requiere RAPIDS)
"""

import argparse
import csv
import json
import os
import platform
import socket
import sys
import time

# ---------------------------------------------------------------------------
# 1) ARGUMENTOS  (se parsean ANTES de importar las librerias pesadas, porque
#    POLARS_MAX_THREADS debe fijarse antes del import de polars)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark de una configuracion.")
    p.add_argument("--backend", required=True,
                   choices=["pandas", "polars", "dask", "cudf"],
                   help="Motor de procesamiento a evaluar.")
    p.add_argument("--threads", type=int, default=1,
                   help="Hilos (polars) o workers (dask). pandas ignora este valor.")
    p.add_argument("--op", required=True,
                   choices=["carga", "filtro", "agregacion", "features", "todas"],
                   help="Operacion a medir.")
    p.add_argument("--datos", required=True,
                   help="Ruta al archivo .parquet de FIRMS.")
    p.add_argument("--salida", default="resultados_benchmark.csv",
                   help="CSV donde se agrega la fila de resultados.")
    p.add_argument("--frac", type=float, default=1.0,
                   help="Fraccion del dataset a usar (1.0 = todo). "
                        "Sirve para escalabilidad DEBIL.")
    p.add_argument("--repeticion", type=int, default=1,
                   help="Numero de repeticion (para promediar varias corridas).")
    p.add_argument("--escalabilidad", default="fuerte",
                   choices=["fuerte", "debil"],
                   help="Tipo de experimento al que pertenece esta corrida.")
    return p.parse_args()


ARGS = parse_args()

# Fijar hilos ANTES de importar polars / numpy. Critico para que la medicion
# sea valida: si no, Polars usa todos los nucleos disponibles siempre.
if ARGS.backend == "polars":
    os.environ["POLARS_MAX_THREADS"] = str(ARGS.threads)
if ARGS.backend == "pandas":
    # Linea base estrictamente secuencial: se limita BLAS/OpenMP a 1 hilo.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

# ---------------------------------------------------------------------------
# 2) UTILIDADES DE MEDICION
# ---------------------------------------------------------------------------

def memoria_pico_mb():
    """Memoria maxima residente (RSS) del proceso, en MB.

    Se usa resource.getrusage, disponible en Linux (Kabre). En Linux
    ru_maxrss viene en kilobytes; en macOS viene en bytes.
    """
    try:
        import resource
        pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return pico / (1024.0 * 1024.0)   # bytes -> MB
        return pico / 1024.0                   # KB -> MB
    except Exception:
        return float("nan")


def nucleos_disponibles():
    """Nucleos realmente asignados al job (respeta la reserva de SLURM)."""
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n:
        return int(n)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# 3) OPERACIONES DEL PIPELINE
#    Se eligieron cuatro operaciones representativas del preprocesamiento
#    real del proyecto, no operaciones artificiales.
# ---------------------------------------------------------------------------
#   carga      : leer el parquet y materializarlo         (I/O + descompresion)
#   filtro     : descartar detecciones de baja confianza  (filtrado por fila)
#   agregacion : conteo y FRP medio por satelite y mes    (group-by, shuffle)
#   features   : delta_t = brightness - bright_t31, mes,  (transformacion
#                hora  -> variables del modelo             columna a columna)
# ---------------------------------------------------------------------------

COLUMNAS_MINIMAS = ["latitude", "longitude", "brightness", "bright_t31",
                    "frp", "acq_date", "acq_time", "satellite", "confidence",
                    "daynight"]

# Valores de `confidence` que indican BAJA confianza.
# FIRMS los codifica distinto segun el sensor:
#   VIIRS -> "l" / "n" / "h"      (low / nominal / high)
#   MODIS -> "low" / "nominal" / "high"
# Se contemplan ambas formas para que el filtro funcione con cualquiera.
# En la formulacion del problema, la baja confianza opera como *proxy* de
# falsa alarma, de modo que este filtro es exactamente el descarte que el
# pipeline aplicara en produccion.
VALORES_BAJA_CONFIANZA = ["l", "low"]


def _columnas_presentes(disponibles):
    """Intersecta las columnas que necesitamos con las que trae el archivo."""
    return [c for c in COLUMNAS_MINIMAS if c in disponibles]


# ----------------------------- PANDAS (secuencial) -------------------------

def correr_pandas(ruta, op, frac):
    import pandas as pd
    import pyarrow.parquet as pq

    esquema = pq.ParquetFile(ruta).schema_arrow.names
    cols = _columnas_presentes(esquema)

    df = pd.read_parquet(ruta, columns=cols)
    if frac < 1.0:
        df = df.iloc[: int(len(df) * frac)]

    if op == "carga":
        return len(df)

    if op == "filtro":
        if "confidence" in df.columns:
            baja = df["confidence"].astype(str).str.lower().isin(VALORES_BAJA_CONFIANZA)
            res = df[~baja]
        else:
            res = df[df["frp"] > 0]
        return len(res)

    if op == "agregacion":
        d = df.copy()
        if "acq_date" in d.columns:
            d["mes"] = pd.to_datetime(d["acq_date"]).dt.month
            claves = ["satellite", "mes"] if "satellite" in d.columns else ["mes"]
        else:
            claves = ["satellite"]
        res = d.groupby(claves).agg(n=("frp", "size"), frp_medio=("frp", "mean"))
        return len(res)

    if op == "features":
        d = df.copy()
        if {"brightness", "bright_t31"}.issubset(d.columns):
            d["delta_t"] = d["brightness"] - d["bright_t31"]
        if "acq_date" in d.columns:
            d["mes"] = pd.to_datetime(d["acq_date"]).dt.month
        if "acq_time" in d.columns:
            d["hora"] = d["acq_time"] // 100
        return len(d)

    raise ValueError(op)


# ----------------------------- POLARS (multihilo) --------------------------

def correr_polars(ruta, op, frac):
    import polars as pl

    lf = pl.scan_parquet(ruta)
    cols = _columnas_presentes(lf.collect_schema().names())
    lf = lf.select(cols)

    if frac < 1.0:
        total = pl.scan_parquet(ruta).select(pl.len()).collect().item()
        lf = lf.head(int(total * frac))

    if op == "carga":
        return lf.collect().height

    if op == "filtro":
        if "confidence" in cols:
            lf2 = lf.filter(
                ~pl.col("confidence").cast(pl.Utf8).str.to_lowercase()
                  .is_in(VALORES_BAJA_CONFIANZA)
            )
        else:
            lf2 = lf.filter(pl.col("frp") > 0)
        return lf2.collect().height

    if op == "agregacion":
        d = lf
        if "acq_date" in cols:
            d = d.with_columns(pl.col("acq_date").dt.month().alias("mes"))
            claves = ["satellite", "mes"] if "satellite" in cols else ["mes"]
        else:
            claves = ["satellite"]
        res = d.group_by(claves).agg([
            pl.len().alias("n"),
            pl.col("frp").mean().alias("frp_medio"),
        ])
        return res.collect().height

    if op == "features":
        exprs = []
        if {"brightness", "bright_t31"}.issubset(set(cols)):
            exprs.append((pl.col("brightness") - pl.col("bright_t31")).alias("delta_t"))
        if "acq_date" in cols:
            exprs.append(pl.col("acq_date").dt.month().alias("mes"))
        if "acq_time" in cols:
            exprs.append((pl.col("acq_time") // 100).alias("hora"))
        return lf.with_columns(exprs).collect().height

    raise ValueError(op)


# ----------------------------- DASK (distribuido) --------------------------

def correr_dask(ruta, op, frac, n_workers):
    from dask.distributed import Client, LocalCluster
    import dask.dataframe as dd
    import pyarrow.parquet as pq

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1,
                           processes=True, dashboard_address=None,
                           silence_logs=50)
    client = Client(cluster)
    try:
        esquema = pq.ParquetFile(ruta).schema_arrow.names
        cols = _columnas_presentes(esquema)

        ddf = dd.read_parquet(ruta, columns=cols)
        if frac < 1.0:
            ddf = ddf.sample(frac=frac, random_state=42)

        if op == "carga":
            return int(ddf.shape[0].compute())

        if op == "filtro":
            if "confidence" in cols:
                baja = ddf["confidence"].astype(str).str.lower().isin(
                    VALORES_BAJA_CONFIANZA)
                res = ddf[~baja]
            else:
                res = ddf[ddf["frp"] > 0]
            return int(res.shape[0].compute())

        if op == "agregacion":
            d = ddf
            if "acq_date" in cols:
                d = d.assign(mes=dd.to_datetime(d["acq_date"]).dt.month)
                claves = ["satellite", "mes"] if "satellite" in cols else ["mes"]
            else:
                claves = ["satellite"]
            res = d.groupby(claves)["frp"].agg(["size", "mean"]).compute()
            return len(res)

        if op == "features":
            d = ddf
            if {"brightness", "bright_t31"}.issubset(set(cols)):
                d = d.assign(delta_t=d["brightness"] - d["bright_t31"])
            if "acq_date" in cols:
                d = d.assign(mes=dd.to_datetime(d["acq_date"]).dt.month)
            if "acq_time" in cols:
                d = d.assign(hora=d["acq_time"] // 100)
            return int(d.shape[0].compute())

        raise ValueError(op)
    finally:
        client.close()
        cluster.close()


# ----------------------------- cuDF (GPU, opcional) ------------------------

def correr_cudf(ruta, op, frac):
    import cudf

    df = cudf.read_parquet(ruta)
    cols = _columnas_presentes(df.columns)
    df = df[cols]
    if frac < 1.0:
        df = df.iloc[: int(len(df) * frac)]

    if op == "carga":
        return len(df)

    if op == "filtro":
        if "confidence" in cols:
            baja = df["confidence"].astype("str").str.lower().isin(
                VALORES_BAJA_CONFIANZA)
            res = df[~baja]
        else:
            res = df[df["frp"] > 0]
        return len(res)

    if op == "agregacion":
        d = df
        if "acq_date" in cols:
            d["mes"] = cudf.to_datetime(d["acq_date"]).dt.month
            claves = ["satellite", "mes"] if "satellite" in cols else ["mes"]
        else:
            claves = ["satellite"]
        res = d.groupby(claves).agg({"frp": ["count", "mean"]})
        return len(res)

    if op == "features":
        d = df
        if {"brightness", "bright_t31"}.issubset(set(cols)):
            d["delta_t"] = d["brightness"] - d["bright_t31"]
        if "acq_date" in cols:
            d["mes"] = cudf.to_datetime(d["acq_date"]).dt.month
        if "acq_time" in cols:
            d["hora"] = d["acq_time"] // 100
        return len(d)

    raise ValueError(op)


# ---------------------------------------------------------------------------
# 4) ORQUESTACION
# ---------------------------------------------------------------------------

DISPATCH = {
    "pandas": lambda r, o, f, t: correr_pandas(r, o, f),
    "polars": lambda r, o, f, t: correr_polars(r, o, f),
    "dask":   lambda r, o, f, t: correr_dask(r, o, f, t),
    "cudf":   lambda r, o, f, t: correr_cudf(r, o, f),
}

CAMPOS_CSV = [
    "backend", "operacion", "hilos", "escalabilidad", "frac", "repeticion",
    "tiempo_s", "filas_resultado", "throughput_filas_s", "throughput_mb_s",
    "memoria_pico_mb", "tam_archivo_mb", "nucleos_disponibles",
    "host", "python", "timestamp",
]


def ejecutar_una(op):
    ruta = ARGS.datos
    tam_mb = os.path.getsize(ruta) / (1024.0 * 1024.0)

    t0 = time.perf_counter()
    filas = DISPATCH[ARGS.backend](ruta, op, ARGS.frac, ARGS.threads)
    t1 = time.perf_counter()

    dur = t1 - t0
    mb_efectivos = tam_mb * ARGS.frac

    return {
        "backend": ARGS.backend,
        "operacion": op,
        "hilos": ARGS.threads if ARGS.backend != "pandas" else 1,
        "escalabilidad": ARGS.escalabilidad,
        "frac": ARGS.frac,
        "repeticion": ARGS.repeticion,
        "tiempo_s": round(dur, 6),
        "filas_resultado": filas,
        "throughput_filas_s": round(filas / dur, 2) if dur > 0 else 0,
        "throughput_mb_s": round(mb_efectivos / dur, 2) if dur > 0 else 0,
        "memoria_pico_mb": round(memoria_pico_mb(), 2),
        "tam_archivo_mb": round(tam_mb, 2),
        "nucleos_disponibles": nucleos_disponibles(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def guardar(filas_resultado, salida):
    nuevo = not os.path.exists(salida)
    with open(salida, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_CSV)
        if nuevo:
            w.writeheader()
        for fila in filas_resultado:
            w.writerow(fila)


def main():
    if not os.path.exists(ARGS.datos):
        print("ERROR: no existe el archivo de datos: %s" % ARGS.datos,
              file=sys.stderr)
        sys.exit(1)

    ops = (["carga", "filtro", "agregacion", "features"]
           if ARGS.op == "todas" else [ARGS.op])

    resultados = []
    for op in ops:
        try:
            r = ejecutar_una(op)
            resultados.append(r)
            print("[OK] %-7s %-11s hilos=%-3s frac=%-4s -> %8.3f s  %s filas"
                  % (r["backend"], r["operacion"], r["hilos"], r["frac"],
                     r["tiempo_s"], format(r["filas_resultado"], ",")))
        except Exception as exc:   # una operacion fallida no aborta el resto
            print("[FALLO] %s / %s: %s" % (ARGS.backend, op, exc),
                  file=sys.stderr)

    if resultados:
        guardar(resultados, ARGS.salida)
        print("Resultados agregados a: %s" % ARGS.salida)


if __name__ == "__main__":
    main()
