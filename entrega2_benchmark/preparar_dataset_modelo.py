#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preparar_dataset_modelo.py -- Etapa 2 (artefacto): genera la tabla limpia y
lista para el modelado a partir del Parquet crudo de FIRMS.

Proyecto: Detección de Incendios Forestales con Imágenes Satelitales
Curso   : TTCT0017 - Computación Paralela y Distribuida (LEAD University)

QUÉ PRODUCE
-----------
Un Parquet con las variables de entrada (features) y la variable objetivo del
problema de reducción de falsas alarmas (Camino A: la baja confianza opera
como proxy de falsa alarma).

Ingeniería de variables:
  delta_t         = brightness - bright_t31   (contraste térmico foco-entorno)
  mes             = mes de acq_date            (estacionalidad)
  hora            = acq_time // 100            (hora UTC 0-23)
  es_noche        = 1 si daynight == 'N'       (contexto)
  es_falsa_alarma = 1 si confidence == 'l'     (VARIABLE OBJETIVO / target)

Predictoras: delta_t, brightness, bright_t31, frp, scan, track, mes, hora,
             es_noche, latitude, longitude.
Se descartan: version (constante), instrument, y las columnas ya derivadas
(acq_date, acq_time, daynight) y confidence (solo se usa para el target, NO
como predictora, para no filtrar información de la etiqueta).

MEMORIA
-------
Usa evaluación perezosa y 'sink_parquet' (streaming): procesa por bloques y
escribe directo a disco, sin materializar los 52 M de filas en RAM. Corre
igual en un portátil que en Kabré.

USO
---
    python3 preparar_dataset_modelo.py ENTRADA.parquet SALIDA.parquet
"""
import sys
import time
import polars as pl

VALORES_BAJA_CONFIANZA = ["l", "low"]   # proxy de falsa alarma (VIIRS / MODIS)


def main():
    if len(sys.argv) < 3:
        print("Uso: python3 preparar_dataset_modelo.py ENTRADA.parquet SALIDA.parquet",
              file=sys.stderr)
        sys.exit(1)

    entrada, salida = sys.argv[1], sys.argv[2]
    print("=" * 62)
    print("PREPARACIÓN DEL DATASET PARA EL MODELO (Etapa 2)")
    print("=" * 62)
    print("Entrada:", entrada)
    print("Salida :", salida)

    t0 = time.perf_counter()
    lf = pl.scan_parquet(entrada)
    cols = lf.collect_schema().names()

    # --- Construcción perezosa de las variables ---------------------------
    lf = lf.with_columns([
        (pl.col("brightness") - pl.col("bright_t31")).alias("delta_t"),
        pl.col("acq_date").dt.month().alias("mes"),
        (pl.col("acq_time") // 100).alias("hora"),
        (pl.col("daynight") == "N").cast(pl.Int8).alias("es_noche"),
        pl.col("confidence").cast(pl.Utf8).str.to_lowercase()
          .is_in(VALORES_BAJA_CONFIANZA).cast(pl.Int8).alias("es_falsa_alarma"),
    ])

    # --- Limpieza (filtrado) ----------------------------------------------
    # Se descartan registros con FRP negativo (mediciones inválidas) y
    # coordenadas fuera de rango físico.
    lf = lf.filter(
        (pl.col("frp") >= 0)
        & (pl.col("latitude").is_between(-90, 90))
        & (pl.col("longitude").is_between(-180, 180))
    )

    # --- Selección final de columnas --------------------------------------
    columnas_modelo = [
        # predictoras físicas
        "delta_t", "brightness", "bright_t31", "frp", "scan", "track",
        # contextuales
        "mes", "hora", "es_noche", "latitude", "longitude",
        # objetivo
        "es_falsa_alarma",
    ]
    columnas_modelo = [c for c in columnas_modelo if c in lf.collect_schema().names()]
    lf = lf.select(columnas_modelo)

    # --- Escritura en streaming (sin cargar todo en memoria) --------------
    lf.sink_parquet(salida, compression="snappy")

    dur = time.perf_counter() - t0

    # --- Verificación y resumen -------------------------------------------
    res = pl.scan_parquet(salida)
    n = res.select(pl.len()).collect().item()
    balance = (res.group_by("es_falsa_alarma").agg(pl.len().alias("n"))
                  .sort("es_falsa_alarma").collect())

    print("")
    print("Filas escritas :", format(n, ","))
    print("Columnas       :", len(columnas_modelo))
    print("Tiempo         : {:.1f}s".format(dur))
    print("")
    print("Columnas del dataset de modelado:")
    print("  ", ", ".join(columnas_modelo))
    print("")
    print("Balance de la variable objetivo (es_falsa_alarma):")
    tot = balance["n"].sum()
    etiquetas = {0: "0 = valida (nominal/alta)", 1: "1 = probable falsa alarma (baja)"}
    for row in balance.iter_rows(named=True):
        print("  {:<34} {:>12,}  ({:5.2f}%)".format(
            etiquetas.get(row["es_falsa_alarma"], str(row["es_falsa_alarma"])),
            row["n"], 100 * row["n"] / tot))
    print("=" * 62)
    print("Listo. Este Parquet es la entrada para el modelo tabular (cuML).")


if __name__ == "__main__":
    main()
