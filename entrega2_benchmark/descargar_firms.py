#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
descargar_firms.py -- Descarga datos de NASA FIRMS y los consolida en Parquet.

Proyecto: Deteccion de Incendios Forestales con Imagenes Satelitales
Curso   : TTCT0017 - Computacion Paralela y Distribuida (LEAD University)
Etapa 1 : Adquisicion y gestion de datos

POR QUE ESTE SCRIPT
-------------------
El parquet que circulaba por WhatsApp llego truncado ("The file must end with
PAR1") y el EDA termino corriendo sobre datos sinteticos. Descargar los datos
directamente desde la fuente, en la maquina donde se van a usar, elimina ese
problema y ademas deja la adquisicion documentada y REPRODUCIBLE, que es lo
que pide la Etapa 1.

POR QUE MODIS Y NO VIIRS
------------------------
El esquema del proyecto usa `brightness`, `bright_t31`, `scan` y `track`, que
son columnas de MODIS. VIIRS entrega `bright_ti4` / `bright_ti5` y NO incluye
`bright_t31`, que es indispensable para calcular:

    delta_t = brightness - bright_t31

la variable central del problema de reduccion de falsas alarmas.

COMO OBTENER UNA MAP_KEY (gratis, inmediata)
--------------------------------------------
    https://firms.modaps.eosdis.nasa.gov/api/area/
Llene el formulario con su correo y recibira la clave al instante.

USO
---
    # Global, dos anios de datos MODIS
    python3 descargar_firms.py --map-key TU_CLAVE \
        --inicio 2023-01-01 --fin 2024-12-31 \
        --salida /data/$USER/incendios_global_consolidado.parquet

    # Solo Canada (mas liviano)
    python3 descargar_firms.py --map-key TU_CLAVE \
        --inicio 2023-01-01 --fin 2024-12-31 --area canada \
        --salida /data/$USER/incendios_canada.parquet

NOTA SOBRE LIMITES
------------------
La API de FIRMS limita el numero de transacciones por intervalo (tipicamente
unos cientos cada 10 minutos). El script respeta una pausa entre peticiones y
reintenta ante errores temporales. Una descarga global de dos anios puede
tardar bastante: conviene lanzarla con `nohup` o dentro de un job de SLURM.
"""

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

# Fuentes disponibles en la API de FIRMS.
#   *_SP  = Standard Processing (archivo historico, calidad definitiva)
#   *_NRT = Near Real Time (ultimos ~2 meses)
FUENTES = {
    "modis-historico": "MODIS_SP",
    "modis-reciente": "MODIS_NRT",
    "viirs-snpp": "VIIRS_SNPP_SP",
    "viirs-noaa20": "VIIRS_NOAA20_NRT",
}

# Areas predefinidas: "oeste,sur,este,norte"
AREAS = {
    "mundo": "world",
    "canada": "-141,41.7,-52.6,83.1",
    "costa-rica": "-86,8,-82.5,11.3",
    "norteamerica": "-170,14,-50,84",
}

URL_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_DIAS_POR_PETICION = 10          # limite que impone la API
PAUSA_SEGUNDOS = 2.0                # cortesia entre peticiones
MAX_REINTENTOS = 4


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------

def construir_url(map_key, fuente, area, dias, fecha_inicio):
    return "%s/%s/%s/%s/%d/%s" % (
        URL_BASE, map_key, fuente, area, dias, fecha_inicio.isoformat())


def descargar_ventana(map_key, fuente, area, dias, fecha_inicio):
    """Descarga una ventana de hasta 10 dias. Devuelve el texto CSV o None."""
    import urllib.error
    import urllib.request

    url = construir_url(map_key, fuente, area, dias, fecha_inicio)

    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                texto = resp.read().decode("utf-8", errors="replace")

            # La API devuelve texto plano tambien para los errores
            minusc = texto[:300].lower()
            if "invalid" in minusc and "key" in minusc:
                print("\nERROR: la MAP_KEY parece invalida.", file=sys.stderr)
                print("Obtenga una en https://firms.modaps.eosdis.nasa.gov/api/area/",
                      file=sys.stderr)
                sys.exit(1)
            if "transaction limit" in minusc or "rate limit" in minusc:
                espera = 60 * intento
                print("  limite de peticiones alcanzado; esperando %ds..." % espera)
                time.sleep(espera)
                continue
            if not texto.strip() or texto.count("\n") < 1:
                return ""      # ventana sin detecciones: valido
            return texto

        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                espera = 15 * intento
                print("  HTTP %d; reintento en %ds..." % (e.code, espera))
                time.sleep(espera)
                continue
            print("\nERROR HTTP %d en %s" % (e.code, fecha_inicio), file=sys.stderr)
            return None
        except Exception as exc:
            espera = 10 * intento
            print("  fallo (%s); reintento en %ds..." % (exc, espera))
            time.sleep(espera)

    print("\nAVISO: no se pudo descargar la ventana del %s" % fecha_inicio,
          file=sys.stderr)
    return None


def rango_de_ventanas(inicio, fin):
    """Divide el periodo en ventanas de hasta 10 dias."""
    ventanas = []
    actual = inicio
    while actual <= fin:
        dias = min(MAX_DIAS_POR_PETICION, (fin - actual).days + 1)
        ventanas.append((actual, dias))
        actual += timedelta(days=dias)
    return ventanas


# ---------------------------------------------------------------------------
# Consolidacion
# ---------------------------------------------------------------------------

def consolidar(carpeta_csv, salida):
    """Une todos los CSV descargados en un unico Parquet con Polars."""
    import polars as pl

    archivos = sorted(
        os.path.join(carpeta_csv, f)
        for f in os.listdir(carpeta_csv) if f.endswith(".csv")
    )
    archivos = [a for a in archivos if os.path.getsize(a) > 0]

    if not archivos:
        print("ERROR: no hay CSV que consolidar.", file=sys.stderr)
        sys.exit(1)

    print("")
    print("Consolidando %d archivos CSV..." % len(archivos))

    lf = pl.scan_csv(archivos, try_parse_dates=True, infer_schema_length=10000)
    df = lf.collect(engine="streaming")

    # Normaliza tipos que la API entrega como texto
    casts = []
    if "acq_time" in df.columns and df["acq_time"].dtype == pl.Utf8:
        casts.append(pl.col("acq_time").cast(pl.Int64, strict=False))
    if casts:
        df = df.with_columns(casts)

    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    df.write_parquet(salida, compression="snappy")

    tam_mb = os.path.getsize(salida) / (1024.0 * 1024.0)

    print("")
    print("=" * 62)
    print("PARQUET GENERADO")
    print("=" * 62)
    print("  Ruta    : %s" % salida)
    print("  Filas   : %s" % format(df.height, ","))
    print("  Columnas: %d" % df.width)
    print("  Tamano  : %.2f MB" % tam_mb)
    print("")
    print("  Esquema:")
    for nombre, tipo in df.schema.items():
        print("    %-16s %s" % (nombre, tipo))

    # Verificacion especifica del proyecto
    print("")
    if {"brightness", "bright_t31"}.issubset(set(df.columns)):
        print("  [OK] Estan 'brightness' y 'bright_t31': se puede calcular delta_t.")
    else:
        print("  [AVISO] Falta 'brightness' o 'bright_t31'. Sin ellas no se")
        print("          puede calcular delta_t. Verifique que uso una fuente MODIS.")
    print("=" * 62)
    return df.height


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Descarga NASA FIRMS y consolida en Parquet.")
    ap.add_argument("--map-key", required=True,
                    help="MAP_KEY de FIRMS (gratis en firms.modaps.eosdis.nasa.gov/api/area/)")
    ap.add_argument("--inicio", required=True, help="Fecha inicial YYYY-MM-DD")
    ap.add_argument("--fin", required=True, help="Fecha final YYYY-MM-DD")
    ap.add_argument("--fuente", default="modis-historico", choices=list(FUENTES),
                    help="Sensor y tipo de procesamiento (por defecto MODIS historico)")
    ap.add_argument("--area", default="mundo",
                    help="Area: %s, o bien 'oeste,sur,este,norte'"
                         % ", ".join(AREAS))
    ap.add_argument("--salida", required=True, help="Ruta del .parquet de salida")
    ap.add_argument("--temp", default=None,
                    help="Carpeta para los CSV intermedios (por defecto junto a la salida)")
    ap.add_argument("--conservar-csv", action="store_true",
                    help="No borrar los CSV intermedios al terminar")
    args = ap.parse_args()

    try:
        inicio = datetime.strptime(args.inicio, "%Y-%m-%d").date()
        fin = datetime.strptime(args.fin, "%Y-%m-%d").date()
    except ValueError:
        print("ERROR: las fechas deben tener formato YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    if inicio > fin:
        print("ERROR: la fecha inicial es posterior a la final.", file=sys.stderr)
        sys.exit(1)

    fuente = FUENTES[args.fuente]
    area = AREAS.get(args.area, args.area)

    temp = args.temp or (os.path.splitext(args.salida)[0] + "_csv")
    os.makedirs(temp, exist_ok=True)

    ventanas = rango_de_ventanas(inicio, fin)

    print("=" * 62)
    print("DESCARGA DE NASA FIRMS")
    print("=" * 62)
    print("  Fuente  : %s (%s)" % (fuente, args.fuente))
    print("  Area    : %s" % area)
    print("  Periodo : %s a %s" % (inicio, fin))
    print("  Ventanas: %d (de hasta %d dias)" % (len(ventanas), MAX_DIAS_POR_PETICION))
    print("  Temp    : %s" % temp)
    print("=" * 62)
    print("")

    descargadas = 0
    omitidas = 0
    fallidas = 0
    t0 = time.perf_counter()

    for i, (fecha, dias) in enumerate(ventanas, 1):
        destino = os.path.join(temp, "firms_%s_%02dd.csv" % (fecha.isoformat(), dias))

        if os.path.exists(destino) and os.path.getsize(destino) > 0:
            omitidas += 1
            continue      # permite reanudar una descarga interrumpida

        texto = descargar_ventana(args.map_key, fuente, area, dias, fecha)

        if texto is None:
            fallidas += 1
        else:
            with open(destino, "w", encoding="utf-8") as fh:
                fh.write(texto)
            descargadas += 1

        transcurrido = time.perf_counter() - t0
        pct = 100.0 * i / len(ventanas)
        sys.stdout.write("\r  [%3d/%3d] %5.1f%%  ok=%d omitidas=%d fallidas=%d  %.0fs"
                         % (i, len(ventanas), pct, descargadas, omitidas,
                            fallidas, transcurrido))
        sys.stdout.flush()

        time.sleep(PAUSA_SEGUNDOS)

    print("")
    print("")
    print("Descarga terminada en %.1f s" % (time.perf_counter() - t0))
    print("  nuevas=%d  ya existentes=%d  fallidas=%d"
          % (descargadas, omitidas, fallidas))

    if fallidas:
        print("")
        print("AVISO: hubo %d ventanas fallidas. Puede volver a ejecutar el" % fallidas)
        print("       mismo comando: las ya descargadas se omiten y solo se")
        print("       reintentaran las que faltan.")

    filas = consolidar(temp, args.salida)

    if not args.conservar_csv:
        print("")
        print("Los CSV intermedios quedan en %s" % temp)
        print("Puede borrarlos con:  rm -rf %s" % temp)

    print("")
    print("Siguiente paso:")
    print("  python3 verificar_parquet.py %s" % args.salida)
    return filas


if __name__ == "__main__":
    main()
