#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_parquet.py -- Diagnostica la integridad del archivo Parquet de FIRMS.

MOTIVO
------
Durante la Etapa 3 el EDA fallo con el error:

    parquet: File out of specification: The file must end with PAR1

Un archivo Parquet valido empieza y termina con los 4 bytes magicos 'PAR1'.
Si el final no coincide, el archivo esta TRUNCADO (descarga o copia
incompleta) y no puede leerse. Este script lo detecta en segundos, antes de
gastar tiempo de computo en el cluster.

USO
---
    python3 verificar_parquet.py /ruta/incendios_global_consolidado.parquet

CODIGOS DE SALIDA
-----------------
    0 = archivo valido
    1 = archivo invalido o ilegible
"""

import os
import sys

MAGIC = b"PAR1"


def humano(n_bytes):
    for unidad in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024.0:
            return "%.2f %s" % (n_bytes, unidad)
        n_bytes /= 1024.0
    return "%.2f PB" % n_bytes


def verificar(ruta):
    print("=" * 60)
    print("VERIFICACION DE INTEGRIDAD DEL PARQUET")
    print("=" * 60)
    print("Archivo: %s" % ruta)

    # --- 1. Existencia y tamano --------------------------------------------
    if not os.path.exists(ruta):
        print("[FALLO] El archivo no existe.")
        return False

    tam = os.path.getsize(ruta)
    print("Tamano : %s (%s bytes)" % (humano(tam), format(tam, ",")))

    if tam < 12:
        print("[FALLO] Archivo demasiado pequeno para ser un Parquet valido.")
        return False

    # --- 2. Bytes magicos al inicio y al final ------------------------------
    with open(ruta, "rb") as fh:
        inicio = fh.read(4)
        fh.seek(-4, os.SEEK_END)
        final = fh.read(4)

    ok_inicio = inicio == MAGIC
    ok_final = final == MAGIC

    print("Magic inicial 'PAR1': %s (leido: %r)"
          % ("OK" if ok_inicio else "FALLO", inicio))
    print("Magic final   'PAR1': %s (leido: %r)"
          % ("OK" if ok_final else "FALLO", final))

    if not ok_final:
        print("")
        print(">> DIAGNOSTICO: el archivo esta TRUNCADO.")
        print(">> Causa tipica: la descarga o la copia se interrumpio, o se")
        print(">> subio parcialmente (por ejemplo, a Google Drive/Colab).")
        print(">> SOLUCION: volver a generar o copiar el archivo completo y")
        print(">> comparar el tamano en bytes contra el origen.")
        return False

    if not ok_inicio:
        print("")
        print(">> DIAGNOSTICO: el archivo no es un Parquet (cabecera invalida).")
        return False

    # --- 3. Lectura de metadatos (sin cargar los datos) --------------------
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(ruta)
        md = pf.metadata
        print("")
        print("Metadatos leidos correctamente:")
        print("  Filas totales    : %s" % format(md.num_rows, ","))
        print("  Columnas         : %d" % md.num_columns)
        print("  Grupos de filas  : %d" % md.num_row_groups)
        print("  Creado por       : %s" % md.created_by)
        print("")
        print("  Esquema:")
        for nombre, tipo in zip(pf.schema_arrow.names, pf.schema_arrow.types):
            print("    %-20s %s" % (nombre, tipo))

        # Chequeo especifico del proyecto: bright_t31 es indispensable para
        # calcular delta_t, la variable clave del problema de falsas alarmas.
        if "bright_t31" not in pf.schema_arrow.names:
            print("")
            print("  [AVISO] Falta la columna 'bright_t31'. Sin ella no se")
            print("          puede calcular delta_t = brightness - bright_t31,")
            print("          que es la variable central del problema definido.")
    except Exception as exc:
        print("")
        print("[FALLO] No se pudieron leer los metadatos: %s" % exc)
        return False

    # --- 4. Lectura de prueba del primer grupo de filas --------------------
    try:
        primera = pf.read_row_group(0)
        print("")
        print("Lectura de prueba del primer grupo: OK (%s filas)"
              % format(primera.num_rows, ","))
    except Exception as exc:
        print("")
        print("[FALLO] El archivo tiene metadatos validos pero no se pueden")
        print("        leer los datos: %s" % exc)
        return False

    print("")
    print("=" * 60)
    print("RESULTADO: el archivo es VALIDO y se puede usar.")
    print("=" * 60)
    return True


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 verificar_parquet.py <ruta_parquet>",
              file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if verificar(sys.argv[1]) else 1)


if __name__ == "__main__":
    main()
