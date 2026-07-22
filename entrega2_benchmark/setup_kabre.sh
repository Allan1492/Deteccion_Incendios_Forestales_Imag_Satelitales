#!/usr/bin/env bash
# ===========================================================================
# setup_kabre.sh -- Monta el entorno de Python en el cluster Kabre
#
# Proyecto: Deteccion de Incendios Forestales con Imagenes Satelitales
# Curso   : TTCT0017 - Computacion Paralela y Distribuida (LEAD University)
#
# POR QUE MINIFORGE Y NO EL CONDA DEL SISTEMA
# -------------------------------------------
# El unico modulo conda de Kabre es `anaconda2/4.2.0`, que trae Python 2 y
# no sirve para Polars, Dask ni RAPIDS. Instalar un Miniforge propio en el
# directorio del usuario es la practica estandar en HPC y deja el entorno
# bajo nuestro control.
#
# Se instala en /work (100 GB de cuota) y NO en /home (solo 10 GB).
#
# USO (ejecutar en el NODO DE LOGIN, una sola vez):
#     bash setup_kabre.sh
# ===========================================================================

set -euo pipefail

ENTORNO="incendios"

# --- Elegir donde instalar --------------------------------------------------
# No todas las cuentas de Kabre tienen /work habilitado. Se prueba en orden
# de preferencia y se usa el primer directorio donde se pueda escribir:
#   /data  (100 GB) -> primera opcion: espacio de sobra para conda + RAPIDS
#   /work  (100 GB) -> si existe
#   /home  ( 10 GB) -> ultimo recurso; alcanza para CPU pero se queda corto
#                      si luego se instala RAPIDS
# Se puede forzar con:  BASE=/ruta bash setup_kabre.sh
if [[ -n "${BASE:-}" ]]; then
  CANDIDATOS=("$BASE")
else
  CANDIDATOS=("/data/$USER" "/work/$USER" "$HOME")
fi

BASE_ELEGIDA=""
for c in "${CANDIDATOS[@]}"; do
  if mkdir -p "$c" 2>/dev/null && [[ -w "$c" ]]; then
    BASE_ELEGIDA="$c"
    break
  fi
done

if [[ -z "$BASE_ELEGIDA" ]]; then
  echo "ERROR: no se encontro ningun directorio con permiso de escritura." >&2
  echo "Probados: ${CANDIDATOS[*]}" >&2
  echo "Ejecuta 'ls -ld /home/\$USER /work/\$USER /data/\$USER' y reintenta" >&2
  echo "forzando la ruta:  BASE=/ruta/valida bash setup_kabre.sh" >&2
  exit 1
fi

DESTINO="$BASE_ELEGIDA/miniforge3"

echo "==========================================================="
echo " Configuracion del entorno en Kabre"
echo " Usuario : $USER"
echo " Base    : $BASE_ELEGIDA"
echo " Destino : $DESTINO"
echo "==========================================================="

# Aviso si quedamos en /home, que tiene solo 10 GB de cuota
if [[ "$BASE_ELEGIDA" == "$HOME" ]]; then
  echo "AVISO: se instalara en /home (cuota de 10 GB). Es suficiente para"
  echo "       el entorno de CPU, pero puede quedarse corto al agregar"
  echo "       RAPIDS mas adelante."
  echo ""
fi

# --- 1. Instalar Miniforge si no existe ------------------------------------
if [[ -d "$DESTINO" ]]; then
  echo "[1/4] Miniforge ya existe en $DESTINO -- se omite la instalacion."
else
  echo "[1/4] Descargando Miniforge..."
  cd "$BASE_ELEGIDA"
  curl -fsSL -o miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  echo "      Instalando (silencioso)..."
  bash miniforge.sh -b -p "$DESTINO"
  rm -f miniforge.sh
  echo "      Miniforge instalado."
fi

# --- 2. Activar conda en esta sesion ---------------------------------------
echo "[2/4] Activando conda..."
source "$DESTINO/etc/profile.d/conda.sh"

# --- 3. Crear el entorno del proyecto --------------------------------------
if conda env list | grep -qE "^${ENTORNO}\s"; then
  echo "[3/4] El entorno '$ENTORNO' ya existe -- se omite la creacion."
else
  echo "[3/4] Creando el entorno '$ENTORNO' (puede tardar unos minutos)..."
  conda create -y -n "$ENTORNO" -c conda-forge \
    python=3.11 \
    polars \
    pandas \
    pyarrow \
    "dask" \
    distributed \
    matplotlib \
    tabulate \
    jupyterlab
fi

conda activate "$ENTORNO"

# --- 4. Verificacion --------------------------------------------------------
echo "[4/4] Verificando la instalacion..."
python - <<'PY'
import importlib, sys
print("  Python:", sys.version.split()[0])
for m in ["polars", "pandas", "pyarrow", "dask", "distributed",
          "matplotlib", "tabulate"]:
    try:
        mod = importlib.import_module(m)
        print("  %-12s %s" % (m, getattr(mod, "__version__", "ok")))
    except Exception as e:
        print("  %-12s FALLO: %s" % (m, e))
PY

# --- Directorio para los datos ---------------------------------------------
mkdir -p "/data/$USER"

cat <<EOF

===========================================================
 Entorno listo.

 Para activarlo en futuras sesiones:
     source $DESTINO/etc/profile.d/conda.sh
     conda activate $ENTORNO

 Coloca el parquet de FIRMS en:
     /data/$USER/incendios_global_consolidado.parquet

 Luego verifica que este integro y lanza el benchmark:
     python3 verificar_parquet.py /data/$USER/incendios_global_consolidado.parquet
     sbatch submit_kabre.slurm
===========================================================

NOTA SOBRE GPU (para la Etapa 4, no ahora):
 Para cuDF/cuML hara falta un entorno aparte con RAPIDS:
     conda create -y -n incendios-gpu -c rapidsai -c conda-forge -c nvidia \\
         cudf cuml python=3.11 cuda-version=12.4
 Y enviar el job a la particion nukwa-l40s con --gres=gpu:1
EOF
