#!/usr/bin/env bash
# ===========================================================================
# run_all.sh -- Driver del benchmark de rendimiento (Etapa 3)
#
# Proyecto: Deteccion de Incendios Forestales con Imagenes Satelitales
# Curso   : TTCT0017 - Computacion Paralela y Distribuida (LEAD University)
#
# Recorre todas las configuraciones del experimento y acumula los resultados
# en un unico CSV. Cada configuracion corre en un PROCESO NUEVO, porque el
# numero de hilos de Polars solo puede fijarse antes de importar la libreria.
#
# USO:
#   ./run_all.sh /ruta/a/incendios_global_consolidado.parquet [salida.csv]
# ===========================================================================

set -u

DATOS="${1:-}"
SALIDA="${2:-resultados_benchmark.csv}"
REPETICIONES="${REPETICIONES:-3}"     # repeticiones por configuracion
PY="${PY:-python3}"

if [[ -z "$DATOS" ]]; then
  echo "Uso: $0 <ruta_parquet> [salida.csv]" >&2
  exit 1
fi
if [[ ! -f "$DATOS" ]]; then
  echo "ERROR: no existe el archivo: $DATOS" >&2
  exit 1
fi

# --- Niveles de paralelismo -------------------------------------------------
# Se ajustan automaticamente a los nucleos reservados por SLURM. Solo se
# incluyen potencias de 2 que quepan en los nucleos disponibles.
NUCLEOS="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 4)}"
NIVELES=""
for n in 1 2 4 8 16 32; do
  if [[ "$n" -le "$NUCLEOS" ]]; then NIVELES="$NIVELES $n"; fi
done
# Garantiza incluir el total exacto de nucleos si no es potencia de 2
if ! echo "$NIVELES" | grep -qw "$NUCLEOS"; then NIVELES="$NIVELES $NUCLEOS"; fi

echo "==========================================================="
echo " Benchmark de rendimiento -- Etapa 3"
echo " Datos      : $DATOS"
echo " Salida     : $SALIDA"
echo " Nucleos    : $NUCLEOS"
echo " Niveles    :$NIVELES"
echo " Repeticion : $REPETICIONES"
echo "==========================================================="

# ---------------------------------------------------------------------------
# 1) LINEA BASE SECUENCIAL (pandas, 1 hilo)
#    Es el T_1 contra el que se calcula el speedup de todo lo demas.
# ---------------------------------------------------------------------------
echo ""
echo "--- [1/4] Linea base secuencial (pandas, 1 hilo) ---"
for r in $(seq 1 "$REPETICIONES"); do
  $PY bench_run.py --backend pandas --threads 1 --op todas \
      --datos "$DATOS" --salida "$SALIDA" --repeticion "$r" \
      --escalabilidad fuerte
done

# ---------------------------------------------------------------------------
# 2) ESCALABILIDAD FUERTE -- Polars
#    Mismo tamano de problema, numero creciente de hilos.
# ---------------------------------------------------------------------------
echo ""
echo "--- [2/4] Escalabilidad fuerte: Polars ---"
for t in $NIVELES; do
  for r in $(seq 1 "$REPETICIONES"); do
    $PY bench_run.py --backend polars --threads "$t" --op todas \
        --datos "$DATOS" --salida "$SALIDA" --repeticion "$r" \
        --escalabilidad fuerte
  done
done

# ---------------------------------------------------------------------------
# 3) ESCALABILIDAD FUERTE -- Dask
# ---------------------------------------------------------------------------
echo ""
echo "--- [3/4] Escalabilidad fuerte: Dask ---"
for t in $NIVELES; do
  for r in $(seq 1 "$REPETICIONES"); do
    $PY bench_run.py --backend dask --threads "$t" --op todas \
        --datos "$DATOS" --salida "$SALIDA" --repeticion "$r" \
        --escalabilidad fuerte
  done
done

# ---------------------------------------------------------------------------
# 4) ESCALABILIDAD DEBIL -- Polars
#    El tamano del problema crece en proporcion al numero de hilos, de modo
#    que la carga POR HILO se mantiene constante. Si el sistema escalara de
#    forma ideal, el tiempo se mantendria plano.
# ---------------------------------------------------------------------------
echo ""
echo "--- [4/4] Escalabilidad debil: Polars ---"
MAX=$(echo "$NIVELES" | tr ' ' '\n' | sort -n | tail -1)
for t in $NIVELES; do
  FRAC=$($PY -c "print(round($t/$MAX, 4))")
  for r in $(seq 1 "$REPETICIONES"); do
    $PY bench_run.py --backend polars --threads "$t" --op todas \
        --datos "$DATOS" --salida "$SALIDA" --repeticion "$r" \
        --frac "$FRAC" --escalabilidad debil
  done
done

# ---------------------------------------------------------------------------
# 5) GPU (opcional) -- solo si cuDF esta disponible
# ---------------------------------------------------------------------------
if $PY -c "import cudf" 2>/dev/null; then
  echo ""
  echo "--- [extra] GPU: cuDF ---"
  for r in $(seq 1 "$REPETICIONES"); do
    $PY bench_run.py --backend cudf --threads 1 --op todas \
        --datos "$DATOS" --salida "$SALIDA" --repeticion "$r" \
        --escalabilidad fuerte
  done
else
  echo ""
  echo "--- [extra] cuDF no disponible: se omite la parte de GPU ---"
fi

echo ""
echo "==========================================================="
echo " Benchmark completo. Resultados en: $SALIDA"
echo " Siguiente paso:  python3 analizar_resultados.py $SALIDA"
echo "==========================================================="
