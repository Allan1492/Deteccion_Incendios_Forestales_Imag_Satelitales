#!/bin/bash
# ===========================================================================
# submit_kabre.slurm -- Job de SLURM para el benchmark en el cluster Kabre
#
# Proyecto: Deteccion de Incendios Forestales con Imagenes Satelitales
# Curso   : TTCT0017 - Computacion Paralela y Distribuida (LEAD University)
# Cluster : Kabre - CeNAT (https://kabre.cenat.ac.cr/)
#
# ENVIAR EL TRABAJO:
#     sbatch submit_kabre.slurm
#
# MONITOREAR:
#     squeue -u $USER          # ver estado del job
#     tail -f bench_<jobid>.out  # seguir la salida en vivo
#     scancel <jobid>          # cancelar
#
# IMPORTANTE: ajusta --partition al nombre real de la particion de Kabre a la
# que tengas acceso (consulta con `sinfo`). Los nombres aqui son un punto de
# partida y pueden variar segun la cuenta del curso.
# ===========================================================================

#SBATCH --job-name=bench_incendios
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err
#SBATCH --nodes=1                  # benchmark de memoria compartida: 1 nodo
#SBATCH --ntasks=1
#SBATCH --partition=kura           # 11 nodos, 40 nucleos, 257 GB RAM
#SBATCH --cpus-per-task=40         # nucleos reservados (define los niveles)
#SBATCH --time=02:00:00            # tiempo maximo (kura permite hasta 3 dias)

# ALTERNATIVAS DE PARTICION (segun `sinfo`):
#   kura        40 nucleos, 257 GB, 3 dias   <- la usada aqui (CPU)
#   kura-debug  40 nucleos, 257 GB, 8 horas  <- para pruebas rapidas, menos cola
#   dribe       36+ nucleos, 514 GB, 3 dias  <- si hace falta mucha memoria
#   nukwa-l40s  20 nucleos, 256 GB + GPU L40S  <- para cuDF/cuML (Etapa 4)
#
# Para la corrida con GPU, cambia a:
#   #SBATCH --partition=nukwa-l40s
#   #SBATCH --cpus-per-task=20
#   #SBATCH --gres=gpu:1

echo "==========================================================="
echo " Job ID        : ${SLURM_JOB_ID:-N/A}"
echo " Nodo          : $(hostname)"
echo " Particion     : ${SLURM_JOB_PARTITION:-N/A}"
echo " CPUs por tarea: ${SLURM_CPUS_PER_TASK:-N/A}"
echo " Inicio        : $(date)"
echo "==========================================================="

# --- Entorno ---------------------------------------------------------------
# En Kabre el unico conda del sistema es anaconda2 (Python 2), que no sirve.
# Por eso se usa un Miniforge instalado por el usuario (ver setup_kabre.sh).
# No todas las cuentas tienen /work, asi que se busca en varias rutas.
CONDA_SH=""
for base in "/data/$USER" "/work/$USER" "$HOME"; do
  if [[ -f "$base/miniforge3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$base/miniforge3/etc/profile.d/conda.sh"
    break
  fi
done

if [[ -z "$CONDA_SH" ]]; then
  echo "ERROR: no se encontro Miniforge. Ejecuta primero: bash setup_kabre.sh"
  exit 1
fi

echo "Usando conda de: $CONDA_SH"
source "$CONDA_SH"
conda activate incendios

# Evita que las librerias de algebra lineal generen hilos por su cuenta y
# contaminen la medicion de paralelismo.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Evita que las librerias de algebra lineal generen hilos por su cuenta y
# contaminen la medicion de paralelismo.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Rutas -----------------------------------------------------------------
# El parquet va en /data (100 GB de cuota), NO en /home (solo 10 GB).
DATOS="${DATOS:-/data/$USER/incendios_global_consolidado.parquet}"
SALIDA="resultados_benchmark_${SLURM_JOB_ID:-local}.csv"

if [[ ! -f "$DATOS" ]]; then
  echo "ERROR: no se encontro el dataset en: $DATOS"
  echo "Exporta la ruta correcta:  DATOS=/ruta/al.parquet sbatch submit_kabre.slurm"
  exit 1
fi

# --- Verificacion de integridad del parquet --------------------------------
# El fallo detectado previamente ('The file must end with PAR1') indica un
# archivo truncado. Esta comprobacion evita gastar horas de computo en balde.
echo ""
echo "Verificando integridad del parquet..."
python3 verificar_parquet.py "$DATOS" || {
  echo "ERROR: el parquet no paso la verificacion. Abortando."
  exit 1
}

# --- Ejecucion -------------------------------------------------------------
echo ""
chmod +x run_all.sh
REPETICIONES=3 ./run_all.sh "$DATOS" "$SALIDA"

# --- Analisis --------------------------------------------------------------
echo ""
echo "Generando tablas y graficos..."
python3 analizar_resultados.py "$SALIDA" --figuras figuras_${SLURM_JOB_ID:-local}

echo ""
echo "==========================================================="
echo " Fin: $(date)"
echo " Resultados: $SALIDA"
echo "==========================================================="
