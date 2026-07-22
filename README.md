# Detección de Incendios Forestales con Imágenes Satelitales

Proyecto del curso **Computación Paralela y Distribuida (TTCT0017)** — LEAD University
Profesor: Johansell Villalobos Cubillo

## Descripción

Sistema para la detección y el análisis de incendios forestales a partir de datos
satelitales masivos, con énfasis en el procesamiento **paralelo y distribuido**.
El objetivo específico es **reducir la tasa de falsas alarmas** en las detecciones
de puntos de calor, combinando variables físicas y contextuales.

El proyecto se ejecuta sobre el clúster de alto rendimiento **Kabré** (CeNAT).

## Equipo

| Integrante | Responsabilidad principal |
|---|---|
| Esteban Gutiérrez Saborío | Adquisición y gestión de datos; administración del repositorio |
| Allan Arnulfo Montes Monge | Preprocesamiento paralelo/distribuido (Dask, Polars, cuDF) |
| Maria Moroney Sole | Análisis exploratorio y estadístico (EDA); informe |
| Marlon Mora Delgado | Modelado ML/IA (RAPIDS cuML + CNN PyTorch); presentación |
| Robson Sthiffen Calvo Ortega | Postprocesamiento, análisis de rendimiento y visualización |

## Datos

- **NASA FIRMS** — puntos de calor VIIRS (Suomi-NPP, NOAA-20, NOAA-21),
  30 jun 2025 – 30 jun 2026, **52 508 485 registros**, 1.24 GB en Parquet.
  Fuente: https://firms.modaps.eosdis.nasa.gov/
- **Kaggle Wildfire Prediction Dataset** — imágenes satelitales (Etapa 4, pendiente).
  https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset

> Los archivos de datos NO se versionan (ver `.gitignore`). Para reconstruir el
> Parquet desde la fuente, usar `entrega2_benchmark/descargar_firms.py`.

## Estructura del repositorio

```
.
├── Notebook/                 Notebook de preprocesamiento (Etapas 1-2)
├── entrega2_benchmark/       Benchmark de rendimiento (secuencial vs. paralelo)
│   ├── descargar_firms.py    Descarga y consolidación de datos FIRMS
│   ├── verificar_parquet.py  Verificación de integridad del Parquet
│   ├── bench_run.py          Ejecuta una configuración del benchmark
│   ├── run_all.sh            Driver: recorre todas las configuraciones
│   ├── analizar_resultados.py  Métricas (speedup, eficiencia) y gráficos
│   ├── setup_kabre.sh        Monta el entorno en Kabré
│   └── submit_kabre.slurm    Job de SLURM
├── eda/                      Análisis exploratorio
│   └── eda_real.py           EDA sobre los datos reales (streaming)
├── resultados/               CSV de resultados y figuras generadas
└── informes/                 Informes IEEE (LaTeX/PDF)
```

## Requisitos

- Python 3.11 (Miniforge/Conda recomendado)
- Ver dependencias en `pyproject.toml`

Instalación rápida del entorno de CPU:

```bash
conda create -n incendios -c conda-forge python=3.11 polars pandas pyarrow \
    dask distributed matplotlib plotly tabulate jupyterlab
conda activate incendios
```

En el clúster Kabré, usar `entrega2_benchmark/setup_kabre.sh`.

## Uso

```bash
# 1. Obtener/verificar los datos
python entrega2_benchmark/verificar_parquet.py /ruta/incendios_global_consolidado.parquet

# 2. Análisis exploratorio
python eda/eda_real.py /ruta/incendios_global_consolidado.parquet eda_salida

# 3. Benchmark de rendimiento (local o en Kabré vía sbatch)
cd entrega2_benchmark
./run_all.sh /ruta/incendios_global_consolidado.parquet resultados.csv
python analizar_resultados.py resultados.csv --figuras figuras/
```

## Estado del proyecto

| Etapa | Estado |
|---|---|
| 1. Adquisición y gestión de datos | Completada |
| 2. Preprocesamiento paralelo/distribuido | Completada (artefacto de dataset limpio en curso) |
| 3. Análisis exploratorio y estadístico | Completada |
| 4. Modelado ML/IA (cuML + CNN) | Pendiente (Entrega 3) |
| 5. Postprocesamiento y análisis de resultados | Parcial (benchmark de rendimiento hecho) |
| 6. Visualización interactiva (dashboard) | Pendiente (Entrega 3) |

### Resultados preliminares de rendimiento (Kabré, nodo de 40 núcleos)

- Speedup global de hasta **45×** frente a la línea base secuencial (pandas 1 hilo).
- Saturación del paralelismo de memoria compartida en torno a los **16 hilos**
  (consistente con la ley de Amdahl).
- El paralelismo por procesos (Dask) falla por memoria con ≥8 *workers* en las
  operaciones intensivas.
