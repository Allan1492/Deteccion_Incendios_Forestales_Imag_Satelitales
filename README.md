# Detección de Incendios Forestales con Imágenes Satelitales

Proyecto del curso **Computación Paralela y Distribuida (TTCT0017)** — LEAD University
Profesor: Johansell Villalobos Cubillo

## Descripción

Pipeline completo de cómputo paralelo y distribuido para la detección y el
análisis de incendios forestales a partir de datos satelitales masivos,
ejecutado sobre el clúster de alto rendimiento **Kabré** (CeNAT). El objetivo
específico es **reducir la tasa de falsas alarmas** en las detecciones de puntos
de calor, combinando variables físicas y contextuales, además de clasificar
imágenes satelitales mediante aprendizaje profundo.

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
- **Kaggle Wildfire Prediction Dataset** — 42 850 imágenes satelitales
  (`wildfire` / `nowildfire`), ~6.5 GB, para la CNN.
  https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset

> Los archivos de datos NO se versionan (ver `.gitignore`). El Parquet crudo se
> reconstruye con `entrega2_benchmark/descargar_firms.py`; las imágenes de
> Kaggle se descargan según `modelado/README.md` (sección 5.1).

## Estructura del repositorio

```
.
├── Notebook/                 Notebook de preprocesamiento (Etapas 1-2)
├── entrega2_benchmark/       Benchmark de rendimiento CPU (secuencial vs. paralelo)
│   ├── descargar_firms.py    Descarga y consolidación de datos FIRMS
│   ├── verificar_parquet.py  Verificación de integridad del Parquet
│   ├── preparar_dataset_modelo.py  Genera la tabla lista para el modelo (Etapa 2)
│   ├── bench_run.py          Ejecuta una configuración del benchmark
│   ├── run_all.sh            Driver: recorre todas las configuraciones
│   ├── analizar_resultados.py  Métricas (speedup, eficiencia) y gráficos
│   ├── setup_kabre.sh        Monta el entorno en Kabré
│   └── submit_kabre.slurm    Job de SLURM
├── eda/                      Análisis exploratorio
│   └── eda_real.py           EDA sobre los datos reales (streaming)
├── modelado/                 Etapa 4: modelado tabular (cuML/XGBoost/NN) y CNN
│   ├── notebooks/            Notebooks de exploración, modelado, hiperparámetros,
│   │                         benchmark CPU vs. GPU y CNN de imágenes
│   └── resultados/           Tablas (CSV/JSON) y figuras generadas
├── dashboard/               Etapa 6: dashboard interactivo (Plotly/Dash)
├── resultados/              Figuras del benchmark de CPU y del EDA
└── informes/                Informes IEEE (Entregas 1, 2 y 3, en LaTeX/PDF)
```

## Requisitos

- Python 3.11 (Miniforge/Conda recomendado).

Entorno de CPU:

```bash
conda create -n incendios -c conda-forge python=3.11 polars pandas pyarrow \
    dask distributed matplotlib plotly tabulate jupyterlab
conda activate incendios
```

En Kabré, usar `entrega2_benchmark/setup_kabre.sh`. Para la parte de GPU
(RAPIDS cuML, PyTorch con CUDA), ver `modelado/README.md`.

## Reproducción (resumen)

```bash
# 1. Verificar los datos
python entrega2_benchmark/verificar_parquet.py /ruta/incendios_global_consolidado.parquet

# 2. Preparar la tabla del modelo
python entrega2_benchmark/preparar_dataset_modelo.py \
    /ruta/incendios_global_consolidado.parquet /ruta/dataset_modelo.parquet

# 3. Análisis exploratorio
python eda/eda_real.py /ruta/incendios_global_consolidado.parquet eda_salida

# 4. Benchmark de preprocesamiento (CPU, en Kabré vía sbatch)
cd entrega2_benchmark && sbatch submit_kabre.slurm

# 5. Modelado y benchmark CPU vs. GPU
#    Ver modelado/README.md (notebooks en orden)

# 6. Dashboard de resultados
cd dashboard && pip install -r requirements.txt && python app.py
```

## Estado del proyecto

| Etapa | Estado |
|---|---|
| 1. Adquisición y gestión de datos | Completada |
| 2. Preprocesamiento paralelo/distribuido | Completada |
| 3. Análisis exploratorio y estadístico | Completada |
| 4. Modelado ML/IA (cuML, XGBoost, red neuronal + CNN) | Completada |
| 5. Postprocesamiento y análisis de rendimiento (CPU y GPU) | Completada |
| 6. Visualización interactiva (dashboard) | Completada |

## Resultados principales

- **Preprocesamiento paralelo (CPU):** speedup global de hasta **45×** frente a la
  línea base secuencial; saturación en torno a los 16 hilos (ley de Amdahl).
- **Modelos tabulares (GPU):** F1 de hasta **0.82** y ROC-AUC de **0.98** en la
  reducción de falsas alarmas (Random Forest, XGBoost, red neuronal).
- **Clasificación de imágenes (CNN, ResNet18):** F1 de **0.96** y ROC-AUC de **0.99**.
- **CPU vs. GPU:** aceleraciones de hasta **6.8×**, con diferencias marcadas en la
  utilización efectiva de GPU entre algoritmos (XGBoost ~96 % vs. red neuronal ~38 %).

El informe técnico completo está en `informes/`.
