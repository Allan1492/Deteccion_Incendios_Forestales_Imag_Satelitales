# Dashboard de Análisis de Rendimiento — Etapa 6

Proyecto: Detección de Incendios Forestales con Imágenes Satelitales
Curso: TTCT0017 — Computación Paralela y Distribuida, LEAD University

Dashboard web interactivo (Plotly + Dash) que comunica los resultados del
análisis de rendimiento del proyecto: la comparación CPU vs. GPU de los modelos
de la Etapa 4, con tiempos de cómputo, speedup, throughput, escalabilidad y uso
de recursos.

## 1. Requisitos

- Python 3.10 o superior.
- Dependencias en `requirements.txt`:

```bash
pip install -r requirements.txt
```

(`dash`, `pandas`, `plotly`)

## 2. Cómo ejecutarlo

Desde esta carpeta:

```bash
python app.py
```

Luego abrir en el navegador:

```
http://127.0.0.1:8050
```

## 3. Datos que consume

El dashboard lee un único archivo consolidado:

```
dashboard/data/benchmark_consolidado.csv
```

Este CSV contiene los resultados del benchmark CPU vs. GPU generado en la
Etapa 5 (notebook `modelado/notebooks/benchmark_cpu_vs_gpu.ipynb`), con una fila
por combinación de modelo, dispositivo (CPU/GPU) y escala del dataset
(30 %, 60 %, 100 %). Sus columnas principales son:

| Columna | Descripción |
|---|---|
| `dataset_pct` | Escala del dataset (30 %, 60 %, 100 %) |
| `modelo` | Random Forest, XGBoost o Red Neuronal |
| `backend` | Dispositivo y librería (ej. `GPU (cuML)`, `CPU (sklearn)`) |
| `tiempo_s` | Tiempo de entrenamiento (s) |
| `f1` | F1-score del modelo |
| `ram_pico_mb` | RAM pico (MB) |
| `gpu_mem_pico_mb` | Memoria GPU pico (MB) |
| `gpu_util_promedio_pct` | Utilización promedio de GPU (%) |
| `throughput_reg_s` | Registros procesados por segundo |

## 4. Qué muestra

- **Speedup GPU vs. CPU** por modelo y escala de datos.
- **Throughput** (registros/s) en escala logarítmica.
- **Tiempos de cómputo** por modelo, filtrables por escala del dataset.
- **Uso de recursos**: RAM pico, memoria GPU y utilización promedio de GPU.
- **Escalabilidad**: índice de crecimiento del tiempo frente al tamaño del
  dataset, comparado con el crecimiento lineal ideal.

Los filtros permiten explorar cada métrica por escala de datos, para comunicar
los hallazgos principales del proyecto: las aceleraciones alcanzadas por GPU y
las diferencias de eficiencia (utilización de GPU) entre algoritmos.

## 5. Estructura

```
dashboard/
├── app.py                        Aplicación Dash (layout, figuras, callbacks)
├── requirements.txt              Dependencias
├── README.md                     Este archivo
└── data/
    └── benchmark_consolidado.csv Resultados del benchmark CPU vs. GPU
```
