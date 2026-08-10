# Dashboard Interactivo — Etapa 6

Proyecto: Detección de Incendios Forestales con Imágenes Satelitales
Curso: TTCT0017 — Computación Paralela y Distribuida, LEAD University

Panel web interactivo (Plotly + Dash) que comunica los resultados del proyecto,
con un diseño tipo consola de instrumentos. Integra el problema, la calidad de
los modelos y el análisis de rendimiento CPU vs. GPU en un solo lugar.

## 1. Requisitos

- Python 3.10 o superior.

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

## 3. Secciones

El panel se organiza en nueve secciones navegables:

1. **Resumen** — indicadores clave del proyecto (registros, mayor speedup, mejor F1, F1 de la CNN).
2. **El problema** — falsas alarmas, balance de clases y el hallazgo del contraste térmico (ΔT).
3. **Modelos** — calidad de los tres modelos tabulares y la CNN (F1, ROC-AUC, precisión, recall) con gráfico de barras y radar comparativo.
4. **Speedup** — aceleración GPU vs. CPU por modelo y escala.
5. **Throughput** — registros procesados por segundo.
6. **Tiempos** — tiempo de entrenamiento por modelo y escala.
7. **Eficiencia** — uso de recursos y utilización real de GPU.
8. **Escalabilidad** — índice de crecimiento del tiempo frente al volumen de datos.
9. **Hallazgos** — conclusiones principales.

## 4. Datos que consume (solo lectura)

```
dashboard/data/benchmark_consolidado.csv   Benchmark CPU vs. GPU (Etapa 5)
dashboard/data/comparacion_modelos.csv     Calidad de los modelos tabulares (Etapa 4)
dashboard/data/metricas_cnn.json           Métricas de la CNN de imágenes (Etapa 4)
```

Estos archivos son copias de los resultados generados en las Etapas 4 y 5
(carpeta `modelado/resultados/`). El dashboard no modifica ningún dato: solo
los lee para visualizarlos.

Columnas principales de `benchmark_consolidado.csv`: `dataset_pct`, `modelo`,
`backend`, `tiempo_s`, `f1`, `ram_pico_mb`, `gpu_mem_pico_mb`,
`gpu_util_promedio_pct`, `throughput_reg_s`.

## 5. Estructura

```
dashboard/
├── app.py             Aplicación Dash (diseño, figuras y callbacks)
├── requirements.txt   Dependencias
├── README.md          Este archivo
└── data/              Datos de resultados (solo lectura)
```
