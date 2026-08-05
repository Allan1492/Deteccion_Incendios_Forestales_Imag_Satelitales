# Etapa 4 — Modelado de ML/IA

Proyecto: Detección de Incendios Forestales con Imágenes Satelitales
Curso: TTCT0017 — Computación Paralela y Distribuida, LEAD University

Este paquete contiene el modelado de la Etapa 4: clasificación tabular para reducción de
falsas alarmas, comparando **Random Forest (RAPIDS cuML)**, **XGBoost (GPU)** y una
**red neuronal (PyTorch)**, con ajuste de hiperparámetros y métricas de calidad y de
rendimiento (tiempo, RAM, uso de GPU) listas para el informe IEEE y para el análisis de
rendimiento de la Etapa 5.

---

## 0. Paso previo obligatorio: generar `dataset_modelo.parquet`

Antes de correr cualquier notebook de esta carpeta, hay que generar el dataset de
modelado a partir del Parquet crudo de FIRMS (`incendios_global_consolidado.parquet`,
producido en la Etapa 1). Esto se hace con el script de la Etapa 2:

```bash
python3 preparar_dataset_modelo.py \
    /ruta/a/incendios_global_consolidado.parquet \
    /ruta/a/dataset_modelo.parquet
```

Qué hace este script:
- Calcula las variables derivadas: `delta_t` (contraste térmico), `mes`, `hora`, `es_noche`.
- Calcula la variable objetivo `es_falsa_alarma` (1 si `confidence` es baja, proxy de falsa alarma).
- Filtra registros inválidos (FRP negativo, coordenadas fuera de rango).
- Escribe el resultado en streaming (`sink_parquet`), sin cargar los 52M+ registros en RAM.

> Este archivo pesa varios GB — **no se sube a GitHub**. Se comparte directamente entre
> el equipo (Drive, SCP, etc.) o se regenera con el comando de arriba.

### 1. En Kabré: guardar el Parquet en `/data`, NUNCA en `/home`

`/home` tiene una cuota de solo **10GB por usuario** — un Parquet de varios GB la llena de
inmediato (ya nos pasó, y frenó todo el trabajo hasta resolverlo). `/data` en cambio tiene
espacio de sobra (decenas de TB compartidos), así que ahí es donde debe vivir el dataset:

```bash
# Crear la carpeta de datos en /data (si no existe todavia)
mkdir -p /data/$USER/deteccion_incendios/data

# Generar el Parquet DIRECTO ahi (mejor opcion, no ocupa /home ni un instante)
python3 preparar_dataset_modelo.py \
    /ruta/a/incendios_global_consolidado.parquet \
    /data/$USER/deteccion_incendios/data/dataset_modelo.parquet

# Si ya lo generaste sin querer en /home, muevelo (no lo copies, mv libera el espacio de /home)
mv ~/deteccion_incendios/data/dataset_modelo.parquet /data/$USER/deteccion_incendios/data/
```

Como `USUARIO` se calcula solo a partir de tu carpeta home (`os.path.expanduser("~")`),
cualquier integrante del equipo puede correr los mismos notebooks sin cambiar ni una
línea — cada quien apunta automáticamente a su propia carpeta en `/data`, siempre que
haya guardado el Parquet en esa misma ubicación relativa (`deteccion_incendios/data/`
dentro de su `/data/<usuario>/`).

Si tu cuenta no tiene `/data/<usuario>` creado todavía, no lo puedes crear tú mismo — hay
que pedírselo al profesor o a quien administra el clúster.

---


`datos/`, `modelos/` y `resultados/` (tablas y figuras) se crean solas la primera vez que
corres un notebook — no hay que crearlas a mano.

### Estructura de esta carpeta

```
modelado/
├── README.md
├── notebooks/
│   ├── pipeline_modelado.ipynb     <- notebook principal (todo el flujo, un solo archivo)
│   ├── exploracion_dataset.ipynb   <- EDA + genera train/val/test.parquet
│   ├── benchmark_cpu_vs_gpu.ipynb  <- benchmark CPU vs GPU 
│   └── ajuste_hiperparametros.ipynb <- busqueda de hiperparametros 
│   └── cnn_imagenes_wildfire.ipynb <- Clasificación imágenes satelitales 
└── resultados/
    ├── tablas/       <- CSV/JSON con métricas
    └── figuras/      <- PNG a 150dpi para el informe IEEE
```

---

## 2. Orden para correr los notebooks

```
exploracion_dataset.ipynb   (genera los splits)
        |
        v
ajuste_hiperparametros.ipynb   (opcional, pero recomendado, busca la mejor config)
        |
        v
pipeline_modelado.ipynb   (entrenamiento final + evaluacion + figuras)
        |
        v
benchmark_cpu_vs_gpu.ipynb   (para el analisis de rendimiento de Robson)

cnn_imagenes_wildfire.ipynb   (independiente -- solo requiere el dataset de imagenes, seccion 5.1)

### 2.1 `01_exploracion_dataset.ipynb`

Hace el EDA (nulos, balance de clases, correlaciones) y **guarda a disco**
`train.parquet`, `val.parquet`, `test.parquet` en `DATOS_BASE_DIR/datos/`. Es un
prerrequisito para `06` y `07` (ambos leen estos archivos). El `00` no depende de este
notebook — genera y usa sus propios splits en memoria.

**Modo de ejecución** (primera celda de configuración):
```python
MODO = "dev"          # muestra pequeña (rapido, para probar)
FRACCION_DEV = 0.3
```
Cambia a `MODO = "full"` para el dataset completo (~52M filas) — solo en Kabré con GPU.

### 2.2 `07_ajuste_hiperparametros.ipynb`

Busca la mejor configuración para cada modelo (grid pequeño) sobre una muestra del 15%
del dataset, evaluando F1 en validación. **El criterio de selección no es solo F1**: de
las configuraciones dentro de 0.01 del mejor F1 encontrado, elige la más rápida — evita
elegir una configuración que gana calidad marginal a cambio de mucho más tiempo de
entrenamiento a escala completa (nos pasó con el batch_size de la red neuronal: ver nota
más abajo).

**Salida:** `resultados/tablas/08_mejores_hiperparametros.json` — el `pipeline` lo lee
automáticamente si existe.

### 2.3 `pipeline_modelado.ipynb` 

Hace los 5 pasos completos: explorar → preparar → entrenar los 3 modelos → comparar →
generar tablas/figuras para el informe. Si existe `08_mejores_hiperparametros.json` (del
`ajuste_hiperparametros.ipynb`), lo usa automáticamente; si no, usa valores por defecto razonables.

**Salidas:** `resultados/tablas/03_comparacion_modelos.csv`, `04_resumen_ejecutivo.json`,
figuras en `resultados/figuras/`, modelos entrenados en `DATOS_BASE_DIR/modelos/`.

**Regla importante:** cualquier vez que cambies `MODO`, hiperparámetros, o cualquier
variable de configuración, reinicia el kernel (`Kernel > Restart & Run All`) en vez de
re-correr celdas sueltas — evita resultados inconsistentes por variables viejas en memoria
(nos costó un bug real de cálculo de porcentajes por esto).

### 2.4 `benchmark_cpu_vs_gpu.ipynb`

Entrena Random Forest, XGBoost y la Red Neuronal dos veces cada uno (forzando CPU y
forzando GPU) sobre los mismos datos, mide tiempo, RAM pico y uso de GPU, y calcula el
speedup real GPU vs CPU.

**Salidas:** `resultados/tablas/05_benchmark_cpu_vs_gpu.csv`, `06_speedup_cpu_vs_gpu.csv`,
`resultados/figuras/08_speedup_cpu_vs_gpu.png`.

---

## 3. Dependencias

```bash
pip install --user polars pandas pyarrow scikit-learn xgboost psutil plotly matplotlib joblib
```

- **RAPIDS cuML/cuDF**: ya viene en el entorno `rapids-23.10` de Kabré — usar ese kernel
  de Jupyter.
- **PyTorch con GPU**: instalar apuntando explícitamente al Python del kernel que estés
  usando (no basta con `pip install`, que puede usar otro Python del sistema):
  ```bash
  export PYTHONUSERBASE=/data/$USER/pylibs   # NO instalar en /home, se llena la cuota
  /opt/python/mamba3/envs/rapids-23.10/bin/python -m pip install --user torch==2.4.1 \
      --index-url https://download.pytorch.org/whl/cu121
  ```
  Verificar con `import torch; torch.cuda.is_available()` **dentro del notebook** (no en
  la terminal) — si da `ModuleNotFoundError` ahí pero el `pip install` salió bien, es
  casi siempre un mismatch de versión de Python entre la terminal y el kernel.
- Si no hay GPU disponible, el pipeline cae a CPU automáticamente y ajusta batch
  size/épocas/submuestreo para seguir siendo viable (más lento, pero funcional).

---

## 4. Resultados de referencia (última corrida completa, dataset real, GPU + hiperparámetros ajustados)

| Modelo | F1 | ROC-AUC | Entrenamiento | RAM pico | GPU uso prom. |
|---|---|---|---|---|---|
| Red Neuronal (PyTorch) | 0.8163 | 0.9824 | 74.5 s | 27.9 GB | 37.9 % |
| XGBoost (GPU) | 0.8018 | 0.9834 | 42.9 s | 26.0 GB | 96.5 % |
| Random Forest (cuML) | 0.8034 | 0.9805 | 113.3 s | 19.0 GB | 59.0 % |

Balance de clases: 87.85 % válida / 12.15 % probable falsa alarma (~7:1).

**Nota de rendimiento:** el hallazgo más útil de esta etapa fue que, aunque los tres
modelos corren sobre GPU, el aprovechamiento del hardware es muy distinto (XGBoost ~96%
de uso de GPU vs. Random Forest ~59%) — no toda aceleración por GPU es igual de eficiente.

---

## 5. CNN de imágenes (transfer learning)  `cnn_imagenes_wildfire.ipynb`

Clasifica imágenes satelitales (`wildfire` / `nowildfire`) usando **transfer learning**
sobre ResNet18 preentrenada en ImageNet — se congelan las capas convolucionales y solo se
entrena una nueva capa clasificadora (0.3% de los parámetros totales), lo que reduce
drásticamente el tiempo de entrenamiento comparado con una CNN desde cero.

**Requiere el dataset de imágenes descargado en:**
```
DATOS_BASE_DIR/data/wildfire_images/{train,valid,test}/{wildfire,nowildfire}/*.jpg
```
Ver sección 5.1 para las instrucciones de descarga.

**Salidas:** `modelos/cnn_wildfire.pt`, `resultados/tablas/metricas_cnn.json`,
`resultados/figuras/cnn_curvas_entrenamiento.png`, `cnn_matriz_confusion.png`.

**Notas de rendimiento:** con pocas imágenes de entrenamiento (o `batch_size` chico), la
GPU se usa muy poco (<10%) porque el cuello de botella es la carga/transformación de
imágenes desde disco (I/O, en CPU), no el cómputo del modelo. Subir `batch_size` (256 en
vez de 64) y `NUM_WORKERS` al máximo de cores disponibles, más `persistent_workers=True`
en los `DataLoader`, mejoró el tiempo de 302.6s a 92.1s (3.3×) con la misma calidad — buen
ejemplo de que el cuello de botella no siempre es el modelo, sino cómo se le alimentan
los datos.

## 5.1 Instrucciones de Descarga (CNN de imágenes)

Dataset de imágenes (para la CNN, notebook `cnn_imagenes_wildfire.ipynb`)

El *Kaggle Wildfire Prediction Dataset* (~6.5GB) también debe vivir en `/data`, no en
`/home`. Descarga manual (sin necesidad de API token de Kaggle):

1. Descarga el ZIP desde [kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) (botón *Download*, requiere cuenta de Kaggle).
2. Crea la carpeta destino en Kabré (por SSH):
   ```bash
   mkdir -p /data/$USER/deteccion_incendios/data/wildfire_images
   ```
3. Sube el ZIP desde tu computadora (no desde Kabré):
   ```bash
   scp -P 22022 "ruta/al/archivo.zip" \
       ulead-28@kabre.cenat.ac.cr:/data/ulead-28/deteccion_incendios/data/wildfire_images/
   ```
4. Descomprime y limpia (de vuelta en Kabré):
   ```bash
   cd /data/$USER/deteccion_incendios/data/wildfire_images
   unzip "archivo.zip" && rm "archivo.zip"
   ```

Estructura esperada después de descomprimir: `{train,valid,test}/{wildfire,nowildfire}/*.jpg`.

> **Imágenes truncadas:** el dataset de Kaggle trae algunas imágenes corruptas/incompletas
> que rompen la carga (`OSError: image file is truncated`). El notebook `cnn_imagenes_wildfire.ipynb` ya incluye
> `ImageFile.LOAD_TRUNCATED_IMAGES = True` para tolerarlas automáticamente.
