# Etapa 3 — Mediciones de rendimiento (secuencial vs. paralelo)

Proyecto: **Detección de Incendios Forestales con Imágenes Satelitales**
Curso: TTCT0017 — Computación Paralela y Distribuida, LEAD University
Responsable de esta parte: **Esteban Gutiérrez Saborío**

Este paquete produce las mediciones de rendimiento exigidas para la
**Entrega 2**: speedup, eficiencia, escalabilidad fuerte y débil, throughput
y uso de memoria, comparando una línea base **secuencial real** contra
motores **paralelos** y, opcionalmente, **GPU**.

---

## 1. Archivos

| Archivo | Qué hace |
|---|---|
| `verificar_parquet.py` | Diagnostica si el Parquet está íntegro. **Correr primero.** |
| `bench_run.py` | Ejecuta UNA configuración y agrega una fila al CSV de resultados. |
| `run_all.sh` | Driver: recorre todas las configuraciones del experimento. |
| `submit_kabre.slurm` | Job de SLURM para lanzar todo en el clúster Kabré. |
| `analizar_resultados.py` | Calcula métricas y genera tablas + gráficos para el informe. |

---

## 2. Por qué son scripts y no un notebook

El número de hilos de Polars se fija con la variable de entorno
`POLARS_MAX_THREADS`, y **solo surte efecto antes de importar la librería**.
En un notebook no se puede cambiar tras el primer `import`, así que cada
configuración debe ejecutarse en un proceso nuevo. Por eso `bench_run.py`
mide una sola configuración y `run_all.sh` lo invoca repetidamente.

Medir el paralelismo dentro de un solo notebook daría resultados inválidos.

---

## 3. Uso rápido

```bash
# 0) Verificar que el dataset está íntegro (evita perder horas de cómputo)
python3 verificar_parquet.py /ruta/incendios_global_consolidado.parquet

# 1) Correr el benchmark completo
chmod +x run_all.sh
./run_all.sh /ruta/incendios_global_consolidado.parquet resultados.csv

# 2) Generar tablas y gráficos
python3 analizar_resultados.py resultados.csv --figuras figuras/
```

### En el clúster Kabré

> **IMPORTANTE — Kabré usa el puerto SSH `22022`, no el 22.**
> Conectarse al puerto por defecto produce `Operation timed out`, que
> parece un bloqueo de red pero en realidad es solo el puerto equivocado.
> Nota que `ssh` usa `-p` minúscula y `scp` usa `-P` mayúscula.

```bash
# Subir el paquete (P mayúscula en scp)
scp -P 22022 -r entrega2_benchmark ulead-17@kabre.cenat.ac.cr:~/

# Entrar (p minúscula en ssh)
ssh -p 22022 ulead-17@kabre.cenat.ac.cr

cd entrega2_benchmark
sinfo                                  # ver el nombre real de las particiones
# editar submit_kabre.slurm: descomentar y ajustar --partition
DATOS=/data/$USER/incendios.parquet sbatch submit_kabre.slurm

squeue -u $USER                        # seguir el estado
tail -f bench_<jobid>.out              # ver la salida en vivo

# Descargar resultados y figuras a la máquina local
scp -P 22022 -r ulead-17@kabre.cenat.ac.cr:~/entrega2_benchmark/figuras_* .
```

**Dónde guardar los datos.** Kabré tiene tres directorios con cuotas
distintas; el Parquet de FIRMS (>2 GB) va en `/data`, no en `/home`:

| Directorio | Cuota | Propósito | Respaldo |
|---|---|---|---|
| `/home/$USER` | 10 GB | Scripts y datos importantes | Mensual |
| `/work/$USER` | 100 GB | Datos temporales y herramientas | No |
| `/data/$USER` | 100 GB | **Datos masivos a analizar** | No |

**Alternativas a la terminal:** Kabré ofrece **JupyterHub** (notebooks por
navegador), y también puede usarse **VS Code con Remote-SSH** o clientes
SFTP gráficos como Cyberduck/FileZilla. Todos requieren el puerto `22022`.

### Dependencias

```bash
pip install polars pandas pyarrow matplotlib "dask[dataframe]" distributed tabulate
# cuDF/RAPIDS solo si hay GPU; el script lo detecta y lo omite si no está.
```

---

## 4. Diseño del experimento

### Backends comparados

| Backend | Rol | Paralelismo |
|---|---|---|
| `pandas` | **Línea base secuencial ($T_1$)** | 1 hilo (BLAS/OpenMP forzados a 1) |
| `polars` | Motor columnar multinúcleo | Se escala con `POLARS_MAX_THREADS` |
| `dask` | Paralelismo por tareas / distribuido | Se escala por número de *workers* |
| `cudf` | Aceleración por GPU (opcional) | Miles de hilos GPU |

La línea base es **pandas con un solo hilo**, no el motor `in-memory` de
Polars. Esto importa: el motor `in-memory` de Polars **también es multihilo**,
así que compararlo contra el `streaming` no mide "secuencial vs. paralelo".

### Operaciones medidas

Son operaciones reales del pipeline del proyecto, no sintéticas:

| Operación | Qué ejercita |
|---|---|
| `carga` | Lectura del Parquet y materialización (I/O + descompresión) |
| `filtro` | Descarte de detecciones de baja confianza (filtrado por fila) |
| `agregacion` | Conteo y FRP medio por satélite y mes (*group-by*, shuffle) |
| `features` | `delta_t = brightness - bright_t31`, mes y hora (transformación) |

`features` calcula justamente la variable **ΔT** del problema de falsas
alarmas, así que el benchmark mide el pipeline que de verdad se usará.

### Experimentos

- **Escalabilidad fuerte:** tamaño de datos fijo, $p$ creciente (1, 2, 4, 8, 16…).
- **Escalabilidad débil:** el tamaño crece en proporción a $p$ (`--frac`), de
  modo que la carga *por hilo* se mantiene constante. Lo ideal es una línea plana.
- **Repeticiones:** 3 por configuración; se reporta la **mediana** (más
  robusta que la media ante la primera lectura sin caché).

---

## 5. Métricas y su interpretación

El script calcula **dos speedups distintos**, y la diferencia es importante:

**1. Speedup global** — $T_{\text{pandas}} / T_p$
Ganancia total de cambiar de tecnología **y** paralelizar. Responde
"¿cuánto ganamos en total?".

**2. Speedup paralelo** — $T_{\text{backend}}(1) / T_{\text{backend}}(p)$
Compara cada motor **consigo mismo**. Aísla el efecto del paralelismo puro.

$$E(p) = \frac{S_{\text{paralelo}}(p)}{p}$$

> **La eficiencia se calcula siempre con el speedup paralelo.** Usar el
> global produce eficiencias imposibles (>100 %), porque atribuiría al
> paralelismo una ganancia que en realidad viene del motor. Este es
> exactamente el tipo de error que el profesor puede señalar.

También se reportan: throughput (filas/s y MB/s), memoria pico (RSS) y
núcleos disponibles (respetando la reserva de SLURM).

---

## 6. Salidas

| Salida | Contenido |
|---|---|
| `resultados.csv` | Una fila por corrida (datos crudos) |
| `tabla_resumen.csv` / `.md` | Promedios con speedup, eficiencia y throughput |
| `figuras/speedup_<op>.png` | Speedup vs. $p$ con la recta ideal |
| `figuras/eficiencia.png` | Eficiencia (%) vs. $p$ |
| `figuras/throughput.png` | Throughput vs. $p$ |
| `figuras/escalabilidad_debil.png` | Tiempo vs. $p$ con carga proporcional |
| `figuras/tiempos_por_backend.png` | Comparación por operación y motor |
| `figuras/memoria.png` | Memoria pico por backend |

Las figuras se guardan en PNG a 150 dpi, listas para insertarse en el
informe IEEE con `\includegraphics`.

---

## 7. Cómo leer los resultados en el informe

- **Speedup por debajo de la recta ideal:** es lo normal. Se explica con la
  **ley de Amdahl**: la fracción secuencial (lectura de disco, coordinación)
  pone un techo a la aceleración.
- **Eficiencia que cae al subir $p$:** indica saturación de memoria o de I/O,
  o sobrecarga de coordinación entre *workers*.
- **Dask más lento que Polars con pocos núcleos:** esperable. Dask paga un
  costo fijo de arranque de procesos y serialización que solo se amortiza
  con datos grandes o varios nodos.
- **Escalabilidad débil plana:** buena señal; el sistema absorbe más datos
  sin degradarse.

---

## 8. Pendientes antes de entregar

- [ ] **Reparar el Parquet real** (el EDA falló con `The file must end with PAR1`;
      el archivo está truncado). Verificarlo con `verificar_parquet.py`.
- [ ] Re-correr el EDA de la Etapa 3 sobre datos **reales**, no sintéticos.
- [ ] Correr este benchmark sobre el dataset real (en Kabré o en local).
- [ ] Subir código, resultados y figuras al repositorio.
- [ ] Completar el README del repo y declarar dependencias en `pyproject.toml`.
