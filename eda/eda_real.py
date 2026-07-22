#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDA de la Etapa 3 sobre los datos REALES de FIRMS (52.5 M registros).

Versión de memoria acotada: usa evaluación perezosa y motor 'streaming', de
modo que NUNCA materializa el conjunto completo. Cada agregación procesa el
parquet por bloques y solo trae a memoria el resultado pequeño. Las figuras
basadas en muestra usan 'gather_every' para submuestrear sin cargar todo.
"""
import os, sys, time
import polars as pl
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA = sys.argv[1] if len(sys.argv) > 1 else "incendios_global_consolidado.parquet"
SAL = sys.argv[2] if len(sys.argv) > 2 else "eda_salida"
os.makedirs(os.path.join(SAL, "html"), exist_ok=True)
os.makedirs(os.path.join(SAL, "png"), exist_ok=True)

rep = open(os.path.join(SAL, "reporte_eda.txt"), "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); rep.write(s + "\n"); rep.flush()

def agg(expr_list, sort=None, desc=False):
    lf = pl.scan_parquet(RUTA)
    out = lf
    return out

log("="*64); log("EDA SOBRE DATOS REALES DE FIRMS (streaming)"); log("="*64)
log("Archivo:", RUTA)
t0 = time.perf_counter()

lf = pl.scan_parquet(RUTA)
n_total = lf.select(pl.len()).collect().item()
log("Filas totales: {:,}".format(n_total))

# Rango de fechas y nulos (barato)
log("\n--- Rango de fechas ---")
log(lf.select(pl.col("acq_date").min().alias("desde"), pl.col("acq_date").max().alias("hasta")).collect())

# 2. Serie diaria
serie = lf.group_by("acq_date").agg(pl.len().alias("n")).sort("acq_date").collect(engine="streaming")
sp = serie.to_pandas()
px.line(sp, x="acq_date", y="n", title="Detecciones por día",
        labels={"acq_date":"Fecha","n":"N° detecciones"}).write_html(os.path.join(SAL,"html","01_serie_diaria.html"))
plt.figure(figsize=(8,3.5)); plt.plot(sp["acq_date"], sp["n"], lw=0.8, color="#c0392b")
plt.title("Detecciones de focos de calor por día"); plt.xlabel("Fecha"); plt.ylabel("N° detecciones")
plt.tight_layout(); plt.savefig(os.path.join(SAL,"png","01_serie_diaria.png"), dpi=150); plt.close()
log("\n--- Días pico ---"); log(serie.sort("n", descending=True).head(3))
log("Progreso: serie diaria OK ({:.0f}s)".format(time.perf_counter()-t0))

# 3. Mensual
mes = lf.with_columns(pl.col("acq_date").dt.month().alias("mes")).group_by("mes").agg(pl.len().alias("n")).sort("mes").collect(engine="streaming")
mp = mes.to_pandas()
px.bar(mp, x="mes", y="n", title="Detecciones por mes (estacionalidad)").write_html(os.path.join(SAL,"html","02_mensual.html"))
plt.figure(figsize=(7,3.5)); plt.bar(mp["mes"], mp["n"], color="#e67e22")
plt.title("Detecciones por mes (estacionalidad)"); plt.xlabel("Mes"); plt.ylabel("N° detecciones")
plt.xticks(range(1,13)); plt.tight_layout(); plt.savefig(os.path.join(SAL,"png","02_mensual.png"), dpi=150); plt.close()

# 4. Horaria
hora = lf.with_columns((pl.col("acq_time")//100).alias("hora")).group_by("hora").agg(pl.len().alias("n")).sort("hora").collect(engine="streaming")
hp = hora.to_pandas()
px.bar(hp, x="hora", y="n", title="Detecciones por hora (UTC)").write_html(os.path.join(SAL,"html","03_horaria.html"))
plt.figure(figsize=(7,3.5)); plt.bar(hp["hora"], hp["n"], color="#2980b9")
plt.title("Detecciones por hora (UTC)"); plt.xlabel("Hora"); plt.ylabel("N° detecciones")
plt.tight_layout(); plt.savefig(os.path.join(SAL,"png","03_horaria.png"), dpi=150); plt.close()
log("Progreso: temporal OK ({:.0f}s)".format(time.perf_counter()-t0))

# 5. Mapa dispersion (submuestreo barato con gather_every)
paso = max(1, n_total // 50_000)
mm = lf.select("latitude","longitude").gather_every(paso).collect(engine="streaming").to_pandas()
fig = px.scatter_geo(mm, lat="latitude", lon="longitude", opacity=0.4,
                     title="Distribución geográfica (muestra {:,})".format(len(mm)))
fig.update_geos(showcountries=True, showcoastlines=True)
fig.write_html(os.path.join(SAL,"html","04_mapa_dispersion.html"))
plt.figure(figsize=(8,4)); plt.scatter(mm["longitude"], mm["latitude"], s=1, alpha=0.3, color="#c0392b")
plt.title("Distribución geográfica (muestra de {:,})".format(len(mm))); plt.xlabel("Longitud"); plt.ylabel("Latitud")
plt.tight_layout(); plt.savefig(os.path.join(SAL,"png","04_mapa_dispersion.png"), dpi=150); plt.close()

# 6. Densidad 1x1
dens = (lf.with_columns([pl.col("latitude").round(0).alias("lat_bin"), pl.col("longitude").round(0).alias("lon_bin")])
          .group_by(["lat_bin","lon_bin"]).agg(pl.len().alias("n")).collect(engine="streaming"))
log("\n--- Top 5 celdas 1x1 con más detecciones ---"); log(dens.sort("n", descending=True).head(5))
log("Progreso: espacial OK ({:.0f}s)".format(time.perf_counter()-t0))

# 7. Satelite
sat = lf.group_by("satellite").agg(pl.len().alias("n")).sort("n", descending=True).collect(engine="streaming")
px.bar(sat.to_pandas(), x="satellite", y="n", title="Detecciones por satélite").write_html(os.path.join(SAL,"html","06_satelite.html"))
log("\n--- Por satélite ---"); log(sat)

# 8. Confianza
conf = lf.group_by("confidence").agg(pl.len().alias("n")).sort("n", descending=True).collect(engine="streaming")
cp = conf.to_pandas(); tot = cp["n"].sum()
px.pie(cp, names="confidence", values="n", title="Nivel de confianza").write_html(os.path.join(SAL,"html","07_confianza.html"))
plt.figure(figsize=(5,5)); plt.pie(cp["n"], labels=cp["confidence"], autopct="%1.1f%%", colors=["#95a5a6","#e74c3c","#27ae60"])
plt.title("Nivel de confianza de las detecciones"); plt.tight_layout()
plt.savefig(os.path.join(SAL,"png","07_confianza.png"), dpi=150); plt.close()
log("\n--- Confianza (variable objetivo) ---")
for row in conf.iter_rows(named=True):
    log("  {:<4} {:>12,}  ({:5.2f}%)".format(row["confidence"], row["n"], 100*row["n"]/tot))

# 9. FRP (submuestreo)
paso_f = max(1, n_total // 500_000)
frp_m = lf.select("frp").gather_every(paso_f).collect(engine="streaming").to_pandas()
px.histogram(frp_m, x="frp", nbins=80, title="Distribución del FRP (MW)").write_html(os.path.join(SAL,"html","08_frp.html"))
plt.figure(figsize=(7,3.5)); plt.hist(frp_m["frp"].clip(upper=frp_m["frp"].quantile(0.99)), bins=80, color="#8e44ad")
plt.title("Distribución del FRP (recortado al p99)"); plt.xlabel("FRP (MW)"); plt.ylabel("Frecuencia")
plt.tight_layout(); plt.savefig(os.path.join(SAL,"png","08_frp.png"), dpi=150); plt.close()
log("\n--- Estadísticas de FRP ---")
log(lf.select(pl.col("frp").mean().alias("media"), pl.col("frp").median().alias("mediana"),
              pl.col("frp").std().alias("desv"), pl.col("frp").quantile(0.95).alias("p95"), pl.col("frp").max().alias("max")).collect(engine="streaming"))

# 10. delta_t por clase (clave del problema de falsas alarmas)
dtc = (lf.with_columns((pl.col("brightness")-pl.col("bright_t31")).alias("delta_t"))
         .group_by("confidence").agg([pl.col("delta_t").mean().alias("delta_t_medio"),
                                      pl.col("frp").mean().alias("frp_medio"), pl.len().alias("n")]).sort("confidence").collect(engine="streaming"))
log("\n--- delta_t y FRP medios por nivel de confianza (clave para falsas alarmas) ---"); log(dtc)
log("Progreso: FRP y delta_t OK ({:.0f}s)".format(time.perf_counter()-t0))

# 11. Correlacion (submuestreo)
num = [c for c in ["brightness","bright_t31","frp","latitude","longitude","scan","track"] if c in lf.collect_schema().names()]
paso_c = max(1, n_total // 1_000_000)
corr = lf.select(num).gather_every(paso_c).collect(engine="streaming").to_pandas().corr()
px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
          title="Matriz de correlación").write_html(os.path.join(SAL,"html","09_correlacion.html"))
plt.figure(figsize=(6,5)); im=plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xticks(range(len(num)), num, rotation=45, ha="right"); plt.yticks(range(len(num)), num)
for i in range(len(num)):
    for j in range(len(num)):
        plt.text(j,i,"{:.2f}".format(corr.iloc[i,j]), ha="center", va="center", fontsize=7)
plt.colorbar(im); plt.title("Matriz de correlación"); plt.tight_layout()
plt.savefig(os.path.join(SAL,"png","09_correlacion.png"), dpi=150); plt.close()

log("\n" + "="*64); log("EDA COMPLETO en {:.1f}s".format(time.perf_counter()-t0)); log("="*64)
rep.close()
