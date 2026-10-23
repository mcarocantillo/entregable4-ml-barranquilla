# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2. Datos: ETL y EDA
#
# :::{admonition} Qué responde este capítulo
# :class: tip
# De dónde salen los datos, cómo se limpiaron y ubicaron, cómo se separaron sin fuga espacial y qué dice el
# análisis exploratorio sobre la relación de cada variable con el estrato.
# :::
#
# El ETL completo está en el Entregable 1 (versión 3, capítulos 1–15); aquí se resume lo que usa el proyecto final y
# se rehace el EDA sobre la submuestra con que se comparan los modelos (10 001 viviendas de entrenamiento), la misma
# que se explora en la página EDA del dashboard.
#
# ## 2.1 Fuente
#
# Catastro abierto de la Alcaldía de Barranquilla (servicio ArcGIS REST de datos abiertos de la Gerencia de Gestión
# Catastral, capas 505, 305, 602, 500, 315 y 601), descargado el 15 de septiembre de 2026. Es una foto transversal:
# no hay dimensión temporal, y ``anio_construccion`` es un atributo de la vivienda, no la fecha de la observación.
# La variable objetivo es el estrato socioeconómico asignado (1 a 6) {cite}`dane2015conceptual`.
#
# ## 2.2 ETL
#
# | Paso | Regla | Viviendas que quedan |
# |---|---|---|
# | Descarga | Registros del catastro con uso residencial | 382 597 |
# | 1 | Quitar usos no habitables | 346 100 |
# | 2 | Área construida 0, menor de 10 m² o registros agregados (> 20 habitaciones, > 15 baños o > 2 000 m²) | 344 811 |
# | 3 | Quitar registros sin estrato residencial (``No_Aplica``, ``Otro``) | **332 718** |
# | 4 | Valores imposibles → faltantes (año 2500 y 1512, pisos 97–99, …) | 332 718 (484 valores) |
# | 5 | Incoherencias → faltantes (< 6 m² por habitación; baños > habitaciones + 3) | 332 718 (396 valores) |
# | 6 | Ubicar por el centroide de su manzana a las viviendas sin coordenadas propias | 96.9 % con coordenadas |
#
# La ubicación por manzana (E1 v2) recuperó 43 594 de las 53 894 viviendas sin coordenadas (casi todas informales de
# estrato 1), con un error mediano de 31 m medido dejando fuera edificios ya ubicados; quedan 10 300 (3.1 %) sin
# ubicar. La partición separa el test en **7 bloques de 2 km** (65 145 viviendas) y el resto es entrenamiento
# (257 273). Dentro del entrenamiento, los pliegues se construyen con **kNNDM** {cite}`linnenbrink2024`: imitan la
# distancia a la que el modelo tendrá que predecir una zona nueva (0.45 km de mediana al dato más cercano), en lugar
# del buffer de 1 km del E1 v2, que evaluaba a 1.4–1.5 km.
#
# Todo lo que aprende de los datos (cuantiles de winsorización, medianas, escalas, categorías infrecuentes,
# remuestreo) se ajusta dentro de cada pliegue con un ``Pipeline``: no hay fuga del pliegue de validación.
#
# ## 2.3 EDA de la submuestra

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
from scipy import stats

RAIZ = Path.cwd().resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import data as D  # noqa: E402
from src import plots as P  # noqa: E402

pio.renderers.default = "notebook"
df = D.viviendas()
print(f"{len(df):,} viviendas, {df.shape[1] - 1} variables; estrato: "
      + ", ".join(f"E{k} {v:.1%}" for k, v in df.estrato_num.value_counts(normalize=True).sort_index().items()))

# %%
P.distribucion_objetivo(df).show()

# %% [markdown]
# **Lectura.** El estrato 1 es el más frecuente (30 %) y el 6 el menos (4.8 %): un desbalance de 6.2 : 1. Por eso la
# métrica principal es el F1 macro, que pesa igual a cada estrato, y se comparan técnicas de balanceo (capítulo 3).

# %%
P.mapa_estratos(df).show()

# %% [markdown]
# **Lectura.** La ciudad crece en estrato de sur a norte y los vecinos se parecen mucho (I de Moran 0.92 en el
# catastro completo, E1 cap. 8). Los cuadrados vacíos son los bloques de test. Esta autocorrelación es la razón de la
# validación espacial: con pliegues aleatorios el F1 se inflaba de 0.46 a 0.58 (E2 v3).

# %%
def eta2_kw(var):
    grupos = [g[var].dropna().to_numpy() for _, g in df.groupby("estrato_num")]
    h = stats.kruskal(*grupos).statistic
    n, k = sum(map(len, grupos)), len(grupos)
    return (h - k + 1) / (n - k)


asoc = pd.DataFrame({"variable": D.NUMERICAS, "eta2_KW": [eta2_kw(v) for v in D.NUMERICAS],
                     "rho_spearman": [df[v].corr(df.estrato_num, method="spearman") for v in D.NUMERICAS]})
asoc.sort_values("eta2_KW", ascending=False).round(3).style.hide(axis="index")

# %% [markdown]
# **Lectura.** Las variables de **vecindad** (``knn8_*``: promedio de los 8 edificios más cercanos) están más asociadas
# al estrato que las de la propia vivienda: el estrato se asigna por zona, y el DANE ajusta solo las viviendas atípicas
# de cada zona {cite}`dane2015conceptual`. Entre las propias, la posición sur–norte (``y_km``), los baños y el área
# construida tienen efecto grande (η² ≥ 0.14).

# %%
P.mapa_correlacion(df, D.NUMERICAS).show()

# %% [markdown]
# **Lectura.** Hay bloques de variables muy correlacionadas (el área propia con la de los vecinos, los baños con las
# habitaciones); la colinealidad es moderada (VIF < 2 en las físicas y 6.1 en ``x_km``, E1 cap. 6) y los modelos con
# regularización o de árboles la toleran.

# %%
P.faltantes(df, D.NUMERICAS + D.CATEGORICAS).show()
P.caja_atipicos(df, "area_construida", "m²", log=True).show()

# %% [markdown]
# **Lectura.** Los faltantes son escasos (máximo 0.15 %, en habitaciones) y se imputan con la mediana del pliegue más
# una columna que indica el faltante. Los atípicos de Tukey (≈ 6 % en el área) son viviendas reales, sobre todo de
# estratos 5 y 6: se conservan y solo se recorta el 0.1 % de cada cola (winsorización).
