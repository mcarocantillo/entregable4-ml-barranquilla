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
# # 3. Modelos benchmark
#
# :::{admonition} Qué responde este capítulo
# :class: tip
# Cuál es la mejor combinación (balanceo y optimizador) de cada modelo, cómo se eligió, cuánto rinde en una validación
# repetida que no participó en la elección y cómo se reentrenó y serializó.
# :::
#
# ## 3.1 Criterio de selección
#
# Del universo del E3 (8 modelos × 4 técnicas de balanceo × 4 métodos de optimización = 128 combinaciones, cada una
# con validación anidada kNNDM de 5 × 3 pliegues y 12 evaluaciones), para cada uno de los **7 modelos de la guía** se
# toma la combinación con mayor **F1 macro medio del bucle externo**. Se agregan el XGBoost geográfico (la propuesta
# del E2/E3, como referencia) y dos métodos recientes del estado del arte, **LightGBM** {cite}`ke2017` y
# **CatBoost** {cite}`prokhorenkova2018`, con pesos de clase y búsqueda bayesiana, el mismo presupuesto y las mismas
# particiones.
#
# La mejor combinación global (Random Forest con SMOTE y algoritmo genético, 0.658) está en el grupo de cabeza de
# Friedman → Nemenyi del E3 (a 1.02 rangos del mejor, con diferencia crítica 1.86): el criterio es estadístico y la
# elección es consistente con la comparación múltiple.
#
# ## 3.2 Mejores combinaciones en la validación repetida 5 × 5

# %%
import json
from pathlib import Path

import pandas as pd

RAIZ = Path.cwd().resolve().parents[1]
T = pd.read_csv(RAIZ / "resultados" / "cv5x5" / "f1_por_pliegue_v2.csv")
meta = json.loads((RAIZ / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
filas = []
for nombre, m in meta["modelos"].items():
    if nombre == "EPL (original)":
        continue
    col = T[nombre]
    filas.append({"modelo": nombre, "grupo": m["grupo"], "balanceo": m["balanceo"], "optimizador": m["metodo"],
                  "F1 rep. 0 (E3)": round(col[T.rep == 0].mean(), 3), "F1 reps. 1–4": round(col[T.rep > 0].mean(), 3),
                  "F1 25 pliegues": f"{col.mean():.3f} ± {col.std(ddof=1):.3f}",
                  "inferencia (ms/viv.)": round(m["ms_inferencia_por_vivienda"], 3)})
tabla = pd.DataFrame(filas).sort_values("F1 reps. 1–4", ascending=False)
tabla.style.hide(axis="index")

# %% [markdown]
# **Lectura.** La repetición 0 son los mismos pliegues con que el E3 eligió las combinaciones; las repeticiones 1–4
# son particiones kNNDM nuevas. **Todos los modelos bajan en las particiones nuevas** (el Random Forest de 0.658 a
# 0.592): es el sesgo de selección de haber elegido y reportado con los mismos pliegues (fila 17 del capítulo 1). El
# orden se mantiene en la cabeza: Random Forest, CatBoost y XGBoost quedan a menos de 0.01 entre sí, y la logística y
# el XGBoost geográfico detrás. Naive Bayes queda último, como en el E3.
#
# ## 3.3 Reentrenamiento final y serialización
#
# Una vez fijada la combinación, el **mismo procedimiento** que se validó (búsqueda con el mismo método y 12
# evaluaciones, usando los 5 pliegues externos de la repetición 0 como validación) se aplica a toda la submuestra
# de entrenamiento y el pipeline completo —preprocesamiento del E1 → [SMOTE] → modelo— se reentrena con las 10 001
# viviendas y se guarda con ``joblib.dump(..., compress=3)`` en ``artifacts/``. Para los 7 modelos de la guía y el
# geográfico se reutilizan los hiperparámetros de esa búsqueda del E3; LightGBM, CatBoost y el modelo original se
# buscan aquí. Junto a cada modelo, ``artifacts/metadata.json`` guarda versiones de las librerías, semilla,
# hiperparámetros, esquema de columnas (tipo y rango válido), métricas del bucle externo y tiempos.

# %%
pd.DataFrame([{"modelo": n, "archivo": m["archivo"], "MB": m["tamano_MB"], "entrenamiento (s)": m["seg_entrenamiento"],
               "ida y vuelta idéntica": m["ida_y_vuelta_identica"], "hiperparámetros": json.dumps(m["params"])[:120]}
              for n, m in meta["modelos"].items()]).style.hide(axis="index")

# %% [markdown]
# **Prueba de ida y vuelta.** Cada archivo se vuelve a cargar y sus probabilidades sobre 2 000 viviendas fijas del test
# se comparan con las del modelo en memoria: son **idénticas** en todos (``tests/test_artefactos.py`` lo repite en
# cada ejecución de las pruebas). Los artefactos pesan entre 0.01 y 10.5 MB; el mayor es el modelo original, que
# guarda sus 7 expertos. Ninguno exige Git LFS ni reducir árboles.
