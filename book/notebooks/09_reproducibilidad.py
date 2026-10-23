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
# # 9. Reproducibilidad y bibliografía
#
# :::{admonition} Qué responde este capítulo
# :class: tip
# Con qué entorno, semillas y pasos se reproducen todos los resultados, y qué controles verifican que se reproducen.
# :::
#
# ## 9.1 Entorno

# %%
import json
from pathlib import Path

import pandas as pd

RAIZ = Path.cwd().resolve().parents[1]
meta = json.loads((RAIZ / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
pd.Series(meta["librerias"], name="versión").to_frame()

# %% [markdown]
# Las versiones exactas están en ``requirements.txt`` (dashboard) y ``requirements-dev.txt`` (experimentos, pruebas y
# libro). Los experimentos se corrieron en Windows 11 con Python 3.14 y 4 núcleos; el dashboard se despliega con
# Python 3.11.15 y las mismas versiones de scikit-learn, numpy y pandas (los ``.joblib`` se cargan igual).
#
# ## 9.2 Semillas y determinismo
#
# - **Semilla global 42** (``estrato3/configuracion.py``): partición, submuestra, pliegues kNNDM de la repetición 0,
#   modelos, optimizadores (Optuna y DEAP), ajuste cruzado y K-medias del modelo original. Las repeticiones 1–4 de la
#   validación usan las semillas 43–46.
# - **Un hilo en BLAS/OpenMP** en todos los lanzadores, en las pruebas (``tests/conftest.py``) y en el dashboard
#   (``app.py``). Con varios hilos, el orden de las sumas en coma flotante cambia: aparecen diferencias de hasta 0.02 en
#   algunas probabilidades del KNN (empates de distancia) y de 0.003 en el modelo original. Con un hilo, la prueba de
#   ida y vuelta es exacta.
#
# ## 9.3 Controles de reproducibilidad
#
# | Control | Resultado |
# |---|---|
# | Los 7 expertos del modelo original reproducen las probabilidades de sus tareas benchmark en cada pliegue | diferencia máxima 3·10⁻⁸ (redondeo de float32) |
# | La variante «solo geográfico» de la versión 2 reproduce la versión 1 (misma búsqueda) | diferencia máxima 0 en los 25 pliegues |
# | ``fit_con_probabilidades`` reproduce ``fit`` con las mismas probabilidades | prueba unitaria (igualdad exacta) |
# | La condición «base» de la robustez reproduce la versión 2 | mismo F1 en los pliegues de la repetición 0 |
# | Ida y vuelta de los 11 artefactos | idénticos (``tests/test_artefactos.py``) |
#
# ## 9.4 Orden de ejecución
#
# Todo se corre desde la raíz del repositorio con ``python -m experimentos.<script>`` (ver ``README.md``):
# ``01_particiones`` → ``02_benchmarks_5x5`` → ``02b_recientes_5x5`` → ``03_epl_5x5`` → ``03b_epl_v2_5x5`` →
# ``04_estadistica v1`` y ``v2`` → ``05_descargar_externos`` → ``06_externos`` (rivales) → ``06b_externos_v2`` →
# ``07_artefactos`` → ``08_datos_dashboard`` → ``09_test_confirmatorio`` → ``10_datos_modelos_dashboard`` →
# ``11_robustez`` → ``12_complejidad`` → ``13_tabla_maestra``. Cada tarea guarda su resultado y se salta si ya existe,
# así que las cadenas (``experimentos/correr_*.sh``) se pueden interrumpir y relanzar. El libro se construye con
# ``python book/construir_libro.py`` a partir de esos resultados.
#
# ## 9.5 Tabla maestra de experimentos

# %%
tm = pd.read_csv(RAIZ / "resultados" / "tabla_maestra.csv")
print(f"{len(tm)} filas (una por combinación y pliegue externo) en resultados/tabla_maestra.csv")
tm.groupby("experimento").agg(filas=("f1_macro", "size"), modelos=("modelo", "nunique"),
                              F1_medio=("f1_macro", "mean")).round(4)

# %% [markdown]
# Las 28 combinaciones de regresión del E3 (140 tareas) están en ``data/e3/tabla_maestra_regresion.csv``.
#
# ## 9.6 Bibliografía
#
# ```{bibliography}
# ```
