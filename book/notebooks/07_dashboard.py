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
# # 7. Dashboard y despliegue
#
# :::{admonition} Qué responde este capítulo
# :class: tip
# Cómo está construido el dashboard, qué decisiones de diseño se tomaron, cómo se despliega en Render y dónde está.
# :::
#
# ## 7.1 Arquitectura
#
# Aplicación **multipágina** de Dash (``use_pages=True``), con una página por sección y la lógica separada de la
# interfaz:
#
# ```text
# app.py                  punto de entrada: crea la app, la barra fija y expone server (gunicorn)
# pages/contexto.py       sección 1: problema, pregunta y objetivos, datos y diccionario, metodología, ética, autores
# pages/eda.py            sección 2: filtros globales → dcc.Store → 7 gráficos con interpretación
# pages/modelos.py        sección 3: líderes, predicción en vivo con explicación local, diagnóstico, pruebas, EPL
# src/original_model.py   clase del modelo original
# src/data.py             carga y caché (lru_cache) de datos y artefactos
# src/plots.py            funciones que construyen todas las figuras (una plantilla Plotly)
# src/explicacion.py      valores de Shapley por permutaciones (explicación local)
# artifacts/              modelos serializados (joblib) + metadata.json + referencia de ida y vuelta
# data/dashboard/         datos livianos y resultados ya calculados
# assets/                 CSS (paleta), tema Flatly local, imagen satelital de portada
# tests/                  pruebas (modelo original, explicación, ida y vuelta)
# ```
#
# El dashboard **nunca entrena ni busca hiperparámetros**: lee ``data/dashboard/`` (generado por
# ``experimentos/08_datos_dashboard.py`` y ``10_datos_modelos_dashboard.py``) y carga los modelos de ``artifacts/``
# una sola vez por proceso (``functools.lru_cache``).
#
# ## 7.2 Contenido de cada sección
#
# | Sección | Qué responde | Componentes |
# |---|---|---|
# | Contexto | ¿Qué problema es y por qué importa? | Portada con cifra clave citada, relevancia (Ley 142, Ley 1450, DANE), pregunta y objetivos, fuente y diccionario de 22 variables, diagrama de la metodología, limitaciones, ética, autores, versión y fecha |
# | EDA | ¿Qué variables se asocian al estrato y dónde? | Filtros globales (estratos, régimen, área, posición sur–norte) que actualizan todos los gráficos; distribución del objetivo; mapa; histograma y atípicos por variable (con escala log); correlación de Spearman; faltantes; explorador bivariado (x, y y color a elección); una frase de interpretación por gráfico, recalculada con los filtros |
# | Modelos | ¿Qué modelo es mejor, qué predice para mi vivienda y por qué? | Tabla de líderes ordenable y filtrable; selector de modelo; predicción en vivo con formulario generado del esquema, validación de rangos y tipos, viviendas de ejemplo del test y clic en el mapa; probabilidad por estrato y explicación local; matriz de confusión, ROC, precisión-recall y calibración en el test; importancia global; diagrama de diferencia crítica y tabla de pruebas (versión 1 y 2); subsección del modelo original con pseudocódigo, diferencias con la literatura, convergencia, ablación, campo de pesos y veredicto |
#
# ## 7.3 Decisiones de diseño
#
# - **Paleta de cinco colores** (azul oscuro, azul, dorado, naranja solo para el modelo original, gris) y una rampa de
#   un solo tono para los estratos; cada gráfico rotula los estratos, así que no depende solo del color.
# - **Tema Flatly** de dash-bootstrap-components como archivo local (no depende de un CDN), barra fija, rejilla
#   ``dbc.Row``/``dbc.Col`` para escritorio y tableta.
# - **Rendimiento**: datos y modelos se cargan una vez; los filtros calculan una sola vez las filas que cumplen y las
#   guardan en un ``dcc.Store`` que leen los demás callbacks; ``prevent_initial_call`` y ``no_update`` donde aplica.
# - **Robustez**: el formulario valida tipo y rango de cada entrada (rango de entrenamiento guardado en
#   ``metadata.json``) y los errores se muestran con ``dbc.Alert``; un error del modelo no detiene la aplicación.
# - **Explicación local sin la librería shap**: valores de Shapley por muestreo de permutaciones
#   {cite}`strumbelj2014` (el mismo estimador de ``shap.PermutationExplainer``) implementados en
#   ``src/explicacion.py``, con eficiencia exacta (las contribuciones suman la diferencia entre la predicción y el
#   valor base). Se evita instalar ``shap`` y sus dependencias en un servidor de 512 MB.
# - **Privacidad**: sin número predial y con las coordenadas redondeadas a ~100 m.
#
# ## 7.4 Despliegue en Render
#
# 1. Verificar en local: ``python app.py`` (puerto 8050) y que ``app.server`` existe.
# 2. ``requirements.txt`` con versiones exactas (incluye ``gunicorn``) y ``render.yaml`` con
#    ``PYTHON_VERSION = 3.11.15``.
# 3. Subir el repositorio a GitHub.
# 4. En Render: *New → Web Service* (o *Blueprint* con ``render.yaml``), conectar el repositorio, *Build Command*
#    ``pip install -r requirements.txt``, *Start Command* ``gunicorn app:server --workers 1 --threads 2 --timeout 120``.
# 5. Revisar los registros de construcción y abrir todas las páginas y callbacks en la URL pública.
#
# En el plan gratuito el servicio se duerme sin visitas (la primera carga tarda cerca de un minuto) y tiene 512 MB de
# memoria; por eso los artefactos se guardan comprimidos y la página de modelos carga cada modelo solo cuando se
# elige. Los tamaños y la memoria medida están en la sección 7.5.
#
# ## 7.5 Artefactos

# %%
import json
from pathlib import Path

import pandas as pd

RAIZ = Path.cwd().resolve().parents[1]
ruta = RAIZ / "artifacts" / "metadata.json"
if ruta.exists():
    meta = json.loads(ruta.read_text(encoding="utf-8"))
    tabla = pd.DataFrame([{"modelo": n, "archivo": m["archivo"], "MB": m["tamano_MB"],
                           "entrenamiento (s)": m["seg_entrenamiento"],
                           "inferencia (ms por vivienda)": round(m["ms_inferencia_por_vivienda"], 3),
                           "ida y vuelta idéntica": m["ida_y_vuelta_identica"]} for n, m in meta["modelos"].items()])
    display(tabla)
    print("Versiones:", meta["librerias"])
else:
    print("Pendiente: experimentos/07_artefactos.py")

# %% [markdown]
# ## 7.6 Capturas
#
# ```{figure} ../figuras/captura_contexto.png
# :width: 100%
# :name: fig-captura-contexto
#
# Página Contexto: portada con la imagen satelital, relevancia, pregunta y objetivos, datos, metodología y autores.
# ```
#
# ```{figure} ../figuras/captura_eda.png
# :width: 100%
# :name: fig-captura-eda
#
# Página EDA: filtros globales arriba y cada gráfico con su frase de interpretación.
# ```
#
# ```{figure} ../figuras/captura_modelos.png
# :width: 100%
# :name: fig-captura-modelos
#
# Página Modelos: tabla de líderes, predicción en vivo y diagnóstico del modelo elegido en el test.
# ```
#
# ## 7.7 URL pública
#
# El dashboard se publica en Render desde el repositorio del proyecto; la dirección exacta queda en la introducción
# del libro y en el ``README.md``.
