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
# # 4. Revisión de literatura y evidencia de novedad
#
# :::{admonition} Qué responde este capítulo
# :class: tip
# ¿Alguien ya usó un ensamble con pesos locales (o algo equivalente) para predecir el estrato socioeconómico de
# viviendas? ¿Qué métodos de otras aplicaciones se le parecen y en qué se diferencia la propuesta?
# :::
#
# ## 4.1 Protocolo
#
# Siguiendo la sección 5.2 de la guía, se consultaron **cuatro bases (Scopus, Web of Science, Google Scholar y
# arXiv)** con **seis cadenas** que combinan (i) la familia del modelo, (ii) el problema y (iii) sinónimos en inglés
# y español, todas el 23-oct-2026. La consulta de aplicación de Web of Science del Entregable 2 (19-oct-2026, 0
# resultados) se conserva en la matriz.
#
# | Cadena | Qué combina | Scopus (``TITLE-ABS-KEY``) |
# |---|---|---|
# | C1 | Problema: estrato + aprendizaje automático + catastro o vivienda | ``("socioeconomic stratum" OR "socioeconomic strata" OR "socioeconomic stratification" OR "socio-economic stratum") AND ("machine learning" OR classifier OR classification OR ensemble) AND (cadastr* OR "property data" OR housing OR dwelling*)`` |
# | C2 | Familia del modelo + problema | ``("locally weighted ensemble" OR "local ensemble weights" OR "spatially varying weights" OR "geographically weighted ensemble" OR "dynamic ensemble selection" OR "mixture of experts") AND (socioeconomic OR "social stratum" OR housing OR cadastr*)`` |
# | C3 | Regla de combinación (sin restringir la aplicación) | ``("exponentiated gradient" OR "multiplicative weights" OR "hedge algorithm") AND (ensemble OR "mixture weights" OR "classifier combination" OR "combining classifiers") AND (spatial OR geographic* OR local*)`` |
# | C4 | Ensambles geográficamente ponderados | ``"geographically weighted" AND (ensemble OR stacking OR "model averaging" OR "model combination") AND (classification OR classifier)`` |
# | C5 | Selección dinámica con información espacial | ``("dynamic ensemble selection" OR "dynamic classifier selection" OR KNORA OR "META-DES") AND (spatial OR geographic* OR "remote sensing" OR urban OR housing)`` |
# | C6 | Español | ``("estrato socioeconómico" OR "estratificación socioeconómica" OR "estratos socioeconómicos") AND ("aprendizaje automático" OR "machine learning" OR clasificación OR ensamble)`` |
#
# Las versiones para Web of Science (``TS=``), Google Scholar y arXiv (``all:``) están en
# ``literatura/PROTOCOLO_BUSQUEDA.md``; los exportes de Scopus, en ``literatura/exportes/``; el cribado registro por
# registro, con el motivo de cada decisión, en ``literatura/cribado.csv``. «Revisados» son los registros cuyo título
# y resumen se leyeron (todos si eran ≤ 50; si no, los 10 primeros por relevancia). «Relevantes» son los que aplican
# un método de la misma familia o abordan el mismo problema.
#
# ## 4.2 Matriz de novedad

# %%
from pathlib import Path

import pandas as pd

RAIZ = Path.cwd().resolve().parents[1]
matriz = pd.DataFrame([
    ("Scopus", "C1", "23-oct-2026", "10", "10", "2", "Ninguno predice el estrato de viviendas con catastro; el más cercano estima el estrato con variables del hogar (logit ordenado)"),
    ("Scopus", "C2", "23-oct-2026", "16", "16", "4", "Mezclas de expertos neuronales o para avalúo catastral (regresión); ninguna para el estrato"),
    ("Scopus", "C3", "23-oct-2026", "1", "1", "1", "Ensamble espaciotemporal en línea con gradiente exponenciado, para carga eléctrica"),
    ("Scopus", "C4", "23-oct-2026", "7", "7", "6", "GWR o ponderación espacial combinada con aprendizaje automático (otras aplicaciones)"),
    ("Scopus", "C5", "23-oct-2026", "33", "10", "7", "Selección dinámica en teledetección, tráfico, deslizamientos y clasificación desbalanceada"),
    ("Scopus", "C6", "23-oct-2026", "0", "0", "0", "—"),
    ("Web of Science", "C1", "23-oct-2026", "5", "5", "0", "Los mismos registros sin relación de Scopus C1"),
    ("Web of Science", "C2", "23-oct-2026", "13", "13", "3", "Mezclas de expertos; una con compuerta entrenada con predicciones fuera de muestra (imágenes)"),
    ("Web of Science", "C3", "23-oct-2026", "1", "1", "1", "Zeng et al. (2026), el mismo de Scopus"),
    ("Web of Science", "C4", "23-oct-2026", "7", "7", "4", "Los mismos ensambles geográficamente ponderados de Scopus"),
    ("Web of Science", "C5", "23-oct-2026", "16", "16", "13", "Selección dinámica con información espacial; ninguna para vivienda ni estrato"),
    ("Web of Science", "C6", "23-oct-2026", "0", "0", "0", "—"),
    ("Web of Science (E2)", "Aplicación", "19-oct-2026", "0", "0", "0", "TS=(estrato socioeconómico AND (datos catastrales OR características de la vivienda))"),
    ("Google Scholar", "C1", "23-oct-2026", "12", "12", "2", "El estrato aparece como variable de entrada (precios, avalúos), no como objetivo"),
    ("Google Scholar", "C2", "23-oct-2026", "13", "13", "1", "Aprendizaje automático geográficamente ponderado (salud)"),
    ("Google Scholar", "C3", "23-oct-2026", "376", "10", "4", "Pesos de ensambles por gradiente exponenciado (en línea) y una combinación convexa por regiones en regresión (Zhou y Lin, 2026)"),
    ("Google Scholar", "C4", "23-oct-2026", "5 550", "10", "4", "Rezagos espaciales y XGBoost o random forest geográficamente ponderados"),
    ("Google Scholar", "C5", "23-oct-2026", "185", "10", "2", "Selección dinámica en imágenes hiperespectrales y fusión ponderada"),
    ("Google Scholar", "C6", "23-oct-2026", "45", "10", "2", "Tesis colombianas: zonas homogéneas con IA; el estrato como entrada de modelos de precio"),
    ("arXiv", "C1", "23-oct-2026", "2", "2", "0", "Sin relación"),
    ("arXiv", "C2", "23-oct-2026", "16", "16", "0", "Mezclas de expertos neuronales (imágenes médicas, lenguaje)"),
    ("arXiv", "C3", "23-oct-2026", "0", "0", "0", "—"),
    ("arXiv", "C4", "23-oct-2026", "1", "1", "1", "Liu (2024): ensamble con ponderación espacial local y explicabilidad"),
    ("arXiv", "C5", "23-oct-2026", "0", "0", "0", "—"),
], columns=["Base", "Cadena", "Fecha", "Resultados", "Revisados", "Relevantes", "Observación"])
matriz.style.hide(axis="index")

# %% [markdown]
# ## 4.3 Trabajos más cercanos
#
# **Al problema (estrato de viviendas).** Ningún registro predice el estrato administrativo de cada vivienda a partir
# del catastro. Lo más cercano:
#
# - {cite:t}`cantillo2019` estiman el estrato como variable dependiente con un **logit ordenado** en Bogotá, Medellín,
#   Cali y Barranquilla, pero con variables del **hogar** (ingreso, tamaño, vehículos) de encuestas de movilidad, para
#   evaluar el estrato como sustituto del ingreso. No usan el catastro, no validan espacialmente ni combinan modelos.
# - {cite:t}`chicaolmo2020` evalúan si el estrato refleja la calidad de vida (logit multinivel, Medellín); el estrato
#   es el criterio que se evalúa, no la salida de un clasificador.
# - {cite:t}`vargas2022` usan XGBoost para precios de vivienda en Bogotá con el estrato como **entrada**, igual que
#   las tesis colombianas de avalúo que devuelve C6.
# - {cite:t}`arias2025` delimitan zonas homogéneas físicas con inteligencia artificial: el primer paso de la
#   metodología del DANE, no la asignación del estrato.
#
# Esto coincide con la revisión del E2 (Scopus, 9 trabajos; Web of Science, 0).
#
# **Al método.** Las familias afines son la combinación ponderada de modelos, la selección dinámica de clasificadores
# y la ponderación geográfica. Los más cercanos son:
#
# - {cite:t}`zhou2026`: una **combinación convexa por regiones** de un grupo de modelos de regresión, comparada con la
#   mejor mezcla fija, y un estimador con ajuste cruzado que anticipa cuándo esa combinación paga. Muestran que la
#   ganancia depende del tipo de desplazamiento entre entrenamiento y destino, lo mismo que encontramos con la versión
#   1 del EPL en «zonas nuevas» (capítulo 6). Es la idea más cercana: **la combinación convexa por regiones no es
#   nueva en sí misma**.
# - {cite:t}`zeng2026`, {cite:t}`jian2024`, {cite:t}`huang2026` y {cite:t}`odonncha2019`: pesos de ensambles
#   aprendidos con **gradiente exponenciado**, siempre en línea (series de tiempo o flujos de datos).
# - {cite:t}`liu2024`: ensamble con **ponderación espacial local** y explicabilidad, para regresión y clasificación.
#
# ## 4.4 Diferencias con los métodos afines

# %%
afines = pd.DataFrame([
    ("Promedio, voto o pesos por error global", "Combinación fija", "Yu et al. (2025); reglas estándar del Cuadro 1",
     "Un peso igual o proporcional al acierto global, para todo el espacio",
     "Pesos aprendidos que cambian según la región; óptimo de una pérdida convexa"),
    ("Pesos de mezcla por gradiente exponenciado", "Combinación aprendida global",
     "Helmbold et al. (1997); Zeng et al. (2026)",
     "Un vector de pesos (o pesos que cambian en el tiempo, en línea)",
     "Un vector por ancla, suavizado con un núcleo y encogido hacia los globales con KL; fuera de línea, con ajuste cruzado espacial"),
    ("Stacking lineal", "Meta-aprendiz", "Wolpert (1992)", "Regresión sin restricciones sobre las salidas de los modelos",
     "Pesos en el símplex (interpretables), dependientes de la región, con óptimo único"),
    ("Selección dinámica (KNORA, META-DES, FCH-DESA, DES-MI)", "Competencia local",
     "Ko et al. (2008); Cruz et al. (2015); Zhang et al. (2026); Aziz et al. (2025)",
     "Elige o pondera modelos por su acierto en los vecinos de cada consulta (reglas heurísticas o meta-aprendiz)",
     "Pesos suaves que minimizan una pérdida global convexa; el espacio de competencia se elige por validación"),
    ("Mezcla de expertos", "Compuerta aprendida",
     "Jacobs et al. (1991); Graczyk et al. (2010); Lasota et al. (2013)",
     "Compuerta (neuronal o gaussiana) y expertos entrenados juntos con EM; no convexo",
     "Expertos fijos ya validados; compuerta no neuronal; problema convexo con óptimo único"),
    ("Regresión o bosques geográficamente ponderados", "Modelos locales",
     "Fotheringham et al. (2002); Georganos et al. (2021); Shi et al. (2026)",
     "Reajusta los coeficientes o el bosque en cada ubicación",
     "Pondera modelos completos en lugar de reajustarlos; clasificación multiclase"),
    ("XGBoost geográfico", "Global + locales", "Grekousis (2025); E2/E3 (adaptación a clasificación)",
     "Promedio con peso fijo α entre un modelo global y modelos locales",
     "Pesos aprendidos por ancla y por experto; cualquier familia de expertos"),
    ("Ensamble con ponderación espacial local y explicabilidad", "Ponderación geográfica + XAI", "Liu (2024)",
     "Pondera localmente modelos de aprendizaje automático y explica con valores de Shapley (datos sintéticos)",
     "Regla derivada (gradiente exponenciado) con garantías; ajuste cruzado por bloques; espacio elegido por validación"),
    ("Combinación convexa por regiones", "Combinación local convexa", "Zhou y Lin (2026)",
     "Mezcla convexa distinta en cada región del espacio de entrada (regresión) y diagnóstico de cuándo supera a la mejor mezcla fija",
     "Clasificación multiclase ordinal; regiones suaves (anclas con núcleo) en lugar de particiones; encogimiento KL hacia "
     "los pesos globales; solución por gradiente exponenciado con garantías; espacio de competencia elegido en el bucle interno"),
], columns=["Método", "Familia", "Trabajos", "Qué hace", "Qué es nuevo en el EPL"])
afines.style.hide(axis="index")

# %% [markdown]
# **Qué se conserva.** Del gradiente exponenciado, la regla multiplicativa sobre el símplex; de la selección dinámica,
# la idea de que la competencia es local; de la ponderación geográfica, el núcleo y el ancho de banda.
#
# **Qué limitación aborda.** La selección dinámica define la competencia con reglas sobre vecinos en el espacio de
# variables y la evalúa con las mismas predicciones que la entrenan; la mezcla de expertos resuelve un problema no
# convexo; la ponderación geográfica reajusta modelos en cada lugar, lo que con «zonas nuevas» (sin datos cerca)
# extrapola mal. El EPL aprende los pesos con predicciones **fuera de muestra por bloques espaciales**, resuelve un
# problema **convexo con óptimo único** y deja que la validación elija **dónde** medir la competencia (geografía,
# variables o desacuerdo entre expertos).
#
# ## 4.5 Declaración de novedad
#
# :::{admonition} Declaración
# :class: important
# Al mejor de nuestro conocimiento y según la búsqueda descrita (Scopus, Web of Science, Google Scholar y arXiv, seis
# cadenas, 23-oct-2026; y Web of Science el 19-oct-2026 en el Entregable 2), **no se encontraron estudios que
# apliquen un ensamble con pesos locales aprendidos por gradiente exponenciado —ni otro ensamble con ponderación local
# o selección dinámica de clasificadores— a la predicción del estrato socioeconómico de viviendas individuales a
# partir de datos catastrales y su ubicación.** Los trabajos más cercanos al problema estiman el estrato con
# variables del hogar y un modelo único (Cantillo-García et al., 2019) o lo usan como entrada. Los más cercanos al
# método combinan modelos con gradiente exponenciado en línea (Zeng et al., 2026; Jian et al., 2024), con
# ponderación espacial (Liu, 2024) o con una combinación convexa por regiones en regresión (Zhou y Lin, 2026).
#
# Por eso la contribución se declara con precisión: **no** es la idea de pesos por región, que ya existe, sino (i) la
# regla de aprendizaje —pérdida logarítmica convexa con encogimiento KL sobre anclas suavizadas por un núcleo,
# resuelta por gradiente exponenciado con garantías (capítulo 5)—, (ii) el aprendizaje con predicciones fuera de
# muestra por bloques espaciales, (iii) la elección del espacio de competencia en el bucle interno y (iv) su
# aplicación, por primera vez, al estrato socioeconómico.
# :::
#
# **Limitaciones de la búsqueda.** Google Scholar no permite exportar y solo se revisaron los 10 primeros resultados cuando había más de 50; de C5 en Scopus se
# exportaron y revisaron 10 de 33; no se consultaron SciELO ni Redalyc. Ninguna de estas limitaciones cambia la conclusión
# para la aplicación: las cuatro bases coinciden entre sí y con la revisión del E2.
