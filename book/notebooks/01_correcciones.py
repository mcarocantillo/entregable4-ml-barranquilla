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
# # 1. Trazabilidad de correcciones del Entregable 3
#
# :::{admonition} Qué responde este capítulo
# :class: tip
# Qué se corrigió del entregable anterior, cómo, dónde está en este libro y con qué evidencia.
# :::
#
# La tabla reúne las correcciones del Entregable 3 (la guía lo llama «Entregable 2»): para cada debilidad del libro,
# el código o los resultados del E3 —las que el propio E3 ya reconocía y las que se identificaron después— se indica
# la acción realizada, dónde está en este libro y con qué evidencia.
#
# La guía pide reportar los cambios desfavorables: varios de estos arreglos **bajan** cifras que el E3 había
# publicado (filas 2, 5 y 17).

# %%
import pandas as pd

filas = [
    ("1", "Con 5 pliegues externos ninguna prueba pareada tiene potencia (p mínimo de Wilcoxon = 0.0625) y el error "
          "estándar ingenuo sd/√5 es optimista porque los pliegues comparten entrenamiento.",
     "Validación anidada repetida 5 × 5 (25 pliegues, mismas particiones para todos los modelos) y t corregida de "
     "Nadeau y Bengio para validación repetida.", "Caps. 3 y 6", "Tabla de la sección 6.2; experimentos/01_particiones.py"),
    ("2", "El IC bootstrap del test no fijaba las 6 clases: las réplicas sin el bloque 4004 perdían los estratos 5 y 6 "
          "del F1 macro y el límite superior salía inflado (RF [0.523, 0.714]).",
     "F1 macro con labels = 1..6 en todo remuestreo; IC BCa por bloques recalculados.", "Cap. 6, sección del test",
     "experimentos/09_test_confirmatorio.py; tabla del test"),
    ("3", "El bloque de test 3002 tiene 3 viviendas (F1 = 1 para todos): con él hay 7 «bloques» pero solo 6 informan.",
     "Se reporta y se excluye de los análisis por bloque (IC y permutación).", "Cap. 6", "test_confirmatorio.json"),
    ("4", "El Friedman del E3 usó 32 bloques espaciales de la submuestra sin tamaño mínimo (había bloques de 2 a 19 "
          "viviendas).", "Friedman y Nemenyi sobre los 25 pliegues externos (≈ 2 000 viviendas cada uno).", "Cap. 6",
     "Diagrama de diferencia crítica (figura 6.1)"),
    ("5", "El E2 publicó un IC percentil [+0.034, +0.199] para el XGBoost geográfico frente al modelo base y concluyó "
          "«se confirma»; con BCa el IC es [−0.024, +0.172].", "Todos los IC en BCa; el veredicto se corrige a «no se "
          "distingue en el test».", "Caps. 3 y 6", "Tabla 3.1"),
    ("6", "Tiempos de la regresión y de la clasificación inflados (8 procesos en 4 núcleos; dos cadenas a la vez).",
     "Tiempos de entrenamiento e inferencia de los modelos finales medidos uno a la vez; complejidad del modelo "
     "original con el equipo libre.", "Caps. 3 y 5", "artifacts/metadata.json; sección 5.7"),
    ("7", "El modelo evaluado en el test salía de una segunda búsqueda no anidada y no se explicaba por qué.",
     "Se documenta: el modelo final aplica a toda la submuestra el MISMO procedimiento validado (búsqueda con el mismo "
     "método y presupuesto); la estimación del desempeño es la de la validación anidada.", "Cap. 3, sección 3.3",
     "experimentos/07_artefactos.py"),
    ("8", "Random Forest no reproducible entre corridas (hilos de BLAS sin fijar).",
     "OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1 en todos los lanzadores; prueba de ida y vuelta "
     "de cada artefacto.", "Cap. 9", "experimentos/correr_*.sh; tests/test_artefactos.py"),
    ("9", "DeLong con probabilidades fuera de muestra de 5 modelos distintos y suponiendo independencia (ICC 0.99).",
     "DeLong y McNemar solo en el test, con un único modelo final por método y con la advertencia de dependencia; "
     "se acompañan de IC y permutación por bloques.", "Cap. 6", "test_confirmatorio.json"),
    ("10", "La ablación del XGBoost geográfico comparaba espacios de búsqueda distintos (no era una ablación limpia).",
     "La ablación del modelo original retira un componente a la vez con los mismos expertos, pliegues y "
     "probabilidades.", "Cap. 5, sección 5.8", "Tabla de ablación"),
    ("11", "En el geográfico con SMOTE/ADASYN las coordenadas sin escalar entraban a las distancias del remuestreo.",
     "Documentado; el geográfico queda como referencia (no es el modelo original del E4).", "Cap. 3", "—"),
    ("12", "El XGBoost geográfico envuelve XGBClassifier: no cumple la implementación propia (R2).",
     "Modelo original nuevo (EPL) con núcleo en numpy/scipy, clase propia compatible con scikit-learn.", "Cap. 5",
     "src/original_model.py; 30 pruebas unitarias"),
    ("13", "El test de 7 bloques se reutilizó en E1, E2, E3 y las mejoras.",
     "Se declara; el veredicto R3 se apoya en la validación repetida y el test es solo confirmatorio.", "Caps. 6 y 8",
     "—"),
    ("14", "Pruebas para series de tiempo (Diebold-Mariano, Giacomini-White, MCS) aplicadas a un orden espacial sur→norte "
           "en la regresión.", "Se retiran: el problema es de corte transversal; se usan las pruebas de la columna "
           "izquierda de la Tabla 3 de la guía.", "Cap. 6", "—"),
    ("15", "Contradicciones menores de texto (10 000 frente a 10 001 viviendas; «6×» frente a «7×»; «se reproduce "
           "exactamente» cuando el F1 cambia en 0.0005).", "Corregidas en este libro.", "Caps. 2 y 9", "—"),
    ("16", "No había modelos serializados: el dashboard no podía predecir con las mejores combinaciones.",
     "artifacts/ con un joblib por modelo (pipeline completo), metadata.json y prueba de ida y vuelta.", "Caps. 3 y 7",
     "artifacts/metadata.json"),
    ("17", "Las mejores combinaciones se eligieron y se reportaron con los mismos 5 pliegues (sesgo de selección).",
     "En las 4 repeticiones nuevas todos los modelos bajan (RF 0.658 → 0.592; EPL 0.679 → 0.602); se reporta el "
     "promedio de las 25 particiones.", "Cap. 3, sección 3.2", "Tabla 3.2 (repetición 0 frente a 1–4)"),
    ("18", "El E3 no comparaba con métodos recientes del estado del arte.",
     "LightGBM y CatBoost con el mismo protocolo y presupuesto.", "Cap. 3", "experimentos/02b_recientes_5x5.py"),
]
tabla = pd.DataFrame(filas, columns=["#", "Observación", "Acción realizada", "Ubicación en el libro",
                                     "Evidencia"])
pd.set_option("display.max_colwidth", None)
tabla.style.hide(axis="index")
