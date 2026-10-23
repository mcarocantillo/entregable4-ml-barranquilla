# -*- coding: utf-8 -*-
"""Valores de Shapley por muestreo de permutaciones para explicar una predicción (explicación local tipo SHAP).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Estimador de Štrumbelj y Kononenko (2014), el mismo que usa ``shap.PermutationExplainer``: para cada una de T
permutaciones π de las variables y una vivienda de referencia b tomada del fondo, se parte de b y se van
reemplazando sus valores por los de x en el orden de π; el cambio de la probabilidad de la clase al agregar la
variable j es su contribución en esa permutación. El promedio sobre las T permutaciones estima el valor de Shapley.

Propiedad de eficiencia (exacta, no aproximada): para cada permutación la suma de contribuciones es
f(x) − f(b) (suma telescópica), así que Σ_j φ_j = f(x) − promedio de f(b).

Se explica en el espacio de las variables originales (las del formulario), no en el de las columnas
preprocesadas, y funciona con cualquier modelo con ``predict_proba`` (los benchmarks y el EPL), sin la librería
``shap`` (que no se instala en el servidor por memoria).
"""
import numpy as np
import pandas as pd


def shapley_muestreo(predecir, x, fondo, clase, n_permutaciones=40, semilla=0):
    """Contribución de cada variable a la probabilidad de ``clase`` para la fila ``x``.

    Parámetros
    ----------
    predecir : callable
        DataFrame → ndarray (n, K) de probabilidades.
    x : pandas.DataFrame
        Una fila con las columnas del modelo.
    fondo : pandas.DataFrame
        Viviendas de referencia (se toma una al azar por permutación).
    clase : int
        Columna de la probabilidad que se explica.
    n_permutaciones : int
        T. El error de Monte Carlo baja como 1/√T.
    semilla : int
        Semilla del muestreo (la explicación es reproducible).

    Devuelve
    --------
    (dict, float, float)
        {variable: φ_j}, el valor base (promedio de f(b)) y f(x).
    """
    rng = np.random.default_rng(semilla)
    cols = list(x.columns)
    p = len(cols)
    filas = []
    ordenes = []
    for _ in range(n_permutaciones):
        orden = rng.permutation(p)
        b = fondo.iloc[rng.integers(len(fondo))].copy()
        actual = b.copy()
        filas.append(actual.copy())
        for j in orden:
            actual[cols[j]] = x.iloc[0, j]
            filas.append(actual.copy())
        ordenes.append(orden)
    lote = pd.DataFrame(filas, columns=cols)
    for c in cols:                                         # conserva los tipos de x (numéricas y categóricas)
        lote[c] = lote[c].astype(x[c].dtype) if x[c].dtype != object else lote[c]
    f = predecir(lote)[:, clase].reshape(n_permutaciones, p + 1)
    phi = np.zeros(p)
    for t, orden in enumerate(ordenes):
        phi[orden] += np.diff(f[t])
    phi /= n_permutaciones
    base = float(f[:, 0].mean())
    return dict(zip(cols, phi)), base, float(predecir(x)[0, clase])
