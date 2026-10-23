# -*- coding: utf-8 -*-
"""Pruebas del estimador de valores de Shapley por permutaciones (src/explicacion.py)."""
import numpy as np
import pandas as pd

from src.explicacion import shapley_muestreo


def modelo_lineal(df):
    """Probabilidad de la clase 1 = sigmoide(2·a − b); la variable c no influye."""
    z = 2 * df["a"].to_numpy(float) - df["b"].to_numpy(float)
    p1 = 1 / (1 + np.exp(-z))
    return np.column_stack([1 - p1, p1])


def datos():
    rng = np.random.default_rng(0)
    fondo = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "c": rng.normal(size=50)})
    x = pd.DataFrame({"a": [1.5], "b": [-0.5], "c": [3.0]})
    return fondo, x


def test_eficiencia_exacta():
    fondo, x = datos()
    phi, base, fx = shapley_muestreo(modelo_lineal, x, fondo, clase=1, n_permutaciones=30)
    assert abs(sum(phi.values()) - (fx - base)) < 1e-10


def test_variable_irrelevante_tiene_aporte_cero():
    fondo, x = datos()
    phi, _, _ = shapley_muestreo(modelo_lineal, x, fondo, clase=1, n_permutaciones=30)
    assert phi["c"] == 0.0
    assert phi["a"] > abs(phi["b"]) > 0          # a pesa el doble que b y x_a está más lejos del fondo


def test_reproducible():
    fondo, x = datos()
    assert shapley_muestreo(modelo_lineal, x, fondo, 1, semilla=3) == shapley_muestreo(modelo_lineal, x, fondo, 1,
                                                                                         semilla=3)
