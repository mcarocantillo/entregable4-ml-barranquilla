# -*- coding: utf-8 -*-
"""Pruebas unitarias del ensamble con pesos locales (sección 5.4 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Correr desde la raíz del repositorio:  python -m pytest -q tests
"""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.original_model import (EnsamblePesosLocales, bloques_espaciales, gradiente, gradiente_exponenciado,
                                kmedias, matriz_nucleo, perdida)


class Regla(ClassifierMixin, BaseEstimator):
    """Experto de juguete: predice la clase 1 si la columna ``columna`` es positiva (con 95 % de confianza)."""

    def __init__(self, columna=0):
        self.columna = columna

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        p1 = np.where(np.asarray(X)[:, self.columna] > 0, 0.95, 0.05)
        return np.column_stack([1 - p1, p1])


def mundo_dos_mitades(n=2000, seed=0):
    """Ciudad sintética: a la izquierda la clase depende de z; a la derecha, de w.

    Columnas de X: z, w, x_km, y_km. El experto A (mira z) acierta solo a la
    izquierda; el B (mira w) solo a la derecha. Ningún peso global supera ~75 %
    de acierto; un campo de pesos local puede acercarse al 95 %.
    """
    rng = np.random.default_rng(seed)
    z, w = rng.normal(size=n), rng.normal(size=n)
    sx, sy = rng.uniform(-5, 5, n), rng.uniform(-5, 5, n)
    y = np.where(sx < 0, z > 0, w > 0).astype(int)
    return np.column_stack([z, w, sx, sy]), y


EXPERTOS = [("A", Regla(0)), ("B", Regla(1))]


def ajustado(**kw):
    X, y = mundo_dos_mitades()
    params = dict(expertos=EXPERTOS, n_anclas=20, ancho_banda=1.0, k_anclas=4, lam=0.01, tam_bloque=1.0,
                  random_state=7)
    params.update(kw)
    return EnsamblePesosLocales(**params).fit(X, y), X, y


# (i) aprende donde la solución es conocida ---------------------------------
def test_aprende_el_campo_de_pesos_conocido():
    m, X, y = ajustado()
    Xn, yn = mundo_dos_mitades(seed=1)
    acierto = (m.predict(Xn) == yn).mean()
    assert acierto > 0.90
    W = m.pesos_locales(Xn)
    izq, der = Xn[:, 2] < -1.5, Xn[:, 2] > 1.5
    assert W[izq, 0].mean() > 0.9          # a la izquierda manda el experto A
    assert W[der, 1].mean() > 0.9          # a la derecha, el B


def test_supera_a_los_pesos_globales():
    local, X, y = ajustado()
    glob, _, _ = ajustado(n_anclas=1)
    Xn, yn = mundo_dos_mitades(seed=1)
    assert (local.predict(Xn) == yn).mean() > (glob.predict(Xn) == yn).mean() + 0.1


# (ii) reproducibilidad --------------------------------------------------------
def test_misma_semilla_mismo_modelo():
    a, X, _ = ajustado()
    b, _, _ = ajustado()
    np.testing.assert_array_equal(a.anclas_, b.anclas_)
    np.testing.assert_array_equal(a.pesos_, b.pesos_)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


# (iii) entradas inválidas -----------------------------------------------------
@pytest.mark.parametrize("cambio", [dict(n_anclas=0), dict(ancho_banda=0.0), dict(lam=-1.0), dict(n_pliegues=1),
                                    dict(peso_clase="otro"), dict(expertos=[]), dict(espacio="barrio"),
                                    dict(expertos=[("A", Regla(0)), ("A", Regla(1))])])
def test_parametros_invalidos(cambio):
    X, y = mundo_dos_mitades(200)
    params = dict(expertos=EXPERTOS)
    params.update(cambio)
    with pytest.raises(ValueError):
        EnsamblePesosLocales(**params).fit(X, y)


def test_datos_invalidos():
    X, y = mundo_dos_mitades(200)
    m = EnsamblePesosLocales(expertos=EXPERTOS)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        m.fit(Xn, y)
    with pytest.raises(ValueError):
        m.fit(X, np.zeros(len(y)))                       # una sola clase
    with pytest.raises(ValueError):
        m.fit(X, np.linspace(0, 1, len(y)))              # objetivo continuo
    with pytest.raises(ValueError):
        m.fit(X[:10], y[:20])                            # largos distintos
    m.fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :3])                              # columnas de menos


def test_predecir_sin_ajustar():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        EnsamblePesosLocales(expertos=EXPERTOS).predict(np.zeros((2, 4)))


# (iv) compatibilidad con scikit-learn ----------------------------------------
def test_clone_y_parametros():
    m = EnsamblePesosLocales(expertos=EXPERTOS, n_anclas=13, lam=0.5)
    c = clone(m)
    assert c.get_params()["n_anclas"] == 13 and c.get_params()["lam"] == 0.5
    assert not hasattr(c, "pesos_")
    a, X, _ = ajustado()
    b = clone(a).fit(*mundo_dos_mitades())
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))


def test_pipeline_y_validacion_cruzada():
    X, y = mundo_dos_mitades(600)
    expertos = [("log_z", LogisticRegression()), ("log_w", LogisticRegression())]
    m = make_pipeline(StandardScaler(), EnsamblePesosLocales(expertos=expertos, n_anclas=8, n_pliegues=3))
    s = cross_val_score(m, X, y, cv=3)
    assert np.all(np.isfinite(s)) and len(s) == 3


def test_etiquetas_no_numericas():
    X, y = mundo_dos_mitades(400)
    yt = np.where(y == 1, "alto", "bajo")
    m = EnsamblePesosLocales(expertos=EXPERTOS, n_anclas=8).fit(X, yt)
    assert set(m.predict(X)) <= {"alto", "bajo"}
    assert list(m.classes_) == ["alto", "bajo"]


# Propiedades del algoritmo ----------------------------------------------------
def test_probabilidades_y_pesos_en_el_simplex():
    m, X, _ = ajustado()
    P = m.predict_proba(X)
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-10)
    assert np.all(P >= 0)
    np.testing.assert_allclose(m.pesos_.sum(1), 1.0, atol=1e-10)
    np.testing.assert_allclose(m.pesos_locales(X).sum(1), 1.0, atol=1e-10)


def test_perdida_monotona_no_creciente():
    m, _, _ = ajustado()
    h = np.array(m.historia_)
    assert np.all(np.diff(h) <= 1e-12)
    assert h[-1] < h[0]


def test_gradiente_analitico_igual_al_numerico():
    rng = np.random.default_rng(3)
    n, J, k, M = 60, 5, 3, 4
    idx = np.stack([rng.choice(J, k, replace=False) for _ in range(n)])
    kap = rng.dirichlet(np.ones(k), n)
    Q = rng.uniform(0.05, 1, (n, M))
    a = rng.uniform(0.5, 2, n)
    pi = rng.dirichlet(np.ones(M))
    W = rng.dirichlet(np.ones(M), J)
    G = gradiente(W, idx, kap, Q, a, pi, 0.3)
    h = 1e-6
    for j, m in [(0, 0), (2, 3), (4, 1)]:
        E = np.zeros_like(W)
        E[j, m] = h
        num = (perdida(W + E, idx, kap, Q, a, pi, 0.3) - perdida(W - E, idx, kap, Q, a, pi, 0.3)) / (2 * h)
        assert abs(num - G[j, m]) < 1e-6


def test_converge_al_optimo_de_un_caso_cerrado():
    # Un ancla, dos expertos, λ = 0: el óptimo de -Σ log(w q1 + (1-w) q2) se puede hallar por búsqueda fina.
    rng = np.random.default_rng(5)
    Q = rng.uniform(0.05, 1, (300, 2))
    idx, kap, a = np.zeros((300, 1), int), np.ones((300, 1)), np.ones(300)
    W, _, _ = gradiente_exponenciado(idx, kap, Q, a, np.array([0.5, 0.5]), 0.0, 1, n_iter=2000, tol=1e-14)
    rejilla = np.linspace(1e-4, 1 - 1e-4, 100001)
    obj = [-np.log(w * Q[:, 0] + (1 - w) * Q[:, 1]).mean() for w in rejilla[::100]]
    w_opt = rejilla[::100][int(np.argmin(obj))]
    assert abs(W[0, 0] - w_opt) < 2e-3


def test_lambda_grande_lleva_a_los_pesos_globales():
    m, _, _ = ajustado(lam=1e4)
    np.testing.assert_allclose(m.pesos_, np.tile(m.pesos_globales_, (len(m.pesos_), 1)), atol=1e-3)


def test_piezas_auxiliares():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0.5, 0.1, (50, 2)), rng.normal(5.5, 0.1, (50, 2))])
    C = kmedias(Z, 2, np.random.default_rng(1))
    assert np.allclose(np.sort(C[:, 0]), [0.5, 5.5], atol=0.1)
    idx, kap = matriz_nucleo(np.array([[100.0, 100.0]]), C, h=0.5, k=2)   # muy lejos: no da 0/0
    assert np.isfinite(kap).all() and np.isclose(kap.sum(), 1)
    f = bloques_espaciales(Z, 1.0, 2, np.random.default_rng(0))
    assert len(np.unique(f[:50])) == 1 and len(np.unique(f[50:])) == 1      # una celda = un pliegue


# Ajuste en dos etapas (lo usan la búsqueda de hiperparámetros y la ablación) ---------------------
def test_fit_con_probabilidades_es_identico_a_fit():
    from src.original_model import probabilidades_fuera_de_muestra
    m, X, y = ajustado()
    Xe, S = m._partir(X)
    P = probabilidades_fuera_de_muestra(EXPERTOS, Xe, y, S, m.classes_, m.tam_bloque, m.n_pliegues,
                                        m.random_state)
    finales = [clone(e).fit(Xe, y) for _, e in EXPERTOS]
    b = clone(m).fit_con_probabilidades(X, y, P, finales)
    np.testing.assert_array_equal(m.anclas_, b.anclas_)
    np.testing.assert_allclose(m.pesos_, b.pesos_)
    np.testing.assert_allclose(m.predict_proba(X), b.predict_proba(X))


def test_fit_con_probabilidades_valida_la_forma():
    m, X, y = ajustado()
    with pytest.raises(ValueError):
        clone(m).fit_con_probabilidades(X, y, np.zeros((len(X), 3, 2)), m.expertos_)


def test_variante_sin_ajuste_cruzado():
    m, X, y = ajustado(ajuste_cruzado=False)
    assert np.isfinite(m.predict_proba(X)).all()
    np.testing.assert_allclose(m.pesos_.sum(1), 1.0)


# Versión 2: espacio de competencia ---------------------------------------------------------------
def mundo_competencia_por_variable(n=2000, seed=0):
    """La clase depende de z si c < 0 y de w si c > 0; la ubicación es ruido.

    Columnas: z, w, c (con varianza grande, para que sea la primera componente principal), x_km, y_km.
    Un campo de pesos geográfico no puede separar las dos regiones; uno en el espacio de las variables sí.
    """
    rng = np.random.default_rng(seed)
    z, w, c = rng.normal(size=n), rng.normal(size=n), 5 * rng.normal(size=n)
    sx, sy = rng.uniform(-5, 5, n), rng.uniform(-5, 5, n)
    y = np.where(c < 0, z > 0, w > 0).astype(int)
    return np.column_stack([z, w, c, sx, sy]), y


def test_espacio_variables_encuentra_la_competencia():
    X, y = mundo_competencia_por_variable()
    Xn, yn = mundo_competencia_por_variable(seed=1)
    comun = dict(expertos=EXPERTOS, n_anclas=20, ancho_banda=1.5, k_anclas=4, lam=0.01, random_state=7)
    geo = EnsamblePesosLocales(espacio="geografico", **comun).fit(X, y)
    var = EnsamblePesosLocales(espacio="variables", **comun).fit(X, y)
    acierto_geo = (geo.predict(Xn) == yn).mean()
    acierto_var = (var.predict(Xn) == yn).mean()
    assert acierto_var > 0.88 and acierto_var > acierto_geo + 0.08


@pytest.mark.parametrize("espacio", ["geografico", "variables", "probabilidades"])
def test_espacios_dan_probabilidades_validas(espacio):
    m, X, _ = ajustado(espacio=espacio)
    P = m.predict_proba(X)
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-10)
    np.testing.assert_allclose(m.pesos_locales(X).sum(1), 1.0, atol=1e-10)
    Pe = m._proba_expertos(m._partir(X)[0])
    np.testing.assert_allclose(m.predict_proba_desde_probabilidades(X, Pe), P)


def test_ajuste_sin_expertos_solo_predice_desde_probabilidades():
    from src.original_model import probabilidades_fuera_de_muestra
    m, X, y = ajustado(espacio="probabilidades")
    Xe, S = m._partir(X)
    P = probabilidades_fuera_de_muestra(EXPERTOS, Xe, y, S, m.classes_, m.tam_bloque, m.n_pliegues, m.random_state)
    b = clone(m).fit_con_probabilidades(X, y, P, None)
    Pe = m._proba_expertos(Xe)
    np.testing.assert_allclose(b.predict_proba_desde_probabilidades(X, Pe), m.predict_proba(X))
    with pytest.raises(ValueError):
        b.predict_proba(X)
