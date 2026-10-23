# -*- coding: utf-8 -*-
"""Modelo original del proyecto: ensamble con pesos locales por gradiente exponenciado (EPL).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Idea
----
Varios clasificadores ya ajustados (los *expertos*) no son igual de buenos en
toda la ciudad: en zonas homogéneas gana un modelo de vecinos; en zonas mixtas,
uno con las variables físicas de la vivienda. El EPL aprende un **campo de pesos**
sobre el espacio geográfico: en cada *ancla* :math:`a_j` (centros de K-medias de
las ubicaciones de entrenamiento) hay un vector de pesos :math:`w_j` en el símplex
:math:`\\Delta_M`, y el peso de una vivienda en :math:`s` es la mezcla suavizada

.. math::  w(s) = \\sum_j \\kappa_j(s)\\, w_j,\\qquad
           \\kappa_j(s) = \\frac{e^{-\\|s-a_j\\|^2/2h^2}}{\\sum_{l\\in N_k(s)} e^{-\\|s-a_l\\|^2/2h^2}}
           \\;(j \\in N_k(s)).

La predicción es la mezcla de las probabilidades de los expertos,
:math:`p(y\\mid x, s) = \\sum_m w_m(s)\\, P_m(y\\mid x)`.

Los pesos :math:`W = (w_1,\\dots,w_J)` minimizan la log-pérdida de la mezcla en
predicciones **fuera de muestra** de los expertos (ajuste cruzado por bloques
espaciales, para no premiar al experto que memoriza), más una penalización KL
hacia los pesos globales :math:`\\pi`:

.. math::  L(W) = -\\frac{1}{\\sum_i a_i}\\sum_i a_i \\log\\Big(\\sum_j K_{ij}\\, w_j^\\top q_i\\Big)
                 + \\frac{\\lambda}{J}\\sum_j \\mathrm{KL}(w_j\\,\\|\\,\\pi),

con :math:`q_{im} = P_m(y_i\\mid x_i)` y :math:`a_i` el peso de la clase de
:math:`y_i`. El primer término es convexo en :math:`W` (menos logaritmo de una
función lineal) y el segundo es estrictamente convexo si :math:`\\lambda > 0`, de
modo que el óptimo existe y es único. Se resuelve con **gradiente exponenciado**
(descenso espejo con entropía), que mantiene cada :math:`w_j` en el símplex:

.. math::  w_{jm} \\leftarrow \\frac{w_{jm}\\, e^{-\\eta\\, g_{jm}}}{\\sum_{m'} w_{jm'}\\, e^{-\\eta\\, g_{jm'}}},\\qquad
           g_{jm} = -\\frac{1}{\\sum_i a_i}\\sum_i a_i\\frac{K_{ij}\\, q_{im}}{r_i}
                   + \\frac{\\lambda}{J}\\log\\frac{w_{jm}}{\\pi_m},

donde :math:`r_i = \\sum_j K_{ij}\\, w_j^\\top q_i`. Si una iteración no baja la
pérdida, el paso :math:`\\eta` se reduce a la mitad (por eso la pérdida es
monótona no creciente); si la acepta, el siguiente intento usa 1.5·η. Con :math:`J = 1` el método es el ensamble de pesos
globales por gradiente exponenciado (Helmbold et al., 1997), que se usa también
para calcular :math:`\\pi`.

Solo usa numpy y scipy en el núcleo (K-medias, núcleo, optimización). Los
expertos son estimadores de scikit-learn cualesquiera con ``predict_proba``.
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


# ---------------------------------------------------------------------------
# Piezas del algoritmo (funciones puras, probadas por separado)
# ---------------------------------------------------------------------------
def kmedias(Z, k, rng, n_iter=50):
    """K-medias de Lloyd con inicialización k-means++.

    Parámetros
    ----------
    Z : ndarray (n, d)
        Puntos.
    k : int
        Número de centros (si hay menos puntos distintos, se usan todos).
    rng : numpy.random.Generator
        Generador con semilla.
    n_iter : int
        Iteraciones máximas de Lloyd.

    Devuelve
    --------
    ndarray (k', d)
        Centros, con k' = min(k, puntos distintos).
    """
    Z = np.unique(np.asarray(Z, float), axis=0)
    k = min(k, len(Z))
    # k-means++: cada centro nuevo con probabilidad proporcional a la distancia² al más cercano
    C = [Z[rng.integers(len(Z))]]
    d2 = ((Z - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(Z), 1 / len(Z))
        C.append(Z[rng.choice(len(Z), p=p)])
        d2 = np.minimum(d2, ((Z - C[-1]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(n_iter):
        _, asig = cKDTree(C).query(Z)
        nuevos = np.array([Z[asig == j].mean(0) if np.any(asig == j) else C[j] for j in range(k)])
        if np.allclose(nuevos, C):
            break
        C = nuevos
    return C


def matriz_nucleo(S, anclas, h, k):
    """Pesos de núcleo gaussiano normalizados sobre las k anclas más cercanas.

    Parámetros
    ----------
    S : ndarray (n, d)
        Ubicaciones.
    anclas : ndarray (J, d)
        Centros.
    h : float
        Ancho de banda (en las unidades de ``S``).
    k : int
        Anclas vecinas que se usan por punto.

    Devuelve
    --------
    (ndarray (n, k) int, ndarray (n, k) float)
        Índices de las anclas y pesos (cada fila suma 1). Se normaliza restando
        la menor distancia² de la fila antes de exponenciar, así un punto lejos
        de todas las anclas no queda con pesos 0/0.
    """
    k = min(k, len(anclas))
    d, idx = cKDTree(anclas).query(S, k=k)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    e = np.exp(-(d ** 2 - (d ** 2).min(1, keepdims=True)) / (2 * h ** 2))
    return idx, e / e.sum(1, keepdims=True)


def mezcla_verdadera(idx, kap, W, Q):
    """r_i = Σ_j K_ij w_jᵀ q_i: probabilidad que la mezcla da a la clase observada."""
    Wi = np.einsum("nk,nkm->nm", kap, W[idx])          # pesos de cada punto (n, M)
    return (Wi * Q).sum(1), Wi


def perdida(W, idx, kap, Q, a, pi, lam):
    """Objetivo L(W): log-pérdida ponderada de la mezcla + (λ/J) Σ_j KL(w_j || π)."""
    r, _ = mezcla_verdadera(idx, kap, W, Q)
    datos = -(a * np.log(r)).sum() / a.sum()
    if lam == 0:
        return datos
    kl = (W * (np.log(W) - np.log(pi))).sum()
    return datos + lam / len(W) * kl


def gradiente(W, idx, kap, Q, a, pi, lam):
    """∂L/∂W (J, M), acumulando la contribución de cada punto a sus k anclas."""
    r, _ = mezcla_verdadera(idx, kap, W, Q)
    c = (a / r / a.sum())[:, None, None] * kap[:, :, None] * Q[:, None, :]   # (n, k, M)
    G = np.zeros_like(W)
    np.add.at(G, idx.ravel(), -c.reshape(-1, W.shape[1]))
    if lam > 0:
        G += lam / len(W) * (np.log(W) - np.log(pi) + 1.0)
    return G


def gradiente_exponenciado(idx, kap, Q, a, pi, lam, J, eta=1.0, n_iter=300, tol=1e-7, W0=None):
    """Minimiza L(W) sobre el producto de símplex con gradiente exponenciado y paso que se reduce a la mitad.

    Parámetros
    ----------
    idx, kap : ndarray (n, k)
        Anclas vecinas y pesos de núcleo (salida de ``matriz_nucleo``).
    Q : ndarray (n, M)
        Probabilidad fuera de muestra que cada experto da a la clase observada.
    a : ndarray (n,)
        Peso de cada observación (clase balanceada o 1).
    pi : ndarray (M,)
        Pesos globales (objetivo de la penalización KL).
    lam : float
        Fuerza de la penalización.
    J : int
        Número de anclas.
    eta : float
        Paso inicial.
    n_iter : int
        Iteraciones máximas.
    tol : float
        Parada: mejora relativa de la pérdida menor que ``tol``.
    W0 : ndarray (J, M), optional
        Punto de partida (por defecto, π en cada ancla).

    Devuelve
    --------
    (ndarray (J, M), list[float], int)
        Pesos, historia de la pérdida (una por iteración aceptada, incluida la
        inicial) e iteraciones usadas.
    """
    M = Q.shape[1]
    W = np.tile(pi, (J, 1)) if W0 is None else W0.copy()
    L = perdida(W, idx, kap, Q, a, pi, lam)
    historia = [L]
    t = 0
    for t in range(1, n_iter + 1):
        G = gradiente(W, idx, kap, Q, a, pi, lam)
        while True:
            # paso multiplicativo en escala logarítmica (estable: se resta el máximo por fila)
            lw = np.log(W) - eta * G
            lw -= lw.max(1, keepdims=True)
            Wn = np.exp(lw)
            Wn /= Wn.sum(1, keepdims=True)
            Wn = np.maximum(Wn, 1e-12)
            Wn /= Wn.sum(1, keepdims=True)
            Ln = perdida(Wn, idx, kap, Q, a, pi, lam)
            if Ln <= L or eta < 1e-8:
                break
            eta /= 2
        mejora = (L - Ln) / max(abs(L), 1e-12)
        W, L = Wn, Ln
        historia.append(L)
        if mejora < tol:
            break
        eta *= 1.5      # paso aceptado: se prueba uno más largo en la siguiente iteración
    assert W.shape[1] == M
    return W, historia, t


def bloques_espaciales(S, tam, n_pliegues, rng):
    """Pliegues del ajuste cruzado: celdas cuadradas de lado ``tam`` repartidas al azar entre pliegues.

    Todas las viviendas de una misma celda caen en el mismo pliegue, de modo que
    la probabilidad fuera de muestra de un experto se calcula sin ver a los
    vecinos inmediatos (si no, un experto de vecinos parecería casi perfecto).
    """
    celda = np.floor(np.asarray(S, float) / tam).astype(np.int64)
    _, cod = np.unique(celda, axis=0, return_inverse=True)
    cod = cod.ravel()
    n_celdas = cod.max() + 1
    pliegue_celda = rng.permutation(n_celdas) % n_pliegues
    return pliegue_celda[cod]


def proba_completa(est, X, clases):
    """predict_proba con una columna por cada clase de ``clases`` (un experto puede no haber visto todas)."""
    P = np.zeros((len(X), len(clases)))
    P[:, np.searchsorted(clases, est.classes_)] = est.predict_proba(X)
    return P


def generadores(random_state):
    """Dos generadores independientes derivados de la semilla: (pliegues del ajuste cruzado, K-medias).

    Separarlos hace que las anclas no dependan de cuántos números aleatorios consumió el ajuste cruzado, así que
    ``fit`` y ``fit_con_probabilidades`` (con las mismas probabilidades) dan el mismo modelo.
    """
    a, b = np.random.SeedSequence(random_state).spawn(2)
    return np.random.default_rng(a), np.random.default_rng(b)


def probabilidades_fuera_de_muestra(expertos, Xe, y, S, clases, tam_bloque=1.0, n_pliegues=5, random_state=42):
    """Probabilidades de cada experto para cada fila, con el experto entrenado SIN el bloque espacial de esa fila.

    Parámetros
    ----------
    expertos : list of (str, estimator)
        Expertos sin ajustar (se clonan).
    Xe : ndarray (n, p)
        Variables que reciben los expertos.
    y : ndarray (n,)
        Etiquetas originales.
    S : ndarray (n, d)
        Ubicaciones (definen los bloques).
    clases : ndarray (K,)
        Clases ordenadas (columnas de la salida).
    tam_bloque, n_pliegues, random_state
        Lado de las celdas, número de pliegues y semilla (la misma del estimador).

    Devuelve
    --------
    ndarray (n, M, K)
    """
    rng_pliegues, _ = generadores(random_state)
    pliegue = bloques_espaciales(S, tam_bloque, n_pliegues, rng_pliegues)
    P = np.zeros((len(Xe), len(expertos), len(clases)))
    for f in np.unique(pliegue):
        tr, ev = pliegue != f, pliegue == f
        for m, (_, est) in enumerate(expertos):
            P[ev, m] = proba_completa(clone(est).fit(Xe[tr], y[tr]), Xe[ev], clases)
    return P


ESPACIOS = ("geografico", "variables", "probabilidades")
DIAGONAL = 20.0      # tamaño al que se llevan los espacios no geográficos (≈ la ciudad en km)


def coordenadas_crudas(espacio, Xe, S, P):
    """Coordenadas de cada fila en el espacio de competencia, antes de proyectar o escalar.

    - ``geografico``: la ubicación ``S`` (km).
    - ``variables``: las variables de los expertos ``Xe`` (se proyectan luego a 2 componentes principales).
    - ``probabilidades``: (clase esperada según la mezcla uniforme de los expertos, desviación de la clase
      esperada entre expertos). La primera ubica la fila en la escala ordinal; la segunda mide el desacuerdo,
      la información que usa la selección dinámica de ensambles para decidir a quién creerle.
    """
    if espacio == "geografico":
        return np.asarray(S, float)
    if espacio == "variables":
        return np.asarray(Xe, float)
    if espacio == "probabilidades":
        P = np.asarray(P, float)
        e = P @ np.arange(P.shape[2])                                  # (n, M) clase esperada por experto
        return np.column_stack([e.mean(1), e.std(1)])
    raise ValueError(f"espacio debe ser uno de {ESPACIOS}; se recibió {espacio!r}.")


def ajustar_transformacion(espacio, Z):
    """Proyección (PCA a 2 dimensiones en ``variables``) y escala aprendidas con los datos de entrenamiento.

    En los espacios no geográficos, la caja entre los percentiles 1 y 99 de cada coordenada se lleva a una
    diagonal de ``DIAGONAL`` unidades, para que el ancho de banda y el número de anclas signifiquen lo mismo en
    los tres espacios. El geográfico se deja en km (el modelo de la versión 1 queda igual).
    """
    if espacio == "geografico":
        return {"media": None, "componentes": None, "escala": 1.0}
    media, comp = None, None
    if espacio == "variables":
        media = Z.mean(0)
        _, _, Vt = np.linalg.svd(Z - media, full_matrices=False)       # PCA por SVD (numpy)
        comp = Vt[:2].T
        Z = (Z - media) @ comp
    lo, hi = np.percentile(Z, 1, axis=0), np.percentile(Z, 99, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return {"media": media, "componentes": comp, "escala": DIAGONAL / max(diag, 1e-12)}


def aplicar_transformacion(t, Z):
    """Aplica la proyección y la escala aprendidas en ``ajustar_transformacion``."""
    if t["componentes"] is not None:
        Z = (Z - t["media"]) @ t["componentes"]
    return Z * t["escala"]


# ---------------------------------------------------------------------------
# Estimador compatible con scikit-learn
# ---------------------------------------------------------------------------
class EnsamblePesosLocales(ClassifierMixin, BaseEstimator):
    """Ensamble de expertos con un campo de pesos local aprendido por gradiente exponenciado.

    Parámetros
    ----------
    expertos : list of (str, estimator)
        Clasificadores con ``predict_proba``. Se clonan; no se modifican.
    columnas_ubicacion : tuple of int
        Columnas de ``X`` con la ubicación (por defecto, las dos últimas: x e y en km).
    expertos_usan_ubicacion : bool
        Si False, las columnas de ubicación se quitan antes de pasar ``X`` a los
        expertos (útil cuando ya vienen estandarizadas en otras columnas).
    n_anclas : int
        J, número de anclas (centros de K-medias de las ubicaciones). 1 = pesos globales.
    ancho_banda : float
        h, ancho de banda del núcleo gaussiano (unidades de la ubicación).
    k_anclas : int
        Anclas vecinas por punto.
    espacio : {"geografico", "variables", "probabilidades"}
        Espacio de competencia donde se ubican las anclas y se miden las distancias (versión 2): la ubicación,
        las 2 primeras componentes principales de las variables de los expertos, o (clase esperada, desacuerdo
        entre expertos). Las probabilidades de entrenamiento son las fuera de muestra. Se elige en el bucle
        interno como cualquier otro hiperparámetro. Las celdas del ajuste cruzado siempre son geográficas.
    lam : float
        λ, fuerza de la penalización KL hacia los pesos globales.
    eta : float
        Paso inicial del gradiente exponenciado.
    n_iter : int
        Iteraciones máximas.
    tol : float
        Parada por mejora relativa de la pérdida.
    n_pliegues : int
        Pliegues del ajuste cruzado de los expertos.
    tam_bloque : float
        Lado de las celdas del ajuste cruzado (unidades de la ubicación).
    ajuste_cruzado : bool
        Si False, los pesos se aprenden con las probabilidades de los expertos
        sobre sus propios datos de entrenamiento (solo para la ablación: premia
        al experto que memoriza).
    peso_clase : {None, "balanced"}
        Peso de cada observación en la pérdida.
    eps : float
        Piso de las probabilidades de los expertos (acota el gradiente por 1/eps).
    random_state : int
        Semilla de los pliegues del ajuste cruzado y de K-medias (dos generadores
        independientes). La semilla de cada experto va en su propio estimador.

    Atributos
    ---------
    classes_ : ndarray
        Clases observadas.
    expertos_ : list
        Expertos reajustados con todos los datos.
    anclas_ : ndarray (J, d)
        Centros.
    pesos_ : ndarray (J, M)
        Pesos de cada ancla (filas en el símplex).
    pesos_globales_ : ndarray (M,)
        π.
    historia_ : list of float
        Pérdida por iteración (curva de convergencia).
    n_iter_ : int
        Iteraciones usadas.
    """

    def __init__(self, expertos, columnas_ubicacion=(-2, -1), expertos_usan_ubicacion=True, n_anclas=40,
                 ancho_banda=1.0, k_anclas=6, espacio="geografico", lam=0.1, eta=1.0, n_iter=300, tol=1e-7,
                 n_pliegues=5,
                 tam_bloque=1.0, ajuste_cruzado=True, peso_clase="balanced", eps=1e-4, random_state=42):
        self.expertos = expertos
        self.columnas_ubicacion = columnas_ubicacion
        self.expertos_usan_ubicacion = expertos_usan_ubicacion
        self.n_anclas = n_anclas
        self.ancho_banda = ancho_banda
        self.k_anclas = k_anclas
        self.espacio = espacio
        self.lam = lam
        self.eta = eta
        self.n_iter = n_iter
        self.tol = tol
        self.n_pliegues = n_pliegues
        self.tam_bloque = tam_bloque
        self.ajuste_cruzado = ajuste_cruzado
        self.peso_clase = peso_clase
        self.eps = eps
        self.random_state = random_state

    # -- utilidades -------------------------------------------------------
    def _validar_parametros(self):
        if not self.expertos:
            raise ValueError("Se necesita al menos un experto.")
        nombres = [n for n, _ in self.expertos]
        if len(set(nombres)) != len(nombres):
            raise ValueError("Los nombres de los expertos deben ser distintos.")
        for nombre, valor, minimo in [("n_anclas", self.n_anclas, 1), ("k_anclas", self.k_anclas, 1),
                                      ("n_iter", self.n_iter, 1), ("n_pliegues", self.n_pliegues, 2)]:
            if int(valor) != valor or valor < minimo:
                raise ValueError(f"{nombre} debe ser un entero ≥ {minimo}; se recibió {valor!r}.")
        for nombre, valor in [("ancho_banda", self.ancho_banda), ("eta", self.eta),
                              ("tam_bloque", self.tam_bloque), ("eps", self.eps)]:
            if not valor > 0:
                raise ValueError(f"{nombre} debe ser positivo; se recibió {valor!r}.")
        if self.lam < 0:
            raise ValueError(f"lam debe ser ≥ 0; se recibió {self.lam!r}.")
        if self.peso_clase not in (None, "balanced"):
            raise ValueError("peso_clase debe ser None o 'balanced'.")
        if self.espacio not in ESPACIOS:
            raise ValueError(f"espacio debe ser uno de {ESPACIOS}; se recibió {self.espacio!r}.")

    def _partir(self, X):
        cols = np.arange(X.shape[1])[list(self.columnas_ubicacion)]
        S = X[:, cols]
        Xe = X if self.expertos_usan_ubicacion else np.delete(X, cols, axis=1)
        return Xe, S

    def _proba_expertos(self, Xe):
        return np.stack([proba_completa(e, Xe, self.classes_) for e in self.expertos_], axis=1)  # (n, M, K)

    def _preparar(self, X, y):
        self._validar_parametros()
        X, y = check_X_y(X, y, ensure_all_finite=True)
        if type_of_target(y) not in ("binary", "multiclass"):
            raise ValueError("y debe ser de clasificación (binaria o multiclase).")
        self.classes_, yc = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("y debe tener al menos dos clases.")
        self.n_features_in_ = X.shape[1]
        return X, y, yc

    # -- API de scikit-learn ---------------------------------------------
    def fit(self, X, y):
        """Ajuste cruzado de los expertos, expertos finales, anclas, pesos globales y campo de pesos local.

        Pasos: (1) probabilidades fuera de muestra de cada experto por bloques
        espaciales (o, si ``ajuste_cruzado=False``, las de los expertos finales
        sobre sus propios datos: variante de ablación); (2) expertos reajustados
        con todos los datos; (3) anclas por K-medias; (4) π con J = 1; (5) W con
        J anclas, partiendo de π.
        """
        X, y, yc = self._preparar(X, y)
        Xe, S = self._partir(X)
        expertos_ = [clone(est).fit(Xe, y) for _, est in self.expertos]
        if self.ajuste_cruzado:
            P_oof = probabilidades_fuera_de_muestra(self.expertos, Xe, y, S, self.classes_, self.tam_bloque,
                                                    int(self.n_pliegues), self.random_state)
        else:
            P_oof = np.stack([proba_completa(e, Xe, self.classes_) for e in expertos_], axis=1)
        return self._combinar(Xe, S, yc, P_oof, expertos_)

    def fit_con_probabilidades(self, X, y, P_oof, expertos_ajustados):
        """Ajusta solo el combinador, con probabilidades fuera de muestra y expertos finales ya calculados.

        Da exactamente el mismo modelo que ``fit`` si ``P_oof`` salió de
        ``probabilidades_fuera_de_muestra`` con los mismos datos, bloques y
        semilla. Sirve para afinar J, h, k, λ, el espacio y los pesos de clase
        (y para la ablación) sin volver a entrenar los expertos. Con
        ``expertos_ajustados=None`` el modelo solo puede predecir con
        ``predict_proba_desde_probabilidades``.
        """
        X, y, yc = self._preparar(X, y)
        Xe, S = self._partir(X)
        P_oof = np.asarray(P_oof, float)
        if P_oof.shape != (len(X), len(self.expertos), len(self.classes_)):
            raise ValueError(f"P_oof debe tener forma {(len(X), len(self.expertos), len(self.classes_))}.")
        return self._combinar(Xe, S, yc, P_oof, None if expertos_ajustados is None else list(expertos_ajustados))

    def _combinar(self, Xe, S_geo, yc, P_oof, expertos_):
        n, K, M = len(S_geo), len(self.classes_), len(self.expertos)
        _, rng_anclas = generadores(self.random_state)
        self.expertos_ = expertos_
        Z = coordenadas_crudas(self.espacio, Xe, S_geo, P_oof)
        self.transformacion_ = ajustar_transformacion(self.espacio, Z)
        S = aplicar_transformacion(self.transformacion_, Z)               # coordenadas de competencia
        Q = np.maximum(P_oof[np.arange(n), :, yc], self.eps)              # (n, M): prob. de la clase observada

        # pesos por clase
        if self.peso_clase == "balanced":
            frec = np.bincount(yc, minlength=K)
            a = (n / (K * np.maximum(frec, 1)))[yc]
        else:
            a = np.ones(n)

        # (4) pesos globales: J = 1, sin penalización
        uno_idx, uno_kap = np.zeros((n, 1), int), np.ones((n, 1))
        uniforme = np.full(M, 1 / M)
        pi, hist_g, _ = gradiente_exponenciado(uno_idx, uno_kap, Q, a, uniforme, 0.0, 1, self.eta, self.n_iter,
                                               self.tol)
        self.pesos_globales_ = pi[0]

        # (3) y (5) anclas y campo de pesos
        self.anclas_ = kmedias(S, int(self.n_anclas), rng_anclas)
        idx, kap = matriz_nucleo(S, self.anclas_, self.ancho_banda, int(self.k_anclas))
        J = len(self.anclas_)
        self.pesos_, self.historia_, self.n_iter_ = gradiente_exponenciado(
            idx, kap, Q, a, self.pesos_globales_, self.lam, J, self.eta, self.n_iter, self.tol)
        self.historia_global_ = hist_g
        self.nombres_expertos_ = [nm for nm, _ in self.expertos]
        self.perdida_oof_expertos_ = (-(a[:, None] * np.log(Q)).sum(0) / a.sum()).tolist()
        return self

    def _validar_X(self, X):
        check_is_fitted(self, "pesos_")
        X = check_array(X, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tiene {X.shape[1]} columnas; el modelo se ajustó con {self.n_features_in_}.")
        return X

    def _pesos_desde(self, Xe, S_geo, P):
        S = aplicar_transformacion(self.transformacion_, coordenadas_crudas(self.espacio, Xe, S_geo, P))
        idx, kap = matriz_nucleo(S, self.anclas_, self.ancho_banda, int(self.k_anclas))
        return np.einsum("nk,nkm->nm", kap, self.pesos_[idx])

    def pesos_locales(self, X):
        """Pesos w(s) de cada experto para cada fila de X (n, M); útiles para mapas e interpretación."""
        X = self._validar_X(X)
        Xe, S = self._partir(X)
        P = self._proba_expertos(Xe) if self.espacio == "probabilidades" else None
        return self._pesos_desde(Xe, S, P)

    def predict_proba_desde_probabilidades(self, X, P):
        """Mezcla con probabilidades de los expertos ya calculadas para X (n, M, K); igual a ``predict_proba``."""
        X = self._validar_X(X)
        P = np.asarray(P, float)
        if P.shape != (len(X), len(self.expertos), len(self.classes_)):
            raise ValueError(f"P debe tener forma {(len(X), len(self.expertos), len(self.classes_))}.")
        Xe, S = self._partir(X)
        return np.einsum("nm,nmk->nk", self._pesos_desde(Xe, S, P), P)

    def predict_proba(self, X):
        """Mezcla Σ_m w_m(s) P_m(y | x) de las probabilidades de los expertos."""
        X = self._validar_X(X)
        if self.expertos_ is None:
            raise ValueError("Ajustado sin expertos: use predict_proba_desde_probabilidades.")
        Xe, _ = self._partir(X)
        return self.predict_proba_desde_probabilidades(X, self._proba_expertos(Xe))

    def predict(self, X):
        """Clase con mayor probabilidad de la mezcla."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
