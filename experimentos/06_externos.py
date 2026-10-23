# -*- coding: utf-8 -*-
"""Evaluación en tres conjuntos de datos adicionales (Cuadro 1 de la guía: «evaluación amplia y justa»).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Misma tarea (clasificación multiclase ordinal, etiquetas 1..K con K ≤ 6) y naturaleza distinta:

| Conjunto | Objetivo (clases) | Ubicación que usa el EPL | Pliegues |
|---|---|---|---|
| King County, Seattle (OpenML 42092) | grado de construcción: ≤6, 7, 8, 9, 10, ≥11 (6) | lat/long → km | por celdas espaciales |
| California, censo de 1990 (OpenML 537) | valor mediano de la vivienda en sextiles (6) | lat/long → km | por celdas espaciales |
| Vino tinto «vinho verde» (OpenML 40691) | calidad: ≤4, 5, 6, ≥7 (4) | plano de 2 componentes principales | estratificados |

Protocolo (el del problema principal, más liviano): submuestra estratificada de 5 000 filas en King County y
California (el vino usa sus 1 599), 3 repeticiones × 5 pliegues externos × 3 internos, semilla 42; los mismos 9
rivales (7 modelos de la guía con su mejor combinación del E3, LightGBM y CatBoost) con 12 evaluaciones por
búsqueda; el EPL con sus 7 expertos, su búsqueda bayesiana de 12 evaluaciones y las tres reglas de contraste.
En King County se excluye el precio de venta (equivale a un avalúo, que el DANE tampoco usa).

Las coordenadas de cada conjunto se reescalan para que la diagonal de su recuadro mida 20 unidades (≈ Barranquilla
en km): así el espacio de búsqueda del ancho de banda y las celdas del ajuste cruzado (1 unidad) son comparables.

Uso (desde la raíz):  python -m experimentos.06_externos        (N_TRABAJOS=6; RAPIDO=1 = prueba de humo)
"""
import importlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from experimentos import comun as K
from src.original_model import probabilidades_fuera_de_muestra, proba_completa

E = importlib.import_module("experimentos.03_epl_5x5")       # modelo_epl y sugerir (mismo espacio de búsqueda)
N_TRABAJOS = int(os.environ.get("N_TRABAJOS", 6))
RAPIDO = os.environ.get("RAPIDO") == "1"
N_REP = 1 if RAPIDO else 3
N_SUB = 1200 if RAPIDO else 5000
PLIEGUES = [0] if RAPIDO else list(range(5))      # la prueba de humo corre un solo pliegue externo
DIAGONAL = 20.0
TAM_CELDA = 1.5          # celdas de los pliegues espaciales (unidades reescaladas)
CONJUNTOS = ["king_county", "california", "vino_tinto"]
RES = K.RAIZ / "resultados" / ("externos_rapido" if RAPIDO else "externos")


# ---------------------------------------------------------------------------
# Preparación de cada conjunto
# ---------------------------------------------------------------------------
def a_km(lat, lon):
    """Proyección equirectangular local y reescalado a una diagonal de 20 unidades."""
    lat0, lon0 = np.mean(lat), np.mean(lon)
    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110.57
    diag = np.hypot(np.ptp(x), np.ptp(y))
    return x * DIAGONAL / diag, y * DIAGONAL / diag


def preparar(nombre):
    """DataFrame con variables, ``clase`` (1..K), ubicación (si hay) y la descripción de columnas."""
    df = pd.read_csv(K.RAIZ / "data" / "externos" / nombre / "datos.csv")
    if nombre == "king_county":
        clase = np.clip(df.grade, 6, 11) - 5
        x, y = a_km(df.lat.to_numpy(), df.long.to_numpy())
        X = df[["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition",
                "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "sqft_living15", "sqft_lot15"]].copy()
        X["zipcode"] = df.zipcode.astype(str)
        num_log = ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15"]
        cat = ["zipcode"]
    elif nombre == "california":
        clase = pd.qcut(df.median_house_value, 6, labels=False) + 1
        x, y = a_km(df.latitude.to_numpy(), df.longitude.to_numpy())
        X = df[["median_income", "housing_median_age", "total_rooms", "total_bedrooms", "population",
                "households"]].copy()
        num_log = ["median_income", "total_rooms", "total_bedrooms", "population", "households"]
        cat = []
    elif nombre == "vino_tinto":
        clase = np.clip(df["class"], 4, 7) - 3
        x = y = None
        X = df.drop(columns="class").copy()
        num_log = ["residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "sulphates"]
        cat = []
    else:
        raise ValueError(nombre)
    espacial = x is not None
    if espacial:
        X["x_km"], X["y_km"] = x, y
    num = [c for c in X.columns if c not in cat]
    X["clase"] = np.asarray(clase, int)
    datos = {"num": num, "cat": cat, "num_log": [c for c in num_log if c in num],
             "dec": {"winsorizacion_cuantiles": {"valor": [0.001, 0.999]}, "onehot_min_frecuencia": {"valor": 0.01}},
             "M1": K.CF.importar_e1()[1], "espacial": espacial}
    return X, datos


def submuestra(df, n, seed=K.SEED):
    if len(df) <= n:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    frac = n / len(df)
    partes = [g.iloc[rng.permutation(len(g))[:max(1, int(round(len(g) * frac)))]] for _, g in df.groupby("clase")]
    return pd.concat(partes).sample(frac=1, random_state=seed).reset_index(drop=True)


def particiones(df, espacial):
    """3 repeticiones × 5 externos (por celdas o estratificados) y 3 internos dentro de cada entrenamiento."""
    from e1.particion import folds_estratificados_por_grupo
    y = df.clase.to_numpy()
    reps = []
    for r in range(N_REP):
        semilla = K.SEED + r
        if espacial:
            celda = (np.floor(df.x_km / TAM_CELDA).astype(int).astype(str) + "_" +
                     np.floor(df.y_km / TAM_CELDA).astype(int).astype(str)).to_numpy()
            ext = folds_estratificados_por_grupo(y, celda, 5, semilla)
            internos = [folds_estratificados_por_grupo(y[itr], celda[itr], 3, semilla) for itr, _ in ext]
        else:
            idx = np.arange(len(y))
            ext = list(StratifiedKFold(5, shuffle=True, random_state=semilla).split(idx, y))
            internos = [list(StratifiedKFold(3, shuffle=True, random_state=semilla).split(itr, y[itr]))
                        for itr, _ in ext]
        reps.append({"semilla": semilla, "externos": ext, "internos": internos})
    return reps


_CACHE = {}


def cargar(nombre):
    """(df submuestreado, datos, particiones), guardado en disco para que todos los procesos usen lo mismo."""
    if nombre in _CACHE:
        return _CACHE[nombre]
    import pickle
    ruta = RES / nombre / "paquete.pkl"
    if ruta.exists():
        with open(ruta, "rb") as fh:
            df, datos, parts = pickle.load(fh)
        datos["M1"] = K.CF.importar_e1()[1]
    else:
        df, datos = preparar(nombre)
        df = submuestra(df, N_SUB)
        parts = particiones(df, datos["espacial"])
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as fh:
            pickle.dump((df, {k: v for k, v in datos.items() if k != "M1"}, parts), fh)
    _CACHE[nombre] = (df, datos, parts)
    return _CACHE[nombre]


# ---------------------------------------------------------------------------
# Tareas
# ---------------------------------------------------------------------------
def ruta_b(nombre, modelo, r, f, ext="json"):
    return RES / nombre / "benchmarks" / f"{K.slug(modelo)}_r{r}_f{f}.{ext}"


def ruta_e(nombre, r, f, ext="json"):
    return RES / nombre / "epl" / f"r{r}_f{f}.{ext}"


def tarea_benchmark(nombre, modelo, balanceo, metodo, r, f):
    if ruta_b(nombre, modelo, r, f).exists():
        return nombre, modelo, r, f, json.loads(ruta_b(nombre, modelo, r, f).read_text(encoding="utf-8"))["f1"]
    warnings.filterwarnings("ignore")
    from estrato3 import evaluacion as EV
    from estrato3 import modelos as MD
    from estrato3 import optimizacion as OP
    K.CF.fijar_semillas()
    df, datos, parts = cargar(nombre)
    y = df.clase.to_numpy()
    itr, iva = parts[r]["externos"][f]
    cols = MD.columnas(datos)
    obj = OP.Objetivo(datos, df.iloc[itr][cols], y[itr], parts[r]["internos"][f], modelo, balanceo)
    t0 = time.perf_counter()
    OP.METODOS[metodo](obj, modelo, presupuesto=K.CF.PRESUPUESTO)
    params, f1_int = obj.mejor()
    t1 = time.perf_counter()
    est = MD.pipeline(datos, modelo, params, balanceo).fit(df.iloc[itr][cols], y[itr])
    t2 = time.perf_counter()
    proba = EV.proba_completa(est, df.iloc[iva][cols])
    t3 = time.perf_counter()
    f1 = K.f1_macro(y[iva], K.pred(proba))
    salida = {"conjunto": nombre, "modelo": modelo, "balanceo": balanceo, "metodo": metodo, "repeticion": r,
              "fold": f, "params": params, "f1_interno_mejor": f1_int, "f1": f1,
              "accuracy": float(np.mean(K.pred(proba) == y[iva])),
              "mae_ordinal": float(np.mean(np.abs(K.pred(proba) - y[iva]))),
              "seg_busqueda": t1 - t0, "seg_ajuste_final": t2 - t1, "seg_inferencia": t3 - t2,
              "n_train": len(itr), "n_val": len(iva)}
    ruta_b(nombre, modelo, r, f).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta_b(nombre, modelo, r, f, "npz"), proba=proba.astype(np.float32), iva=iva)
    K.escribir_json(ruta_b(nombre, modelo, r, f), salida)
    return nombre, modelo, r, f, f1


def matrices(df, datos, filas_tr, filas_ev):
    """[X preprocesada | ubicación] de entrenamiento y evaluación (preprocesador y PCA ajustados con filas_tr)."""
    from estrato3 import modelos as MD
    cols = MD.columnas(datos)
    prep = MD.preprocesador(datos).fit(df.iloc[filas_tr][cols])
    A, B = (prep.transform(df.iloc[f][cols]).astype(float) for f in (filas_tr, filas_ev))
    if datos["espacial"]:
        Sa, Sb = (df.iloc[f][["x_km", "y_km"]].to_numpy(float) for f in (filas_tr, filas_ev))
    else:
        pca = PCA(2, random_state=K.SEED).fit(A)
        Sa, Sb = pca.transform(A), pca.transform(B)
        esc = DIAGONAL / np.hypot(*np.ptp(Sa, axis=0))
        Sa, Sb = Sa * esc, Sb * esc
    return np.column_stack([A, Sa]), np.column_stack([B, Sb])


def etapa_expertos(expertos, Xa, ya, clases):
    P = probabilidades_fuera_de_muestra(expertos, Xa[:, :-2], ya, Xa[:, -2:], clases, E.TAM_BLOQUE_KM,
                                        E.N_PLIEGUES_CRUZADOS, K.SEED)
    return P, [clone(e).fit(Xa[:, :-2], ya) for _, e in expertos]


def tarea_epl(nombre, r, f):
    if ruta_e(nombre, r, f).exists():
        return nombre, r, f, json.loads(ruta_e(nombre, r, f).read_text(encoding="utf-8"))["f1"]["EPL"]
    warnings.filterwarnings("ignore")
    import optuna
    K.CF.fijar_semillas()
    df, datos, parts = cargar(nombre)
    y = df.clase.to_numpy()
    clases = np.unique(y)
    itr, iva = parts[r]["externos"][f]
    combos = K.mejores_combinaciones(incluir_extra=False)
    expertos = [(c.modelo, K.experto(c.modelo, c.balanceo, json.loads(
        ruta_b(nombre, c.modelo, r, f).read_text(encoding="utf-8"))["params"])) for _, c in combos.iterrows()]
    t0 = time.perf_counter()
    cache = []
    for tr_rel, va_rel in parts[r]["internos"][f]:
        ftr, fva = itr[tr_rel], itr[va_rel]
        Xa, Xb = matrices(df, datos, ftr, fva)
        P, fin = etapa_expertos(expertos, Xa, y[ftr], clases)
        cache.append((Xa, y[ftr], P, fin, Xb, y[fva]))
    hist = []

    def objetivo(params):
        f1s = [K.f1_macro(yb, E.modelo_epl(expertos, params).fit_con_probabilidades(Xa, ya, P, fin).predict(Xb))
               for Xa, ya, P, fin, Xb, yb in cache]
        hist.append({"params": params, "f1_interno": float(np.mean(f1s))})
        return float(np.mean(f1s))

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=K.SEED, n_startup_trials=4, multivariate=True))
    while len(hist) < K.CF.PRESUPUESTO:
        trial = estudio.ask()
        estudio.tell(trial, objetivo(E.sugerir(trial)))
    mejor = max(hist, key=lambda h: h["f1_interno"])
    t1 = time.perf_counter()
    Xtr, Xva = matrices(df, datos, itr, iva)
    P_oof, fin = etapa_expertos(expertos, Xtr, y[itr], clases)
    epl = E.modelo_epl(expertos, mejor["params"]).fit_con_probabilidades(Xtr, y[itr], P_oof, fin)
    t2 = time.perf_counter()
    proba = {"EPL": epl.predict_proba(Xva)}
    t3 = time.perf_counter()
    P_va = epl._proba_expertos(Xva[:, :-2])
    proba["promedio_simple"] = P_va.mean(1)
    w = 1 / np.asarray(epl.perdida_oof_expertos_)
    proba["pesos_error_global"] = np.einsum("m,nmk->nk", w / w.sum(), P_va)
    stk = LogisticRegression(max_iter=3000, class_weight="balanced").fit(P_oof.reshape(len(P_oof), -1), y[itr])
    proba["stacking_lineal"] = proba_completa(stk, P_va.reshape(len(P_va), -1), clases)
    proba["sin_campo_local"] = np.einsum("m,nmk->nk", epl.pesos_globales_, P_va)
    f1 = {k: K.f1_macro(y[iva], clases[np.argmax(v, axis=1)]) for k, v in proba.items()}
    salida = {"conjunto": nombre, "repeticion": r, "fold": f, "params": mejor["params"],
              "f1_interno_mejor": mejor["f1_interno"], "historial": hist, "f1": f1,
              "pesos_globales": epl.pesos_globales_, "expertos": [n for n, _ in expertos],
              "seg_busqueda": t1 - t0, "seg_ajuste_final": t2 - t1, "seg_inferencia": t3 - t2,
              "n_train": len(itr), "n_val": len(iva)}
    ruta_e(nombre, r, f).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta_e(nombre, r, f, "npz"), iva=iva, clases=clases,
                        **{k: v.astype(np.float32) for k, v in proba.items()})
    K.escribir_json(ruta_e(nombre, r, f), salida)
    return nombre, r, f, f1["EPL"]


def resumen():
    """F1 medio por modelo y conjunto, y las pruebas del EPL frente a cada rival (mismas funciones que la 5×5)."""
    S = importlib.import_module("experimentos.04_estadistica")
    from estrato3 import estadistica as ES
    rivales = [m for m, _, _, g in K.benchmarks() if g != "referencia"]
    out = {}
    for nombre in CONJUNTOS:
        filas = []
        for r in range(N_REP):
            for f in range(5):
                fila = {m: json.loads(ruta_b(nombre, m, r, f).read_text(encoding="utf-8"))["f1"] for m in rivales}
                e = json.loads(ruta_e(nombre, r, f).read_text(encoding="utf-8"))
                fila.update({S.ORIGINAL: e["f1"]["EPL"], **{f"[{k}]": v for k, v in e["f1"].items() if k != "EPL"}})
                fila["n_val"], fila["n_train"] = e["n_val"], e["n_train"]
                filas.append(fila)
        T = pd.DataFrame(filas)
        n_val, n_tr = T.pop("n_val").mean(), T.pop("n_train").mean()
        univ = T[rivales + [S.ORIGINAL]]
        fr = ES.friedman(univ)
        comp = S.comparar(T, S.ORIGINAL, rivales, n_val, n_tr)
        variantes = S.comparar(T, S.ORIGINAL, [c for c in T.columns if c.startswith("[")], n_val, n_tr)
        out[nombre] = {"media": T.mean().to_dict(), "sd": T.std(ddof=1).to_dict(),
                       "friedman": {"chi2": fr["chi2"], "p": fr["p"], "rangos": fr["rangos"].to_dict(),
                                    "cd": ES.diferencia_critica(univ.shape[1], len(univ))},
                       "nemenyi_p_epl": ES.nemenyi(univ)[S.ORIGINAL].to_dict(), "epl_vs_rivales": comp,
                       "epl_vs_reglas": variantes}
        K.log(f"{nombre}: " + ", ".join(f"{k} {v:.3f}" for k, v in T.mean().sort_values(ascending=False).items()))
    K.escribir_json(RES / "resumen_externos.json", out)


def main():
    for nombre in CONJUNTOS:
        df, datos, parts = cargar(nombre)
        K.log(f"{nombre}: {len(df)} filas, clases {df.clase.value_counts().sort_index().to_dict()}")
    rivales = [(m, b, o) for m, b, o, g in K.benchmarks() if g != "referencia"]
    tareas = [(n, m, b, o, r, f) for m, b, o in rivales for n in CONJUNTOS for r in range(N_REP) for f in PLIEGUES]
    K.log(f"{len(tareas)} tareas benchmark; {N_TRABAJOS} procesos")
    for n, m, r, f, f1 in Parallel(n_jobs=N_TRABAJOS, return_as="generator_unordered")(
            delayed(tarea_benchmark)(*t) for t in tareas):
        K.log(f"  {n} {m} r{r} f{f}: F1 {f1:.4f}")
    for n, r, f, f1 in Parallel(n_jobs=N_TRABAJOS, return_as="generator_unordered")(
            delayed(tarea_epl)(n, r, f) for n in CONJUNTOS for r in range(N_REP) for f in PLIEGUES):
        K.log(f"  {n} EPL r{r} f{f}: F1 {f1:.4f}")
    if not RAPIDO:
        resumen()
    K.log("FIN externos")


if __name__ == "__main__":
    main()
