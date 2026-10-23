# -*- coding: utf-8 -*-
"""Modelo original, versión 2: el espacio de competencia se elige en el bucle interno (validación 5×5).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Por qué hay una versión 2. En la versión 1 (``03_epl_5x5.py``) el campo de pesos vivía en el espacio geográfico y
la ablación mostró que casi no aportaba (+0.002 de F1 frente a los pesos globales): con pliegues de «zonas nuevas»,
los pesos que se aprenden en una zona no se trasladan a otra. La guía permite iterar el diseño si se reporta todo
(Figura 2). La versión 2 agrega el hiperparámetro ``espacio`` ∈ {geográfico, variables, probabilidades} y lo deja
elegir al bucle interno, igual que J, h, k, λ y los pesos de clase: la elección nunca ve el pliegue externo, así
que la estimación externa sigue siendo honesta. Se reportan las dos versiones.

Etapa A (cara, se guarda en ``resultados/cv5x5/cache/``): para cada pliegue externo (r, f) y cada uno de sus 3
pliegues internos, las probabilidades fuera de muestra de los 7 expertos (ajuste cruzado por bloques) y sus
probabilidades sobre la validación correspondiente. Con eso, cualquier variante del combinador se evalúa sin volver
a entrenar los expertos.

Etapa B (barata): búsqueda bayesiana de 12 evaluaciones (mismo método y presupuesto que en la versión 1 y en los
benchmarks) con el espacio incluido; reajuste; variantes con el espacio fijo (cada una con su propia búsqueda
interna: análisis de sensibilidad, no se usan para elegir); reglas estándar y ablación de la configuración
elegida. Control: la variante «solo geográfico» repite exactamente la búsqueda de la versión 1.

Uso (desde la raíz):  python -m experimentos.03b_epl_v2_5x5      (N_TRABAJOS=3 por defecto)
"""
import importlib
import json
import os
import time
import warnings

import numpy as np
import optuna
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from experimentos import comun as K
from src.original_model import ESPACIOS, EnsamblePesosLocales, probabilidades_fuera_de_muestra, proba_completa

V1 = importlib.import_module("experimentos.03_epl_5x5")
N_TRABAJOS = int(os.environ.get("N_TRABAJOS", 3))
TAM_BLOQUE_KM, N_PLIEGUES_CRUZADOS = V1.TAM_BLOQUE_KM, V1.N_PLIEGUES_CRUZADOS
optuna.logging.set_verbosity(optuna.logging.WARNING)
CACHE = K.RES / "cache"
SALIDA = K.RES / "epl_v2"


def ruta(r, f, ext="json"):
    return SALIDA / f"r{r}_f{f}.{ext}"


def modelo(params, M):
    """EPL sin expertos ajustados (predice con las probabilidades de la caché). M = número de expertos."""
    p = dict(params)
    p["peso_clase"] = None if p.get("peso_clase") == "ninguno" else p.get("peso_clase", "balanced")
    expertos = [(f"e{m}", None) for m in range(M)]          # solo se usan los nombres y el número de expertos
    return EnsamblePesosLocales(expertos=expertos, expertos_usan_ubicacion=False, tam_bloque=TAM_BLOQUE_KM,
                                n_pliegues=N_PLIEGUES_CRUZADOS, random_state=K.SEED, **p)


def sugerir(trial, espacio=None):
    """Espacio de búsqueda de la versión 1 más el espacio de competencia (o fijo, si se pasa)."""
    p = V1.sugerir(trial)
    p["espacio"] = espacio or trial.suggest_categorical("espacio", list(ESPACIOS))
    return p


# ---------------------------------------------------------------------------
# Etapa A: caché de probabilidades de los expertos
# ---------------------------------------------------------------------------
def expertos_del_pliegue(r, f):
    combos = K.mejores_combinaciones(incluir_extra=False)
    return [(c.modelo, K.experto(c.modelo, c.balanceo, K.leer_benchmark(c.modelo, r, f)[0]["params"]))
            for _, c in combos.iterrows()]


def etapa(expertos, Xa, ya, Xb):
    """Probabilidades fuera de muestra en Xa, en Xa con los expertos finales (ablación) y en Xb."""
    Xe, S = Xa[:, :-2], Xa[:, -2:]
    P_oof = probabilidades_fuera_de_muestra(expertos, Xe, ya, S, K.CLASES, TAM_BLOQUE_KM, N_PLIEGUES_CRUZADOS, K.SEED)
    finales = [clone(e).fit(Xe, ya) for _, e in expertos]
    P_in = np.stack([proba_completa(e, Xe, K.CLASES) for e in finales], axis=1)
    P_b = np.stack([proba_completa(e, Xb[:, :-2], K.CLASES) for e in finales], axis=1)
    return P_oof, P_in, P_b


def cache(r, f):
    """Calcula (o lee) la caché del pliegue externo (r, f)."""
    archivo = CACHE / f"r{r}_f{f}.npz"
    if archivo.exists():
        return dict(np.load(archivo))
    K.CF.fijar_semillas()
    d = K.datos()
    rep = K.particiones()[r]
    y = d["sub"].estrato_num.to_numpy().astype(int)
    itr, iva = rep["externos"][f]
    expertos = expertos_del_pliegue(r, f)
    t0 = time.perf_counter()
    z = {}
    for i, (tr_rel, va_rel) in enumerate(rep["internos"][f]):
        ftr, fva = itr[tr_rel], itr[va_rel]
        Xa, Xb = K.matrices(d, ftr, fva)
        P_oof, _, P_b = etapa(expertos, Xa, y[ftr], Xb)
        z.update({f"i{i}_Xa": Xa, f"i{i}_ya": y[ftr], f"i{i}_Poof": P_oof, f"i{i}_Xb": Xb, f"i{i}_yb": y[fva],
                  f"i{i}_Pb": P_b})
    t1 = time.perf_counter()
    Xtr, Xva = K.matrices(d, itr, iva)
    P_oof, P_in, P_va = etapa(expertos, Xtr, y[itr], Xva)
    t2 = time.perf_counter()
    z.update({"Xtr": Xtr, "ytr": y[itr], "Poof": P_oof, "Pin": P_in, "Xva": Xva, "yva": y[iva], "Pva": P_va,
              "iva": iva, "seg_expertos_internos": t1 - t0, "seg_expertos_externo": t2 - t1})
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archivo, **{k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64
                                        else v) for k, v in z.items()})
    return dict(np.load(archivo))


# ---------------------------------------------------------------------------
# Etapa B: búsqueda, variantes, reglas y ablación (sin reentrenar expertos)
# ---------------------------------------------------------------------------
def buscar(z, M, espacio=None):
    """Búsqueda bayesiana de 12 evaluaciones en los pliegues de z (i0, i1, ...); devuelve (mejor, historial)."""
    n_pliegues = sum(1 for k in z if k.endswith("_Xa"))
    internos = [(z[f"i{i}_Xa"].astype(float), z[f"i{i}_ya"], z[f"i{i}_Poof"].astype(float), z[f"i{i}_Xb"].astype(float),
                 z[f"i{i}_yb"], z[f"i{i}_Pb"].astype(float)) for i in range(n_pliegues)]
    hist = []

    def objetivo(params):
        f1s = []
        for Xa, ya, P, Xb, yb, Pb in internos:
            m = modelo(params, M).fit_con_probabilidades(Xa, ya, P, None)
            f1s.append(K.f1_macro(yb, m.classes_[np.argmax(m.predict_proba_desde_probabilidades(Xb, Pb), 1)]))
        hist.append({"evaluacion": len(hist) + 1, "params": params, "f1_interno": float(np.mean(f1s)),
                     "f1_interno_sd": float(np.std(f1s))})
        return float(np.mean(f1s))

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=K.SEED, n_startup_trials=4, multivariate=True))
    while len(hist) < K.CF.PRESUPUESTO:
        trial = estudio.ask()
        estudio.tell(trial, objetivo(sugerir(trial, espacio)))
    return max(hist, key=lambda h: h["f1_interno"]), hist


def tarea(r, f):
    if ruta(r, f).exists():
        return r, f, json.loads(ruta(r, f).read_text(encoding="utf-8"))["f1"]["EPL v2"]
    warnings.filterwarnings("ignore")
    from estrato3 import evaluacion as EV
    t0 = time.perf_counter()
    z = cache(r, f)
    t1 = time.perf_counter()
    Xtr, ytr, Poof, Pin = z["Xtr"].astype(float), z["ytr"], z["Poof"].astype(float), z["Pin"].astype(float)
    Xva, yva, Pva = z["Xva"].astype(float), z["yva"], z["Pva"].astype(float)
    M = Poof.shape[1]

    def ajustar_predecir(params, P_entrenamiento=Poof):
        m = modelo(params, M).fit_con_probabilidades(Xtr, ytr, P_entrenamiento, None)
        return m, m.predict_proba_desde_probabilidades(Xva, Pva)

    mejor, hist = buscar(z, M)
    t2 = time.perf_counter()
    epl, p_epl = ajustar_predecir(mejor["params"])
    t3 = time.perf_counter()
    proba = {"EPL v2": p_epl}
    elegidos = {}
    for esp in ESPACIOS:                                   # sensibilidad: cada espacio con su propia búsqueda
        m_esp, _ = buscar(z, M, esp)
        elegidos[esp] = m_esp
        proba[f"solo {esp}"] = ajustar_predecir(m_esp["params"])[1]
    proba["promedio_simple"] = Pva.mean(1)
    w = 1 / np.asarray(epl.perdida_oof_expertos_)
    proba["pesos_error_global"] = np.einsum("m,nmk->nk", w / w.sum(), Pva)
    stk = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Poof.reshape(len(Poof), -1), ytr)
    proba["stacking_lineal"] = proba_completa(stk, Pva.reshape(len(Pva), -1), K.CLASES)
    proba["sin_campo_local"] = np.einsum("m,nmk->nk", epl.pesos_globales_, Pva)
    contrario = "ninguno" if mejor["params"]["peso_clase"] == "balanced" else "balanced"
    for nombre, cambio in [("sin_kl", {"lam": 0.0}), ("sin_suavizado", {"k_anclas": 1}),
                           ("pesos_clase_contrarios", {"peso_clase": contrario})]:
        proba[nombre] = ajustar_predecir(dict(mejor["params"], **cambio))[1]
    proba["sin_ajuste_cruzado"] = ajustar_predecir(mejor["params"], Pin)[1]

    # controles: expertos = benchmarks del mismo pliegue; «solo geográfico» = versión 1
    combos = K.mejores_combinaciones(incluir_extra=False)
    control = {c.modelo: float(np.abs(Pva[:, m] - K.leer_benchmark(c.modelo, r, f)[1]).max())
               for m, (_, c) in enumerate(combos.iterrows())}
    v1 = json.loads((K.RES / "epl" / f"r{r}_f{f}.json").read_text(encoding="utf-8"))
    f1 = {k: K.f1_macro(yva, K.pred(v)) for k, v in proba.items()}
    salida = {"repeticion": r, "fold": f, "params": mejor["params"], "espacio_elegido": mejor["params"]["espacio"],
              "f1_interno_mejor": mejor["f1_interno"], "historial": hist, "f1": f1,
              "f1_interno_por_espacio": {e: elegidos[e]["f1_interno"] for e in ESPACIOS},
              "params_por_espacio": {e: elegidos[e]["params"] for e in ESPACIOS},
              "metricas_EPL_v2": EV.metricas(yva, p_epl), "expertos": list(combos.modelo),
              "pesos_globales": epl.pesos_globales_, "n_anclas_reales": len(epl.anclas_), "n_iter": epl.n_iter_,
              "historia_perdida": epl.historia_, "dif_max_prob_expertos_vs_benchmark": control,
              "control_v1": {"f1_v1": v1["f1"]["EPL"], "f1_solo_geografico": f1["solo geografico"],
                             "params_v1": v1["params"], "params_solo_geografico": elegidos["geografico"]["params"]},
              "seg_cache": t1 - t0, "seg_busqueda": t2 - t1, "seg_ajuste_final": t3 - t2,
              "seg_expertos_internos": float(z["seg_expertos_internos"]),
              "seg_expertos_externo": float(z["seg_expertos_externo"]), "n_train": len(ytr), "n_val": len(yva)}
    SALIDA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta(r, f, "npz"), iva=z["iva"], **{k: v.astype(np.float32) for k, v in proba.items()})
    K.escribir_json(ruta(r, f), salida)
    return r, f, f1["EPL v2"]


def main():
    pares = [(r, f) for r in range(K.N_REPETICIONES) for f in range(5)]
    K.log(f"EPL v2: {len(pares)} pliegues; {N_TRABAJOS} procesos")
    for r, f, f1 in Parallel(n_jobs=N_TRABAJOS, return_as="generator_unordered")(
            delayed(tarea)(r, f) for r, f in pares):
        K.log(f"  EPL v2 r{r} f{f}: F1 {f1:.4f}")
    filas = [json.loads(ruta(r, f).read_text(encoding="utf-8")) for r, f in pares]
    resumen = {v: {"media": float(np.mean([x["f1"][v] for x in filas])),
                   "sd": float(np.std([x["f1"][v] for x in filas], ddof=1))} for v in filas[0]["f1"]}
    resumen["espacio_elegido"] = {e: sum(x["espacio_elegido"] == e for x in filas) for e in ESPACIOS}
    resumen["control_v1_dif_max"] = max(abs(x["control_v1"]["f1_v1"] - x["control_v1"]["f1_solo_geografico"])
                                        for x in filas)
    K.escribir_json(K.RES / "epl_v2_resumen.json", resumen)
    for v, s in resumen.items():
        K.log(f"{v:>24}: {s}")
    K.log("FIN EPL v2")


if __name__ == "__main__":
    main()
