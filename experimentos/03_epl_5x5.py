# -*- coding: utf-8 -*-
"""Modelo original (EPL) en la validación anidada repetida 5×5, con reglas de contraste y ablación.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

En cada pliegue externo (r, f), con las MISMAS particiones, semilla y preprocesamiento que los benchmarks:

1. **Expertos** = la mejor combinación de cada uno de los 7 modelos de la guía, con los hiperparámetros que eligió
   SU búsqueda interna en ese mismo pliegue (``02_benchmarks_5x5.py``). No se ve el pliegue externo.
2. **Búsqueda del EPL** en los 3 pliegues internos: bayesiana (TPE de Optuna, 12 evaluaciones, 4 de arranque,
   semilla 42: el mismo método y presupuesto de las búsquedas del E3) sobre J, h, k, λ y los pesos de clase. Las
   probabilidades fuera de muestra de los expertos de cada pliegue interno se calculan una vez y se reutilizan en
   las 12 evaluaciones (``fit_con_probabilidades`` da el mismo modelo que ``fit``).
3. **Reajuste** con todo el entrenamiento externo y evaluación en el pliegue externo.
4. **Reglas estándar de contraste** (Cuadro 1 de la guía) sobre las mismas probabilidades fuera de muestra:
   promedio simple, pesos por error global (∝ 1 / log-pérdida fuera de muestra) y stacking lineal (logística
   multinomial sobre las probabilidades de los expertos).
5. **Ablación** (cada componente retirado por separado, el resto con los hiperparámetros elegidos): sin campo
   local (J = 1: pesos globales), sin penalización KL (λ = 0), sin suavizado (k = 1: ancla más cercana), con los
   pesos de clase contrarios y sin ajuste cruzado (pesos aprendidos con las predicciones de los expertos sobre sus
   propios datos).

Salida por pliegue: ``resultados/cv5x5/epl/r<r>_f<f>.json`` y ``.npz`` (probabilidades de cada variante).

Uso (desde la raíz, después de 02):  python -m experimentos.03_epl_5x5      (N_TRABAJOS=4 por defecto)
"""
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
from src.original_model import EnsamblePesosLocales, probabilidades_fuera_de_muestra, proba_completa

N_TRABAJOS = int(os.environ.get("N_TRABAJOS", 4))
TAM_BLOQUE_KM = 1.0          # celdas del ajuste cruzado (fijas: definen las probabilidades que se reutilizan)
N_PLIEGUES_CRUZADOS = 5
ESPACIO = {"n_anclas": ("intlog", 4, 80), "ancho_banda": ("log", 0.25, 4.0), "k_anclas": ("int", 1, 10),
           "lam": ("log", 1e-3, 10.0), "peso_clase": ("cat", ["balanced", "ninguno"])}
optuna.logging.set_verbosity(optuna.logging.WARNING)


def ruta(r, f, ext="json"):
    return K.RES / "epl" / f"r{r}_f{f}.{ext}"


def modelo_epl(expertos, params):
    p = dict(params)
    p["peso_clase"] = None if p.get("peso_clase") == "ninguno" else p.get("peso_clase", "balanced")
    return EnsamblePesosLocales(expertos=expertos, expertos_usan_ubicacion=False, tam_bloque=TAM_BLOQUE_KM,
                                n_pliegues=N_PLIEGUES_CRUZADOS, random_state=K.SEED, **p)


def etapa_expertos(expertos, Xa, ya):
    """Probabilidades fuera de muestra (n, M, K) y expertos finales ajustados con [Xa sin coordenadas]."""
    Xe, S = Xa[:, :-2], Xa[:, -2:]
    P = probabilidades_fuera_de_muestra(expertos, Xe, ya, S, K.CLASES, TAM_BLOQUE_KM, N_PLIEGUES_CRUZADOS, K.SEED)
    finales = [clone(e).fit(Xe, ya) for _, e in expertos]
    return P, finales


def sugerir(trial):
    return {"n_anclas": trial.suggest_int("n_anclas", 4, 80, log=True),
            "ancho_banda": trial.suggest_float("ancho_banda", 0.25, 4.0, log=True),
            "k_anclas": trial.suggest_int("k_anclas", 1, 10),
            "lam": trial.suggest_float("lam", 1e-3, 10.0, log=True),
            "peso_clase": trial.suggest_categorical("peso_clase", ["balanced", "ninguno"])}


def tarea(r, f):
    if ruta(r, f).exists():
        return r, f, json.loads(ruta(r, f).read_text(encoding="utf-8"))["f1"]["EPL"]
    warnings.filterwarnings("ignore")
    from estrato3 import evaluacion as EV
    K.CF.fijar_semillas()
    d = K.datos()
    rep = K.particiones()[r]
    y = d["sub"].estrato_num.to_numpy().astype(int)
    itr, iva = rep["externos"][f]
    combos = K.mejores_combinaciones(incluir_extra=False)
    params_expertos = {c.modelo: K.leer_benchmark(c.modelo, r, f)[0]["params"] for _, c in combos.iterrows()}
    expertos = [(c.modelo, K.experto(c.modelo, c.balanceo, params_expertos[c.modelo])) for _, c in combos.iterrows()]

    # 2. búsqueda interna
    t0 = time.perf_counter()
    cache = []
    for tr_rel, va_rel in rep["internos"][f]:
        ftr, fva = itr[tr_rel], itr[va_rel]
        Xa, Xb = K.matrices(K.datos(), ftr, fva)
        P, finales = etapa_expertos(expertos, Xa, y[ftr])
        cache.append((Xa, y[ftr], P, finales, Xb, y[fva]))
    historial = []

    def objetivo(params):
        f1s = []
        for Xa, ya, P, finales, Xb, yb in cache:
            m = modelo_epl(expertos, params).fit_con_probabilidades(Xa, ya, P, finales)
            f1s.append(K.f1_macro(yb, m.predict(Xb)))
        historial.append({"evaluacion": len(historial) + 1, "params": params, "f1_interno": float(np.mean(f1s)),
                          "f1_interno_sd": float(np.std(f1s)), "segundos_acum": time.perf_counter() - t0})
        return float(np.mean(f1s))

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=K.SEED, n_startup_trials=4, multivariate=True))
    while len(historial) < K.CF.PRESUPUESTO:
        trial = estudio.ask()
        estudio.tell(trial, objetivo(sugerir(trial)))
    mejor = max(historial, key=lambda h: h["f1_interno"])
    t_busqueda = time.perf_counter() - t0
    del cache

    # 3. reajuste con el entrenamiento externo
    t1 = time.perf_counter()
    Xtr, Xva = K.matrices(d, itr, iva)
    P_oof, finales = etapa_expertos(expertos, Xtr, y[itr])
    epl = modelo_epl(expertos, mejor["params"]).fit_con_probabilidades(Xtr, y[itr], P_oof, finales)
    t_ajuste = time.perf_counter() - t1
    t2 = time.perf_counter()
    proba = {"EPL": epl.predict_proba(Xva)}
    t_inferencia = time.perf_counter() - t2
    P_va = epl._proba_expertos(Xva[:, :-2])                                      # (n_va, M, K)
    M = P_va.shape[1]

    # 4. reglas estándar sobre las mismas probabilidades
    proba["promedio_simple"] = P_va.mean(1)
    w_err = 1 / np.asarray(epl.perdida_oof_expertos_)
    proba["pesos_error_global"] = np.einsum("m,nmk->nk", w_err / w_err.sum(), P_va)
    stk = LogisticRegression(max_iter=3000, class_weight="balanced").fit(P_oof.reshape(len(P_oof), -1), y[itr])
    proba["stacking_lineal"] = proba_completa(stk, P_va.reshape(len(P_va), -1), K.CLASES)

    # 5. ablación
    proba["sin_campo_local"] = np.einsum("m,nmk->nk", epl.pesos_globales_, P_va)
    for nombre, cambio in [("sin_kl", {"lam": 0.0}), ("sin_suavizado", {"k_anclas": 1}),
                           ("pesos_clase_contrarios", {"peso_clase": "ninguno" if mejor["params"]["peso_clase"]
                                                       == "balanced" else "balanced"})]:
        p = dict(mejor["params"], **cambio)
        proba[nombre] = modelo_epl(expertos, p).fit_con_probabilidades(Xtr, y[itr], P_oof, finales).predict_proba(Xva)
    P_in = np.stack([proba_completa(e, Xtr[:, :-2], K.CLASES) for e in finales], axis=1)
    proba["sin_ajuste_cruzado"] = modelo_epl(expertos, mejor["params"]).fit_con_probabilidades(
        Xtr, y[itr], P_in, finales).predict_proba(Xva)

    # control: cada experto debe reproducir la probabilidad de su tarea benchmark
    control = {}
    for m, (nombre, _) in enumerate(expertos):
        _, pb, ivb = K.leer_benchmark(nombre, r, f)
        assert np.array_equal(ivb, iva)
        control[nombre] = float(np.abs(P_va[:, m] - pb).max())

    f1 = {k: K.f1_macro(y[iva], K.pred(v)) for k, v in proba.items()}
    salida = {"repeticion": r, "fold": f, "semilla_particion": rep["semilla"], "params": mejor["params"],
              "f1_interno_mejor": mejor["f1_interno"], "historial": historial, "f1": f1,
              "metricas_EPL": EV.metricas(y[iva], proba["EPL"]),
              "expertos": [n for n, _ in expertos], "pesos_globales": epl.pesos_globales_,
              "pesos_error_global": w_err / w_err.sum(), "n_anclas_reales": len(epl.anclas_),
              "n_iter": epl.n_iter_, "historia_perdida": epl.historia_,
              "dif_max_prob_expertos_vs_benchmark": control,
              "seg_busqueda": t_busqueda, "seg_ajuste_final": t_ajuste, "seg_inferencia": t_inferencia,
              "n_train": len(itr), "n_val": len(iva), "presupuesto": K.CF.PRESUPUESTO,
              "tam_bloque_km": TAM_BLOQUE_KM, "n_pliegues_cruzados": N_PLIEGUES_CRUZADOS}
    K.RES.joinpath("epl").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta(r, f, "npz"), iva=iva, **{k: v.astype(np.float32) for k, v in proba.items()})
    K.escribir_json(ruta(r, f), salida)
    return r, f, f1["EPL"]


def main():
    pares = [(r, f) for r in range(K.N_REPETICIONES) for f in range(5)]
    faltan = [(r, f) for r, f in pares if not ruta(r, f).exists()]
    K.log(f"EPL: {len(pares)} pliegues, pendientes {len(faltan)}; {N_TRABAJOS} procesos")
    for r, f, f1 in Parallel(n_jobs=N_TRABAJOS, return_as="generator_unordered")(
            delayed(tarea)(r, f) for r, f in faltan):
        K.log(f"  EPL r{r} f{f}: F1 {f1:.4f}")
    filas = [json.loads(ruta(r, f).read_text(encoding="utf-8")) for r, f in pares]
    variantes = list(filas[0]["f1"])
    resumen = {v: {"media": float(np.mean([x["f1"][v] for x in filas])),
                   "sd": float(np.std([x["f1"][v] for x in filas], ddof=1))} for v in variantes}
    K.escribir_json(K.RES / "epl_resumen.json", resumen)
    for v in variantes:
        K.log(f"{v:>24}: {resumen[v]['media']:.4f} ± {resumen[v]['sd']:.4f}")
    K.log("FIN EPL")


if __name__ == "__main__":
    main()
