# -*- coding: utf-8 -*-
"""Robustez, sensibilidad y limitaciones del modelo original (Cuadro 1 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

1. **Sensibilidad a sus hiperparámetros** (25 pliegues, con la caché de la versión 2, sin reentrenar expertos): se
   parte de la configuración que eligió el bucle interno en cada pliegue y se cambia un hiperparámetro a la vez
   (J, h, k, λ, espacio). Es descriptivo: estas variaciones no se usan para elegir nada.
2. **Estabilidad ante semillas**: 3 semillas del EPL (pliegues del ajuste cruzado y K-medias) en los 5 pliegues de
   la repetición 0, reentrenando los expertos del ajuste cruzado.
3. **Ruido, tamaño y desbalance** (5 pliegues de la repetición 0; hiperparámetros fijos de cada pliegue, sin nueva
   búsqueda): 20 % de etiquetas de entrenamiento cambiadas a un estrato vecino; 25 % del entrenamiento; estratos
   5 y 6 reducidos al 25 %. En cada condición se comparan el EPL, el Random Forest, CatBoost,
   el promedio simple y los pesos por error global, todos con los mismos datos alterados (la validación no se toca).

Uso (desde la raíz, después de 03b):  python -m experimentos.11_robustez      (N_TRABAJOS=4)
"""
import importlib
import json
import os
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from experimentos import comun as K
from src.original_model import proba_completa

V2 = importlib.import_module("experimentos.03b_epl_v2_5x5")
N_TRABAJOS = int(os.environ.get("N_TRABAJOS", 4))
SAL = K.RES / "robustez"
REJILLAS = {"n_anclas": [1, 5, 10, 20, 40, 80], "ancho_banda": [0.25, 0.5, 1.0, 2.0, 4.0], "k_anclas": [1, 3, 6, 10],
            "lam": [0.0, 0.01, 0.1, 1.0, 10.0], "espacio": ["geografico", "variables", "probabilidades"]}
# versión recortada por tiempo: 3 semillas y una condición de cada tipo (ruido, tamaño, desbalance)
SEMILLAS = [42, 1, 2]
CONDICIONES = ["base", "ruido_20", "tamano_25", "desbalance_56"]


def sensibilidad(r, f):
    z = V2.cache(r, f)
    v2 = json.loads(V2.ruta(r, f).read_text(encoding="utf-8"))
    Xtr, ytr, Poof = z["Xtr"].astype(float), z["ytr"], z["Poof"].astype(float)
    Xva, yva, Pva = z["Xva"].astype(float), z["yva"], z["Pva"].astype(float)
    out = {}
    for h, valores in REJILLAS.items():
        for v in valores:
            p = dict(v2["params"], **{h: v})
            m = V2.modelo(p, Poof.shape[1]).fit_con_probabilidades(Xtr, ytr, Poof, None)
            out[f"{h}={v}"] = K.f1_macro(yva, m.classes_[np.argmax(m.predict_proba_desde_probabilidades(Xva, Pva), 1)])
    return r, f, out


def alterar(y, cond, rng):
    """Índices y etiquetas de entrenamiento alterados según la condición (la validación nunca se altera)."""
    y = y.copy()
    idx = np.arange(len(y))
    if cond.startswith("ruido"):
        frac = int(cond.split("_")[1]) / 100
        cambia = rng.random(len(y)) < frac
        paso = np.where(rng.random(len(y)) < 0.5, -1, 1)
        nuevo = np.clip(y + paso, 1, 6)
        nuevo = np.where(nuevo == y, np.clip(y - paso, 1, 6), nuevo)
        y[cambia] = nuevo[cambia]
    elif cond.startswith("tamano"):
        frac = int(cond.split("_")[1]) / 100
        idx = np.sort(rng.choice(len(y), int(frac * len(y)), replace=False))
    elif cond == "desbalance_56":
        quita = np.isin(y, [5, 6]) & (rng.random(len(y)) > 0.25)
        idx = idx[~quita]
    return idx, y[idx]


def condicion(cond, f, semilla_epl=K.SEED):
    """Una condición (o semilla) en el pliegue externo f de la repetición 0."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(1000 + f)
    d = K.datos()
    rep = K.particiones()[0]
    y_all = d["sub"].estrato_num.to_numpy().astype(int)
    itr, iva = rep["externos"][f]
    sel, ytr = alterar(y_all[itr], cond, rng)
    Xtr, Xva = K.matrices(d, itr[sel], iva)
    yva = y_all[iva]
    v2 = json.loads(V2.ruta(0, f).read_text(encoding="utf-8"))
    expertos = V2.expertos_del_pliegue(0, f)
    epl = V2.modelo(v2["params"], len(expertos)).set_params(expertos=expertos, random_state=semilla_epl)
    epl.fit(Xtr, ytr)
    P_va = epl._proba_expertos(Xva[:, :-2])
    out = {"EPL": K.f1_macro(yva, K.pred(epl.predict_proba(Xva))),
           "promedio_simple": K.f1_macro(yva, K.pred(P_va.mean(1)))}
    w = 1 / np.asarray(epl.perdida_oof_expertos_)
    out["pesos_error_global"] = K.f1_macro(yva, K.pred(np.einsum("m,nmk->nk", w / w.sum(), P_va)))
    nombres = [n for n, _ in expertos]
    out["Random Forest"] = K.f1_macro(yva, K.pred(P_va[:, nombres.index("Random Forest")]))
    if semilla_epl == K.SEED:                      # CatBoost solo en las condiciones (no depende de la semilla del EPL)
        j, _, _ = K.leer_benchmark("CatBoost", 0, f)
        cb = clone(K.experto("CatBoost", "class_weight", j["params"])).fit(Xtr[:, :-2], ytr)
        out["CatBoost"] = K.f1_macro(yva, K.pred(proba_completa(cb, Xva[:, :-2], K.CLASES)))
    return cond, f, semilla_epl, out


def main():
    SAL.mkdir(parents=True, exist_ok=True)
    pares = [(r, f) for r in range(K.N_REPETICIONES) for f in range(5)]
    sens = Parallel(n_jobs=N_TRABAJOS)(delayed(sensibilidad)(r, f) for r, f in pares)
    tabla = {k: [s[2][k] for s in sens] for k in sens[0][2]}
    K.escribir_json(SAL / "sensibilidad.json", {k: {"media": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                                     "por_pliegue": v} for k, v in tabla.items()})
    K.log("sensibilidad lista")
    ruta_c = SAL / "condiciones.json"
    hechos = json.loads(ruta_c.read_text(encoding="utf-8")) if ruta_c.exists() else {}
    tareas = [(c, f, K.SEED) for c in CONDICIONES for f in range(5)] + \
             [("base", f, s) for s in SEMILLAS[1:] for f in range(5)]
    tareas = [t for t in tareas if f"{t[0]}|{t[1]}|{t[2]}" not in hechos]
    for cond, f, s, out in Parallel(n_jobs=N_TRABAJOS, return_as="generator_unordered")(
            delayed(condicion)(*t) for t in tareas):
        hechos[f"{cond}|{f}|{s}"] = out
        K.escribir_json(ruta_c, hechos)
        K.log(f"  {cond} f{f} semilla {s}: EPL {out['EPL']:.4f}")
    resumen = {}
    for c in CONDICIONES:
        filas = [hechos[f"{c}|{f}|{K.SEED}"] for f in range(5)]
        resumen[c] = {m: float(np.mean([x[m] for x in filas])) for m in filas[0]}
    semillas = {s: float(np.mean([hechos[f"base|{f}|{s}"]["EPL"] for f in range(5)])) for s in SEMILLAS}
    K.escribir_json(SAL / "resumen.json", {"condiciones": resumen, "semillas_EPL": semillas,
                                           "sd_entre_semillas": float(np.std(list(semillas.values()), ddof=1))})
    for c, v in resumen.items():
        K.log(f"{c:>14}: " + ", ".join(f"{m} {x:.3f}" for m, x in v.items()))
    K.log(f"semillas: {semillas}")
    K.log("FIN robustez")


if __name__ == "__main__":
    main()
