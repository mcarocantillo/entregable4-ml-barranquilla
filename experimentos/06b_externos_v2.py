# -*- coding: utf-8 -*-
"""Modelo original versión 2 en los tres conjuntos adicionales (mismas particiones y rivales que ``06_externos.py``).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Para cada conjunto y pliegue externo: caché de las probabilidades de los 7 expertos (con los hiperparámetros que
eligió la búsqueda interna de cada benchmark en ese pliegue), búsqueda bayesiana de 12 evaluaciones con el espacio de
competencia incluido y reajuste. El resumen compara la versión 1 (``06_externos.py``), la versión 2 y los 9 rivales
con las mismas pruebas de la 5×5. (La versión 1 en estos conjuntos se retiró por tiempo: solo se evalúa el modelo final.)

Uso (desde la raíz, después de los rivales de 06):  python -m experimentos.06b_externos_v2      (N_TRABAJOS=4)
"""
import importlib
import json
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from experimentos import comun as K
from src.original_model import probabilidades_fuera_de_muestra, proba_completa

X6 = importlib.import_module("experimentos.06_externos")
V2 = importlib.import_module("experimentos.03b_epl_v2_5x5")
N_TRABAJOS = int(os.environ.get("N_TRABAJOS", 4))


def ruta(nombre, r, f, ext="json"):
    return X6.RES / nombre / "epl_v2" / f"r{r}_f{f}.{ext}"


def etapa(expertos, Xa, ya, Xb, clases):
    Xe, S = Xa[:, :-2], Xa[:, -2:]
    P = probabilidades_fuera_de_muestra(expertos, Xe, ya, S, clases, V2.TAM_BLOQUE_KM, V2.N_PLIEGUES_CRUZADOS, K.SEED)
    fin = [clone(e).fit(Xe, ya) for _, e in expertos]
    return P, np.stack([proba_completa(e, Xb[:, :-2], clases) for e in fin], axis=1)


def tarea(nombre, r, f):
    if ruta(nombre, r, f).exists():
        return nombre, r, f, json.loads(ruta(nombre, r, f).read_text(encoding="utf-8"))["f1"]["EPL v2"]
    warnings.filterwarnings("ignore")
    K.CF.fijar_semillas()
    df, datos, parts = X6.cargar(nombre)
    y = df.clase.to_numpy()
    clases = np.unique(y)
    itr, iva = parts[r]["externos"][f]
    combos = K.mejores_combinaciones(incluir_extra=False)
    expertos = [(c.modelo, K.experto(c.modelo, c.balanceo, json.loads(
        X6.ruta_b(nombre, c.modelo, r, f).read_text(encoding="utf-8"))["params"])) for _, c in combos.iterrows()]
    z = {}
    for i, (tr_rel, va_rel) in enumerate(parts[r]["internos"][f]):
        ftr, fva = itr[tr_rel], itr[va_rel]
        Xa, Xb = X6.matrices(df, datos, ftr, fva)
        P, Pb = etapa(expertos, Xa, y[ftr], Xb, clases)
        z.update({f"i{i}_Xa": Xa, f"i{i}_ya": y[ftr], f"i{i}_Poof": P, f"i{i}_Xb": Xb, f"i{i}_yb": y[fva],
                  f"i{i}_Pb": Pb})
    Xtr, Xva = X6.matrices(df, datos, itr, iva)
    Poof, Pva = etapa(expertos, Xtr, y[itr], Xva, clases)
    mejor, hist = V2.buscar(z, len(expertos))
    epl = V2.modelo(mejor["params"], len(expertos)).fit_con_probabilidades(Xtr, y[itr], Poof, None)
    proba = {"EPL v2": epl.predict_proba_desde_probabilidades(Xva, Pva)}
    for esp in ("geografico", "variables", "probabilidades"):
        m_esp, _ = V2.buscar(z, len(expertos), esp)
        proba[f"solo {esp}"] = V2.modelo(m_esp["params"], len(expertos)).fit_con_probabilidades(
            Xtr, y[itr], Poof, None).predict_proba_desde_probabilidades(Xva, Pva)
    f1 = {k: K.f1_macro(y[iva], clases[np.argmax(v, 1)]) for k, v in proba.items()}
    ruta(nombre, r, f).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta(nombre, r, f, "npz"), iva=iva, clases=clases,
                        **{k: v.astype(np.float32) for k, v in proba.items()})
    K.escribir_json(ruta(nombre, r, f), {"conjunto": nombre, "repeticion": r, "fold": f, "params": mejor["params"],
                                         "espacio_elegido": mejor["params"]["espacio"], "historial": hist, "f1": f1,
                                         "n_train": len(itr), "n_val": len(iva)})
    return nombre, r, f, f1["EPL v2"]


def resumen():
    S = importlib.import_module("experimentos.04_estadistica")
    from estrato3 import estadistica as ES
    rivales = [m for m, _, _, g in K.benchmarks() if g != "referencia"]
    out = {}
    for nombre in X6.CONJUNTOS:
        filas = []
        for r in range(X6.N_REP):
            for f in X6.PLIEGUES:
                fila = {m: json.loads(X6.ruta_b(nombre, m, r, f).read_text(encoding="utf-8"))["f1"] for m in rivales}
                v2 = json.loads(ruta(nombre, r, f).read_text(encoding="utf-8"))
                fila.update({S.ORIGINAL: v2["f1"]["EPL v2"],
                             **{f"[{k}]": v for k, v in v2["f1"].items() if k != "EPL v2"},
                             "espacio": v2["espacio_elegido"], "n_val": v2["n_val"], "n_train": v2["n_train"]})
                filas.append(fila)
        T = pd.DataFrame(filas)
        espacios = T.pop("espacio").value_counts().to_dict()
        n_val, n_tr = T.pop("n_val").mean(), T.pop("n_train").mean()
        univ = T[rivales + [S.ORIGINAL]]
        fr = ES.friedman(univ)
        out[nombre] = {"media": T.mean().to_dict(), "sd": T.std(ddof=1).to_dict(), "espacio_elegido": espacios,
                       "friedman": {"chi2": fr["chi2"], "p": fr["p"], "rangos": fr["rangos"].to_dict(),
                                    "cd": ES.diferencia_critica(univ.shape[1], len(univ))},
                       "nemenyi_p_epl": ES.nemenyi(univ)[S.ORIGINAL].to_dict(),
                       "epl_v2_vs_rivales": S.comparar(T, S.ORIGINAL, rivales, n_val, n_tr),
                       "epl_v2_vs_variantes": S.comparar(T, S.ORIGINAL, [c for c in T.columns if c.startswith("[")],
                                                         n_val, n_tr)}
        K.log(f"{nombre}: " + ", ".join(f"{k} {v:.3f}" for k, v in T.mean().sort_values(ascending=False).items()
                                        if not k.startswith("[")) + f" | espacios {espacios}")
    K.escribir_json(X6.RES / "resumen_externos_v2.json", out)


def main():
    tareas = [(n, r, f) for n in X6.CONJUNTOS for r in range(X6.N_REP) for f in X6.PLIEGUES]
    K.log(f"externos v2: {len(tareas)} pliegues; {N_TRABAJOS} procesos")
    for n, r, f, f1 in Parallel(n_jobs=N_TRABAJOS, return_as="generator_unordered")(
            delayed(tarea)(*t) for t in tareas):
        K.log(f"  {n} EPL v2 r{r} f{f}: F1 {f1:.4f}")
    resumen()
    K.log("FIN externos v2")


if __name__ == "__main__":
    main()
