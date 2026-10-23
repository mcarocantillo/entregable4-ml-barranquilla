# -*- coding: utf-8 -*-
"""El modelo original frente al modelo base del Entregable 1 (logística B, sin vecindad física).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

El modelo base no es un benchmark de la condición R3 (la guía compara con los 7 modelos base y el mejor de ellos),
pero es el punto de partida del proyecto. Aquí se reproduce tal como lo definió el E1 v3 y lo usó el E3 (logística
multinomial, C = 0.01, sin pesos de clase, sin las variables de vecindad, sin búsqueda) en las mismas 25 particiones
de la validación 5×5 y en el test, y se le aplican al EPL (versión 2) las mismas pruebas que frente a los benchmarks.
Control: en la repetición 0 debe dar los F1 que publicó el E3 (``data/e3/finalistas.json``).

Uso (desde la raíz):  python -m experimentos.04b_modelo_base
"""
import importlib
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from experimentos import comun as K
from estrato3 import estadistica as ES
from estrato3 import evaluacion as EV
from estrato3 import modelos as MD

S = importlib.import_module("experimentos.04_estadistica")
T9 = importlib.import_module("experimentos.09_test_confirmatorio")
SAL = K.RES / "base"


def nuevo(d):
    datos_b = dict(d, num=[c for c in d["num"] if not c.startswith("knn")])
    return Pipeline([("prep", MD.preprocesador(datos_b)), ("modelo", LogisticRegression(C=0.01, max_iter=1000))])


def main():
    warnings.filterwarnings("ignore")
    K.CF.fijar_semillas()
    d = K.datos()
    sub = d["sub"]
    y = sub.estrato_num.to_numpy().astype(int)
    parts = K.particiones()
    f1 = {}
    for r, rep in enumerate(parts):
        for f, (itr, iva) in enumerate(rep["externos"]):
            e = nuevo(d).fit(sub.iloc[itr], y[itr])
            f1[(r, f)] = K.f1_macro(y[iva], e.predict(sub.iloc[iva]))
    publicado = json.loads((K.DATOS_E3 / "finalistas.json").read_text(encoding="utf-8"))["modelo_base"]
    control = max(abs(f1[(0, f)] - publicado["f1_pliegues_externos"][f]) for f in range(5))
    K.log(f"modelo base: F1 25 pliegues {np.mean(list(f1.values())):.4f}; control rep. 0 frente al E3: dif. máx. {control:.2e}")

    T, _, n_val, n_tr = S.tabla_f1()
    T["Modelo base (E1)"] = [f1[(r, f)] for r, f in T.index]
    comp = S.comparar(T, S.ORIGINAL, ["Modelo base (E1)"], n_val, n_tr)[0]

    # test: el base entrenado con la submuestra completa, frente al EPL serializado
    meta = json.loads((K.RAIZ / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    cols = [c["columna"] for c in meta["esquema_entrada"]]
    with open(K.DATOS_E3 / "test_clasificacion.pkl", "rb") as fh:
        test = pickle.load(fh)
    yte = test.estrato_num.to_numpy().astype(int)
    bloques = test.bloque.to_numpy()
    p_base = EV.proba_completa(nuevo(d).fit(sub[cols], y), test[cols])
    p_epl = np.load(K.RAIZ / "resultados" / "test" / "proba_test.npz")["EPL (original)"].astype(float)
    f1_te = {"Modelo base (E1)": K.f1_macro(yte, K.pred(p_base), labels=K.CLASES),
             "EPL (original)": K.f1_macro(yte, K.pred(p_epl), labels=K.CLASES)}
    inf = bloques != T9.BLOQUE_DIMINUTO
    yi, bi = yte[inf], bloques[inf]
    ic = ES.bootstrap_bca_bloques(lambda idx: K.f1_macro(yi[idx], K.pred(p_epl[inf][idx]), labels=K.CLASES) -
                                  K.f1_macro(yi[idx], K.pred(p_base[inf][idx]), labels=K.CLASES), bi)
    mc = T9.mcnemar(K.pred(p_epl) == yte, K.pred(p_base) == yte)
    dif_b = [K.f1_macro(yte[bloques == b], K.pred(p_epl[bloques == b])) -
             K.f1_macro(yte[bloques == b], K.pred(p_base[bloques == b])) for b in np.unique(bi)]

    salida = {"descripcion": "logística multinomial C = 0.01, sin pesos, sin vecindad física (E1 v3 / E3)",
              "f1_por_pliegue": {f"r{r}_f{f}": v for (r, f), v in f1.items()},
              "f1_medio_25": float(np.mean(list(f1.values()))), "control_rep0_dif_max": control,
              "epl_vs_base": comp, "test": {"f1": f1_te, "dif": f1_te["EPL (original)"] - f1_te["Modelo base (E1)"],
                                           "ic95_bca_bloques": ic, "bloques_a_favor": int(sum(x > 0 for x in dif_b)),
                                           "n_bloques": len(dif_b), "mcnemar": mc}}
    K.escribir_json(SAL / "base_vs_epl.json", salida)
    print(pd.DataFrame([{k: comp[k] for k in ("dif_media", "IC95_BCa", "pliegues_a_favor", "wilcoxon_p", "t_NB_p",
                                              "delta_cliff", "delta_magnitud", "d_cohen_pareada")}]).to_string())
    print("test:", {k: round(v, 4) for k, v in f1_te.items()}, "IC dif", np.round(ic, 4), "McNemar", mc)


if __name__ == "__main__":
    main()
