# -*- coding: utf-8 -*-
"""Test confirmatorio (65 145 viviendas en 7 bloques de 2 km) con los modelos serializados.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

El veredicto R3 se apoya en la validación repetida 5×5 (``04_estadistica.py``); el test se mira una sola vez, con
los modelos ya elegidos y reentrenados (``07_artefactos.py``), como confirmación. Pruebas de la Tabla 3 de la guía
para el conjunto de prueba:

- **McNemar** (aciertos y errores pareados vivienda a vivienda), EPL frente a cada rival, con Holm;
- **DeLong** para el AUC uno-contra-el-resto de cada estrato, EPL frente al mejor rival, con Holm sobre los 6;
- IC 95 % **BCa por bloques** del F1 macro y de la diferencia (las 6 clases siempre en el F1: corrige el sesgo del
  E3, donde las réplicas sin el bloque 4004 perdían los estratos 5 y 6);
- control de dependencia: **permutación por signo** de las diferencias de F1 por bloque (6 bloques informativos;
  el bloque 3002 tiene 3 viviendas y se excluye de los análisis por bloque).

Advertencia que se reporta: McNemar y DeLong suponen observaciones independientes; las viviendas de un mismo
edificio y barrio no lo son (ICC 0.99 en el E1), así que sus p-valores son optimistas. Por eso se acompañan del
IC por bloques y de la permutación por bloques.

Además guarda, para el dashboard, la matriz de confusión, las curvas ROC y precisión-recall y la calibración de cada
modelo (``resultados/test/diagnosticos.json``).

Uso (desde la raíz, después de 07):  python -m experimentos.09_test_confirmatorio
"""
import itertools
import json
import pickle
import warnings

import joblib
import numpy as np
from scipy import stats
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from experimentos import comun as K
from estrato3 import estadistica as ES
from estrato3 import evaluacion as EV

SAL = K.RAIZ / "resultados" / "test"
ORIGINAL = "EPL (original)"
BLOQUE_DIMINUTO = 3002


def mcnemar(acierto_a, acierto_b):
    """McNemar con corrección de continuidad (b + c grande) o binomial exacta (b + c < 25)."""
    b = int(np.sum(acierto_a & ~acierto_b))
    c = int(np.sum(~acierto_a & acierto_b))
    if b + c == 0:
        return {"b": b, "c": c, "estadistico": 0.0, "p": 1.0, "metodo": "sin discordancias"}
    if b + c < 25:
        return {"b": b, "c": c, "estadistico": float(min(b, c)),
                "p": float(stats.binomtest(min(b, c), b + c, 0.5).pvalue), "metodo": "binomial exacta"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "estadistico": float(chi2), "p": float(stats.chi2.sf(chi2, 1)),
            "metodo": "chi cuadrado con corrección de continuidad"}


def curvas_clase(y, p, k, tipo, puntos=60):
    yb = (y == k).astype(int)
    if tipo == "roc":
        x, yv, _ = roc_curve(yb, p)
        area = auc(x, yv)
    else:
        yv, x, _ = precision_recall_curve(yb, p)
        area = auc(x, yv)
    idx = np.unique(np.linspace(0, len(x) - 1, puntos).astype(int))
    return {"x": np.round(x[idx], 4).tolist(), "y": np.round(yv[idx], 4).tolist(), "area": float(area)}


def calibracion(y, p, bins=10):
    pred = p.argmax(1)
    conf = p.max(1)
    acierto = (K.CLASES[pred] == y)
    bordes = np.linspace(0, 1, bins + 1)
    out = {"confianza": [], "acierto": [], "n": []}
    for a, b in zip(bordes[:-1], bordes[1:]):
        m = (conf > a) & (conf <= b)
        if m.sum() >= 20:
            out["confianza"].append(float(conf[m].mean()))
            out["acierto"].append(float(acierto[m].mean()))
            out["n"].append(int(m.sum()))
    return out


def main():
    warnings.filterwarnings("ignore")
    meta = json.loads((K.RAIZ / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    cols = [c["columna"] for c in meta["esquema_entrada"]]
    with open(K.DATOS_E3 / "test_clasificacion.pkl", "rb") as fh:
        test = pickle.load(fh)
    y = test.estrato_num.to_numpy().astype(int)
    bloques = test.bloque.to_numpy()
    SAL.mkdir(parents=True, exist_ok=True)
    ruta_p = SAL / "proba_test.npz"
    if ruta_p.exists():
        P = dict(np.load(ruta_p))
    else:
        P = {}
        for nombre, m in meta["modelos"].items():
            P[nombre] = EV.proba_completa(joblib.load(K.RAIZ / "artifacts" / m["archivo"]), test[cols]).astype(np.float32)
            K.log(f"test: {nombre}")
        np.savez_compressed(ruta_p, **P)
    rivales = [n for n in P if n != ORIGINAL]
    f1 = {n: K.f1_macro(y, K.pred(p), labels=K.CLASES) for n, p in P.items()}
    met = {n: EV.metricas(y, p.astype(float)) for n, p in P.items()}

    informativos = bloques != BLOQUE_DIMINUTO
    yi, bi = y[informativos], bloques[informativos]
    ic = {n: ES.bootstrap_bca_bloques(lambda idx, p=P[n][informativos]: K.f1_macro(
        yi[idx], K.pred(p[idx]), labels=K.CLASES), bi) for n in P}

    # mejor rival = el de mayor F1 medio en la validación 5×5 (no el del test: el test no elige)
    cv = json.loads((K.RES / f"estadistica_{meta['version_modelo_original']}.json").read_text(encoding="utf-8"))
    medias = {x["modelo"]: x["media"] for x in cv["resumen"] if x["modelo"] != ORIGINAL}
    mejor = max(medias, key=medias.get)

    acierto = {n: K.pred(p) == y for n, p in P.items()}
    mc = {n: mcnemar(acierto[ORIGINAL], acierto[n]) for n in rivales}
    for n, p in zip(rivales, ES.holm([mc[n]["p"] for n in rivales])):
        mc[n]["p_holm"] = float(p)
    dl = {int(k): ES.delong(y == k, P[ORIGINAL][:, k - 1], P[mejor][:, k - 1]) for k in K.CLASES}
    for k, p in zip(dl, ES.holm([dl[k]["p"] for k in dl])):
        dl[k]["p_holm"] = float(p)

    # diferencia de F1 por bloque y permutación por signo (6 bloques informativos)
    def f1_bloque(n):
        return {int(b): K.f1_macro(y[bloques == b], K.pred(P[n][bloques == b])) for b in np.unique(bi)}
    fb = {n: f1_bloque(n) for n in P}
    dif_b = np.array([fb[ORIGINAL][b] - fb[mejor][b] for b in fb[ORIGINAL]])
    signos = np.array(list(itertools.product([1, -1], repeat=len(dif_b))))
    p_perm = float(np.mean(np.abs((signos * dif_b).mean(1)) >= abs(dif_b.mean()) - 1e-12))
    ic_dif = ES.bootstrap_bca_bloques(lambda idx: K.f1_macro(yi[idx], K.pred(P[ORIGINAL][informativos][idx]),
                                                             labels=K.CLASES) -
                                      K.f1_macro(yi[idx], K.pred(P[mejor][informativos][idx]), labels=K.CLASES), bi)

    salida = {"n_test": len(y), "bloques": {int(b): int((bloques == b).sum()) for b in np.unique(bloques)},
              "bloque_excluido_por_tamano": BLOQUE_DIMINUTO, "mejor_rival_en_validacion": mejor,
              "f1_macro": f1, "ic95_bca_bloques": ic, "metricas": met, "f1_por_bloque": fb,
              "epl_menos_mejor": {"dif_f1": f1[ORIGINAL] - f1[mejor], "ic95_bca_bloques": ic_dif,
                                  "dif_por_bloque": dif_b.tolist(), "p_permutacion_signo_bloques": p_perm},
              "mcnemar": mc, "delong_vs_mejor": dl,
              "advertencia": "McNemar y DeLong suponen independencia entre viviendas; con ICC 0.99 sus p-valores "
                             "son optimistas. El IC y la permutación por bloques respetan la dependencia."}
    K.escribir_json(SAL / "test_confirmatorio.json", salida)

    diag = {}
    for n, p in P.items():
        p = p.astype(float)
        diag[n] = {"confusion": confusion_matrix(y, K.pred(p), labels=K.CLASES).tolist(),
                   "roc": {int(k): curvas_clase(y, p[:, k - 1], k, "roc") for k in K.CLASES},
                   "pr": {int(k): curvas_clase(y, p[:, k - 1], k, "pr") for k in K.CLASES},
                   "calibracion": calibracion(y, p)}
    K.escribir_json(SAL / "diagnosticos.json", diag)
    for n in sorted(f1, key=f1.get, reverse=True):
        extra = "" if n == ORIGINAL else f" | McNemar p_holm {mc[n]['p_holm']:.2g}"
        K.log(f"{n:>20}: F1 test {f1[n]:.4f} IC {np.round(ic[n], 3)}{extra}")
    K.log(f"EPL − {mejor}: {salida['epl_menos_mejor']}")
    K.log("DeLong: " + ", ".join(f"E{k} {d['diferencia']:+.4f} (p_holm {d['p_holm']:.2g})" for k, d in dl.items()))


if __name__ == "__main__":
    main()
