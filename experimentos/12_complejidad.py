# -*- coding: utf-8 -*-
"""Complejidad empírica del modelo original (sección 5.4 de la guía; proposición 5.4 del libro).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Con el equipo libre (un solo proceso, hilos de BLAS en 1), en el pliegue r0 f0 y con la configuración que eligió su
bucle interno en la versión 2:

- combinador solo (``fit_con_probabilidades``) al variar n, el número de expertos M y el de anclas J;
- EPL completo (ajuste cruzado + expertos + combinador) al variar n y el número de variables p;
- inferencia por vivienda.

Para cada serie se ajusta la pendiente b de log(tiempo) = a + b·log(tamaño): la teoría predice b ≈ 1 en n para el
combinador (lineal en n) y el costo de los expertos domina el total.

Uso (desde la raíz, con el equipo libre):  python -m experimentos.12_complejidad
"""
import importlib
import json
import platform
import time
import warnings

import numpy as np

from experimentos import comun as K

V2 = importlib.import_module("experimentos.03b_epl_v2_5x5")
SAL = K.RAIZ / "resultados" / "complejidad"
REPETICIONES = 3


def cronometrar(f):
    t = []
    for _ in range(REPETICIONES):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def pendiente(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main():
    warnings.filterwarnings("ignore")
    z = V2.cache(0, 0)
    params = json.loads(V2.ruta(0, 0).read_text(encoding="utf-8"))["params"]
    Xtr, ytr, Poof = z["Xtr"].astype(float), z["ytr"], z["Poof"].astype(float)
    rng = np.random.default_rng(K.SEED)
    filas = []

    # combinador: n, M, J
    for n in (1000, 2000, 4000, len(ytr)):         # el último tamaño es el pliegue completo (≈ 8 000)
        i = rng.choice(len(ytr), n, replace=False)
        s = cronometrar(lambda: V2.modelo(params, 7).fit_con_probabilidades(Xtr[i], ytr[i], Poof[i], None))
        filas.append({"serie": "combinador vs n", "tamano": n, "segundos": s})
    for M in (1, 2, 4, 7):
        s = cronometrar(lambda: V2.modelo(params, M).fit_con_probabilidades(Xtr, ytr, Poof[:, :M], None))
        filas.append({"serie": "combinador vs M", "tamano": M, "segundos": s})
    for J in (5, 10, 20, 40, 80):
        p = dict(params, n_anclas=J)
        s = cronometrar(lambda: V2.modelo(p, 7).fit_con_probabilidades(Xtr, ytr, Poof, None))
        filas.append({"serie": "combinador vs J", "tamano": J, "segundos": s})
    K.log("combinador medido")

    # EPL completo: n y p (una repetición: son minutos)
    d = K.datos()
    itr, iva = K.particiones()[0]["externos"][0]
    Xa, Xb = K.matrices(d, itr, iva)
    y = d["sub"].estrato_num.to_numpy().astype(int)[itr]
    expertos = V2.expertos_del_pliegue(0, 0)
    for n in (1000, 2000, 4000, 8000):
        i = np.sort(rng.choice(len(y), min(n, len(y)), replace=False))
        m = V2.modelo(params, 7).set_params(expertos=expertos)
        t0 = time.perf_counter()
        m.fit(Xa[i], y[i])
        filas.append({"serie": "EPL completo vs n", "tamano": int(len(i)), "segundos": time.perf_counter() - t0})
        t0 = time.perf_counter()
        m.predict_proba(Xb)
        filas.append({"serie": "inferencia por vivienda (ms) vs n de entrenamiento", "tamano": int(len(i)),
                      "segundos": 1000 * (time.perf_counter() - t0) / len(Xb)})
        K.log(f"EPL completo n = {n}")
    p_total = Xa.shape[1] - 2
    for p in (8, 16, p_total):
        cols = list(range(p)) + [Xa.shape[1] - 2, Xa.shape[1] - 1]
        m = V2.modelo(params, 7).set_params(expertos=expertos)
        t0 = time.perf_counter()
        m.fit(Xa[:, cols], y)
        filas.append({"serie": "EPL completo vs p", "tamano": p, "segundos": time.perf_counter() - t0})
        K.log(f"EPL completo p = {p}")

    pend = {}
    for serie in sorted({f["serie"] for f in filas}):
        f = [x for x in filas if x["serie"] == serie]
        pend[serie] = pendiente([x["tamano"] for x in f], [x["segundos"] for x in f])
    lectura = (f"Pendiente log-log del combinador en n: {pend['combinador vs n']:.2f} (teoría: 1); en M: "
               f"{pend['combinador vs M']:.2f}; en J: {pend['combinador vs J']:.2f}. EPL completo en n: "
               f"{pend['EPL completo vs n']:.2f} (domina el costo de los expertos).")
    SAL.mkdir(parents=True, exist_ok=True)
    K.escribir_json(SAL / "complejidad.json", {"equipo": {"procesador": platform.processor(), "python":
                                                          platform.python_version(), "sistema": platform.platform()},
                                               "tabla": filas, "pendientes": pend, "lectura": lectura})
    K.log(lectura)


if __name__ == "__main__":
    main()
