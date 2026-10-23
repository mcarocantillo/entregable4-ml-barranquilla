# -*- coding: utf-8 -*-
"""Benchmarks en la validación anidada repetida 5×5: la mejor combinación de cada modelo, con su búsqueda interna.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Para cada uno de los 7 modelos de la guía (y el XGBoost geográfico como referencia) se toma su mejor combinación
del E3 (balanceo + optimizador, por la media del F1 externo) y, en cada pliegue externo de cada repetición, se
repite exactamente la tarea del E3: búsqueda de 12 evaluaciones en los 3 pliegues internos, reajuste con el
entrenamiento externo y evaluación en el pliegue externo.

- Repetición 0: se copian las tareas del E3 (mismas particiones y semilla; ``data/e3/tareas``).
- Repeticiones 1–4: se calculan aquí. Cada tarea guarda ``<modelo>_r<r>_f<f>.json`` (hiperparámetros, métricas,
  tiempos, historial) y ``.npz`` (probabilidades del pliegue externo). Si el archivo existe, la tarea se salta
  (se puede interrumpir y relanzar).

Uso (desde la raíz):  python -m experimentos.02_benchmarks_5x5      (N_TRABAJOS=6 por defecto)
"""
import json
import os
import shutil
import time
import warnings

import numpy as np
from joblib import Parallel, delayed

from experimentos import comun as K

N_TRABAJOS = int(os.environ.get("N_TRABAJOS", 6))
# orden: de la más pesada a la más liviana, para que el paralelismo no quede esperando a una sola tarea larga
ORDEN = ["Random Forest", "XGBoost geográfico", "XGBoost", "SVM", "KNN", "Logística L1/L2", "Árbol de decisión",
         "Naive Bayes"]


def copiar_repeticion_0(combos):
    """Copia las 5 tareas del E3 de cada mejor combinación con el nombre de este experimento."""
    for _, c in combos.iterrows():
        for f in range(5):
            h, s = K.EX.nombre_tarea(c.modelo, c.balanceo, c.metodo, f)
            for ext in ("json", "npz"):
                destino = K.ruta_benchmark(c.modelo, 0, f, ext)
                if not destino.exists():
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(K.DATOS_E3 / "tareas" / f"{h}.{ext}", destino)


def tarea(modelo, balanceo, metodo, r, f):
    """Una combinación en el pliegue externo f de la repetición r (misma lógica que ``estrato3.experimento.tarea``)."""
    destino = K.ruta_benchmark(modelo, r, f)
    if destino.exists():
        return modelo, r, f, json.loads(destino.read_text(encoding="utf-8"))["metricas"]["f1_macro"], 0.0
    from estrato3 import evaluacion as EV
    from estrato3 import modelos as MD
    from estrato3 import optimizacion as OP
    K.CF.fijar_semillas()
    d = K.datos()
    rep = K.particiones()[r]
    sub = d["sub"]
    y = sub.estrato_num.to_numpy()
    itr, iva = rep["externos"][f]
    Xtr, Xva = sub.iloc[itr], sub.iloc[iva]
    obj = OP.Objetivo(d, Xtr, y[itr], rep["internos"][f], modelo, balanceo)
    t0 = time.perf_counter()
    extra = OP.METODOS[metodo](obj, modelo, presupuesto=K.CF.PRESUPUESTO)
    t_busqueda = time.perf_counter() - t0
    params, f1_int = obj.mejor()
    t1 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = MD.pipeline(d, modelo, params, balanceo).fit(Xtr, y[itr])
    t_ajuste = time.perf_counter() - t1
    t2 = time.perf_counter()
    proba = EV.proba_completa(est, Xva)
    t_inferencia = time.perf_counter() - t2
    m = EV.metricas(y[iva], proba)
    salida = {"tarea": f"{modelo}|{balanceo}|{metodo}|r{r}|f{f}", "modelo": modelo, "balanceo": balanceo,
              "metodo": metodo, "repeticion": r, "fold": f, "semilla_particion": rep["semilla"],
              "presupuesto": K.CF.PRESUPUESTO, "semilla": K.SEED, "params": params, "f1_interno_mejor": f1_int,
              "metricas": m, "n_evaluaciones": len(obj.historial), "seg_busqueda": t_busqueda,
              "seg_ajuste_final": t_ajuste, "seg_inferencia": t_inferencia, "n_train": len(itr),
              "n_val": len(iva), "historial": obj.historial, "extra": extra}
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(K.ruta_benchmark(modelo, r, f, "npz"), proba=proba.astype(np.float32), iva=iva)
    destino.write_text(json.dumps(salida, ensure_ascii=False, default=float), encoding="utf-8")
    return modelo, r, f, m["f1_macro"], t_busqueda


def main():
    combos = K.mejores_combinaciones()
    K.log("mejores combinaciones:\n" + combos.to_string(index=False))
    copiar_repeticion_0(combos)
    por_modelo = {c.modelo: (c.balanceo, c.metodo) for _, c in combos.iterrows()}
    tareas = [(m, *por_modelo[m], r, f) for m in ORDEN for r in range(1, K.N_REPETICIONES) for f in range(5)]
    pendientes = [t for t in tareas if not K.ruta_benchmark(t[0], t[3], t[4]).exists()]
    K.log(f"{len(tareas)} tareas en las repeticiones 1–4; pendientes {len(pendientes)}; {N_TRABAJOS} procesos")
    for modelo, r, f, f1, seg in Parallel(n_jobs=N_TRABAJOS, return_as="generator_unordered")(
            delayed(tarea)(*t) for t in pendientes):
        K.log(f"  {modelo} r{r} f{f}: F1 {f1:.4f} ({seg:.0f} s de búsqueda)")
    resumen = {}
    for m in por_modelo:
        f1 = [[json.loads(K.ruta_benchmark(m, r, f).read_text(encoding="utf-8"))["metricas"]["f1_macro"]
               for f in range(5)] for r in range(K.N_REPETICIONES)]
        resumen[m] = {"balanceo": por_modelo[m][0], "metodo": por_modelo[m][1], "f1_por_repeticion": f1,
                      "f1_media_25": float(np.mean(f1)), "f1_sd_25": float(np.std(f1, ddof=1)),
                      "f1_media_rep0_E3": float(np.mean(f1[0]))}
        K.log(f"{m}: F1 25 pliegues {resumen[m]['f1_media_25']:.4f} ± {resumen[m]['f1_sd_25']:.4f} "
              f"(E3: {resumen[m]['f1_media_rep0_E3']:.4f})")
    K.escribir_json(K.RES / "benchmarks_resumen.json", resumen)
    K.log("FIN benchmarks")


if __name__ == "__main__":
    main()
