# -*- coding: utf-8 -*-
"""Métodos recientes del estado del arte (LightGBM y CatBoost) en las mismas 25 particiones de la 5×5.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Misma tarea que ``02_benchmarks_5x5.py`` (búsqueda de 12 evaluaciones en los 3 pliegues internos, reajuste y
evaluación externa), en las 5 repeticiones: no existen en el E3. Balanceo por pesos de clase y búsqueda bayesiana.

Uso (desde la raíz):  python -m experimentos.02b_recientes_5x5      (N_TRABAJOS=6 por defecto)
"""
import importlib
import json
import os

import numpy as np
from joblib import Parallel, delayed

from experimentos import comun as K

B = importlib.import_module("experimentos.02_benchmarks_5x5")
N_TRABAJOS = int(os.environ.get("N_TRABAJOS", 6))


def main():
    tareas = [(m, b, o, r, f) for m, b, o in reversed(K.SOTA) for r in range(K.N_REPETICIONES) for f in range(5)]
    pendientes = [t for t in tareas if not K.ruta_benchmark(t[0], t[3], t[4]).exists()]
    K.log(f"{len(tareas)} tareas recientes; pendientes {len(pendientes)}; {N_TRABAJOS} procesos")
    for modelo, r, f, f1, seg in Parallel(n_jobs=N_TRABAJOS, return_as="generator_unordered")(
            delayed(B.tarea)(*t) for t in pendientes):
        K.log(f"  {modelo} r{r} f{f}: F1 {f1:.4f} ({seg:.0f} s de búsqueda)")
    for m, _, _ in K.SOTA:
        f1 = [json.loads(K.ruta_benchmark(m, r, f).read_text(encoding="utf-8"))["metricas"]["f1_macro"]
              for r in range(K.N_REPETICIONES) for f in range(5)]
        K.log(f"{m}: F1 25 pliegues {np.mean(f1):.4f} ± {np.std(f1, ddof=1):.4f}")
    K.log("FIN recientes")


if __name__ == "__main__":
    main()
