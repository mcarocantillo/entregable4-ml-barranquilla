# -*- coding: utf-8 -*-
"""Utilidades comunes de los experimentos del Entregable 4.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Protocolo (el del Entregable 3, sección 3 de BASE_ENTREGABLES_PREVIOS.md):

- submuestra estratificada de 10 001 viviendas de entrenamiento (``data/e3/datos_clasificacion.pkl``);
- validación anidada kNNDM: 5 pliegues externos × 3 internos, 12 evaluaciones por búsqueda, semilla 42;
- **repetida 5 veces** (sección 6.2 de la guía): la repetición 0 son los pliegues del E3 y las repeticiones 1–4
  son kNNDM con semillas 43–46 (``01_particiones.py``). Todos los modelos usan las mismas 25 particiones;
- métrica principal: F1 macro (misma definición que el E3; en el bootstrap del test, con las 6 clases fijas).

Correr desde la raíz del repositorio: ``python -m experimentos.<script>``.
"""
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import experimento as EX  # noqa: E402
from estrato3 import modelos as MD  # noqa: E402

DATOS_E3 = RAIZ / "data" / "e3"
PARTICIONES = RAIZ / "data" / "particiones_5x5.pkl"
RES = RAIZ / "resultados" / "cv5x5"
CLASES = CF.CLASES
SEED = CF.SEED
N_REPETICIONES = 5
MODELOS_GUIA = MD.MODELOS_GUIA                       # los 7 modelos base de la guía
REFERENCIA_EXTRA = "XGBoost geográfico"             # propuesta del E2/E3: referencia, no modelo original
T0 = time.time()


def log(msg):
    """Imprime con el tiempo transcurrido desde que arrancó el proceso."""
    print(f"[{(time.time() - T0) / 60:7.1f} min] {msg}", flush=True)


def f1_macro(y, pred, labels=None):
    """F1 macro.

    Por defecto, como ``f1_score`` (y como el E3): promedia las clases presentes en ``y`` o en ``pred``. En el
    bootstrap del test se pasa ``labels=CLASES`` para que una réplica sin estratos 5–6 no los saque del promedio.
    """
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="macro", labels=labels, zero_division=0))


def pred(proba):
    """Estrato con mayor probabilidad (columnas = estratos 1..6)."""
    return CLASES[np.argmax(proba, axis=1)]


def datos():
    """Paquete del E3 (submuestra, columnas, decisiones del EDA y módulo ``modelo`` del E1)."""
    return EX._datos()


def particiones():
    """Lista de 5 repeticiones; cada una con ``externos`` [(itr, iva)] e ``internos`` [[(tr, va) relativos a itr]]."""
    with open(PARTICIONES, "rb") as fh:
        return pickle.load(fh)


def mejores_combinaciones(incluir_extra=True):
    """Mejor combinación (balanceo, optimizador) de cada modelo por la media del F1 del bucle externo del E3.

    Devuelve
    --------
    pandas.DataFrame
        Columnas ``modelo``, ``balanceo``, ``metodo``, ``f1_macro_mean``, ``f1_macro_std``; los 7 modelos de la
        guía y, si ``incluir_extra``, el XGBoost geográfico.
    """
    r = pd.read_csv(DATOS_E3 / "resumen_combinaciones.csv")
    modelos = MODELOS_GUIA + ([REFERENCIA_EXTRA] if incluir_extra else [])
    r = r[r.modelo.isin(modelos)]
    mejor = r.loc[r.groupby("modelo").f1_macro_mean.idxmax(),
                  ["modelo", "balanceo", "metodo", "f1_macro_mean", "f1_macro_std"]]
    return mejor.set_index("modelo").loc[modelos].reset_index()


SOTA = [("LightGBM", "class_weight", "Bayesiana"), ("CatBoost", "class_weight", "Bayesiana")]


def benchmarks():
    """Todos los rivales del modelo original: (modelo, balanceo, método, grupo).

    Grupos: ``guia`` (los 7 modelos base, con su mejor combinación del E3), ``referencia`` (XGBoost geográfico)
    y ``reciente`` (LightGBM y CatBoost, agregados en el E4 con pesos de clase y búsqueda bayesiana).
    """
    out = [(c.modelo, c.balanceo, c.metodo, "guia" if c.modelo in MODELOS_GUIA else "referencia")
           for _, c in mejores_combinaciones().iterrows()]
    return out + [(m, b, o, "reciente") for m, b, o in SOTA]


def slug(texto):
    """Nombre de archivo sin tildes ni espacios."""
    import unicodedata
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return "".join(c if c.isalnum() else "_" for c in t).strip("_").lower()


def ruta_benchmark(modelo, r, f, ext="json"):
    return RES / "benchmarks" / f"{slug(modelo)}_r{r}_f{f}.{ext}"


def leer_benchmark(modelo, r, f):
    """Resultado de una tarea benchmark (dict con ``params``, ``metricas``, tiempos) y sus probabilidades."""
    j = json.loads(ruta_benchmark(modelo, r, f).read_text(encoding="utf-8"))
    z = np.load(ruta_benchmark(modelo, r, f, "npz"))
    return j, z["proba"], z["iva"]


def experto(modelo, balanceo, params, seed=SEED):
    """Pipeline [SMOTE | ADASYN] → modelo para una matriz ya preprocesada (mismo orden que ``MD.pipeline``)."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    pasos = []
    if balanceo == "smote":
        pasos.append(("muestreo", SMOTE(k_neighbors=5, random_state=seed)))
    elif balanceo == "adasyn":
        pasos.append(("muestreo", MD.ADASYNPorClase(n_neighbors=5, random_state=seed)))
    pasos.append(("modelo", MD.estimador(modelo, params, balanceo, seed)))
    return Pipeline(pasos)


def matrices(d, filas_tr, filas_ev):
    """Preprocesador del E1 ajustado con ``filas_tr`` de la submuestra; devuelve [X | x_km | y_km] de ambas.

    Las dos últimas columnas son las coordenadas en km sin escalar: el modelo original las usa para ubicar sus
    anclas y se las quita a los expertos (``expertos_usan_ubicacion=False``).
    """
    sub, cols = d["sub"], MD.columnas(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep = MD.preprocesador(d).fit(sub.iloc[filas_tr][cols])
        out = [np.column_stack([prep.transform(sub.iloc[f][cols]), sub.iloc[f][["x_km", "y_km"]].to_numpy()])
               for f in (filas_tr, filas_ev)]
    return out[0].astype(float), out[1].astype(float)


def escribir_json(ruta, obj):
    """JSON legible con tipos de numpy convertidos."""
    def limpio(o):
        if isinstance(o, dict):
            return {str(k): limpio(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [limpio(v) for v in o]
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return limpio(o.tolist())
        return o
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(limpio(obj), ensure_ascii=False, indent=1), encoding="utf-8")
