# -*- coding: utf-8 -*-
"""Reentrenamiento final y serialización de los modelos (bloque B.2 de la guía) con prueba de ida y vuelta.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Cada modelo se reentrena con TODA la submuestra de entrenamiento (10 001 viviendas) aplicando el mismo
procedimiento que se validó: los hiperparámetros salen de una búsqueda con el mismo método y presupuesto (12
evaluaciones) usando los 5 pliegues externos de la repetición 0 como validación (así lo hizo el E3 con sus
finalistas; aquí se reutilizan esos hiperparámetros y se buscan los de LightGBM, CatBoost y el modelo original).

- ``artifacts/<modelo>.joblib``: pipeline completo (preprocesamiento del E1 → [SMOTE] → modelo), con
  ``joblib.dump(..., compress=3)``. El modelo original es Pipeline(preprocesamiento + coordenadas → EPL).
- ``artifacts/metadata.json``: versiones de las librerías, semilla, hiperparámetros, esquema de columnas de entrada
  (nombre, tipo, rango válido o categorías), métricas del bucle externo (25 pliegues), tamaño, tiempos de
  entrenamiento e inferencia, y el resultado de la prueba de ida y vuelta.
- ``artifacts/referencia_ida_vuelta.npz``: 2 000 viviendas fijas del test y las probabilidades del modelo en memoria;
  ``tests/test_artefactos.py`` vuelve a cargar cada archivo y exige predicciones idénticas.

Uso (desde la raíz):  python -m experimentos.07_artefactos [v1|v2]      (versión del modelo original; v2 por defecto)
"""
import json
import pickle
import platform
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from experimentos import comun as K
from estrato3 import evaluacion as EV
from estrato3 import modelos as MD
from estrato3 import optimizacion as OP

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
ART = K.RAIZ / "artifacts"
N_REFERENCIA = 2000


def versiones():
    import catboost
    import imblearn
    import lightgbm
    import pandas
    import scipy
    import sklearn
    import xgboost
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__, "imbalanced-learn": imblearn.__version__,
            "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
            "joblib": joblib.__version__}


def esquema(d):
    """Columnas de entrada: tipo y rango válido (mínimo–máximo de entrenamiento) o categorías conocidas."""
    sub = d["sub"]
    out = []
    for c in MD.columnas(d):
        if c in d["cat"]:
            out.append({"columna": c, "tipo": "categórica", "categorias": sorted(sub[c].dropna().unique().tolist())})
        else:
            out.append({"columna": c, "tipo": "numérica", "minimo": float(sub[c].min()), "maximo": float(sub[c].max()),
                        "admite_faltante": True})
    return out


def hiperparametros_finales(d, y):
    """Hiperparámetros finales de cada rival: finalistas del E3 y búsqueda nueva para LightGBM y CatBoost."""
    fin = json.loads((K.DATOS_E3 / "finalistas.json").read_text(encoding="utf-8"))["finalistas"]
    ruta = K.RES / "finales_recientes.json"
    nuevos = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    out = {}
    for m, b, o, g in K.benchmarks():
        if m in fin:
            out[m] = {"balanceo": fin[m]["balanceo"], "metodo": fin[m]["metodo"], "params": fin[m]["params"],
                      "grupo": g, "f1_busqueda_final": fin[m]["f1_busqueda_final"]}
            continue
        if m not in nuevos:
            K.log(f"búsqueda final de {m} (12 evaluaciones, pliegues externos de la repetición 0)")
            obj = OP.Objetivo(d, d["sub"][MD.columnas(d)], y, K.particiones()[0]["externos"], m, b)
            OP.METODOS[o](obj, m, presupuesto=K.CF.PRESUPUESTO)
            params, f1 = obj.mejor()
            nuevos[m] = {"params": params, "f1_busqueda_final": f1}
            K.escribir_json(ruta, nuevos)
        out[m] = {"balanceo": b, "metodo": o, "grupo": g, **nuevos[m]}
    return out


def epl_final(d, y, finales):
    """Hiperparámetros del modelo original con la misma búsqueda (pliegues externos de la repetición 0)."""
    import importlib
    V2 = importlib.import_module("experimentos.03b_epl_v2_5x5")
    ruta = K.RES / f"final_epl_{VERSION}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    combos = K.mejores_combinaciones(incluir_extra=False)
    expertos = [(c.modelo, K.experto(c.modelo, finales[c.modelo]["balanceo"], finales[c.modelo]["params"]))
                for _, c in combos.iterrows()]
    z = {}
    for i, (itr, iva) in enumerate(K.particiones()[0]["externos"]):
        Xa, Xb = K.matrices(d, itr, iva)
        P, _, Pb = V2.etapa(expertos, Xa, y[itr], Xb)
        z.update({f"i{i}_Xa": Xa, f"i{i}_ya": y[itr], f"i{i}_Poof": P, f"i{i}_Xb": Xb, f"i{i}_yb": y[iva],
                  f"i{i}_Pb": Pb})
        K.log(f"  probabilidades de los expertos, pliegue {i}")
    mejor, hist = V2.buscar(z, len(expertos), "geografico" if VERSION == "v1" else None)
    salida = {"version": VERSION, "params": mejor["params"], "f1_busqueda_final": mejor["f1_interno"],
              "historial": hist}
    K.escribir_json(ruta, salida)
    return salida


def main():
    warnings.filterwarnings("ignore")
    K.CF.fijar_semillas()
    d = K.datos()
    sub, cols = d["sub"], MD.columnas(d)
    y = sub.estrato_num.to_numpy().astype(int)
    with open(K.DATOS_E3 / "test_clasificacion.pkl", "rb") as fh:
        test = pickle.load(fh)
    ref = test.iloc[np.random.default_rng(K.SEED).choice(len(test), N_REFERENCIA, replace=False)]
    cv = json.loads((K.RES / f"estadistica_{VERSION}.json").read_text(encoding="utf-8"))
    medias = {x["modelo"]: (x["media"], x["sd"]) for x in cv["resumen"]}
    finales = hiperparametros_finales(d, y)
    ART.mkdir(exist_ok=True)
    meta = {"proyecto": "Estrato socioeconómico en Barranquilla — proyecto final", "semilla": K.SEED,
            "version_modelo_original": VERSION, "librerias": versiones(), "entrenamiento": {
                "filas": len(sub), "descripcion": "submuestra estratificada de entrenamiento del E3 (10 001 viviendas)"},
            "esquema_entrada": esquema(d), "clases": K.CLASES.tolist(), "modelos": {}}
    referencia = {"X_indices_test": ref.index.to_numpy()}

    def guardar(nombre, est, t_fit, extra):
        archivo = f"{K.slug(nombre)}.joblib"
        t0 = time.perf_counter()
        p_mem = EV.proba_completa(est, ref[cols])
        t_inf = (time.perf_counter() - t0) / len(ref)
        joblib.dump(est, ART / archivo, compress=3)
        p_disco = EV.proba_completa(joblib.load(ART / archivo), ref[cols])
        referencia[archivo] = p_mem
        m, s = medias.get(nombre, (None, None))
        meta["modelos"][nombre] = {"archivo": archivo, "tamano_MB": round((ART / archivo).stat().st_size / 1e6, 2),
                                   "seg_entrenamiento": round(t_fit, 1), "ms_inferencia_por_vivienda": 1000 * t_inf,
                                   "f1_bucle_externo_media": m, "f1_bucle_externo_sd": s,
                                   "ida_y_vuelta_identica": bool(np.array_equal(p_mem, p_disco)), **extra}
        K.log(f"{nombre}: {meta['modelos'][nombre]['tamano_MB']} MB, ida y vuelta "
              f"{'idéntica' if meta['modelos'][nombre]['ida_y_vuelta_identica'] else 'DISTINTA'}")

    for nombre, info in finales.items():
        t0 = time.perf_counter()
        est = MD.pipeline(d, nombre, info["params"], info["balanceo"]).fit(sub[cols], y)
        guardar(nombre, est, time.perf_counter() - t0, {k: info[k] for k in ("balanceo", "metodo", "params", "grupo",
                                                                            "f1_busqueda_final")})

    # modelo original: preprocesamiento + coordenadas sin escalar → EPL con los 7 expertos finalistas
    cfg = epl_final(d, y, finales)
    import importlib
    V2 = importlib.import_module("experimentos.03b_epl_v2_5x5")
    combos = K.mejores_combinaciones(incluir_extra=False)
    expertos = [(c.modelo, K.experto(c.modelo, finales[c.modelo]["balanceo"], finales[c.modelo]["params"]))
                for _, c in combos.iterrows()]
    epl = V2.modelo(cfg["params"], len(expertos))
    epl.set_params(expertos=expertos)
    t0 = time.perf_counter()
    # el preprocesador con con_xy=True agrega x_km e y_km sin escalar al final: el EPL los toma como ubicación
    est = Pipeline([("prep", MD.preprocesador(d, con_xy=True)), ("epl", epl)]).fit(sub[cols], y)
    guardar("EPL (original)", est, time.perf_counter() - t0,
            {"version": VERSION, "params": cfg["params"], "f1_busqueda_final": cfg["f1_busqueda_final"],
             "expertos": [n for n, _ in expertos], "historia_perdida": est.named_steps["epl"].historia_,
             "pesos_globales": est.named_steps["epl"].pesos_globales_.tolist()})
    np.savez_compressed(ART / "referencia_ida_vuelta.npz", **referencia)
    K.escribir_json(ART / "metadata.json", meta)
    K.log("FIN artefactos")


if __name__ == "__main__":
    main()
