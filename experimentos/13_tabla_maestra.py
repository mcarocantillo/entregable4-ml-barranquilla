# -*- coding: utf-8 -*-
"""Tabla maestra de experimentos (entregable 5 de la guía): una fila por combinación y pliegue externo.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Reúne en ``resultados/tabla_maestra.csv``:

- ``E3``: las 128 combinaciones de clasificación del Entregable 3 (8 modelos × 4 balanceos × 4 optimizadores) en sus
  5 pliegues externos (repetición 0): 640 filas. Las 28 combinaciones de regresión del E3 (140 tareas) quedan en
  ``data/e3/tabla_maestra_regresion.csv``;
- ``5x5``: la mejor combinación de cada modelo en las repeticiones 1–4 (160 filas) y LightGBM y CatBoost en las 5
  repeticiones (50 filas);
- ``original``: el modelo original, versiones 1 y 2, en los 25 pliegues (50 filas).

Columnas comunes: experimento, modelo, balanceo, método, repetición, pliegue, semilla de la partición, semilla,
presupuesto, F1 macro y demás métricas del bucle externo, tiempos (búsqueda, ajuste, inferencia) e hiperparámetros.

Uso (desde la raíz):  python -m experimentos.13_tabla_maestra
"""
import json

import pandas as pd

from experimentos import comun as K

METRICAS = ["f1_macro", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "auc_ovr", "log_loss",
            "brier", "ece", "mae_ordinal", "acc_pm1", "kappa_cuadratico"]


def main():
    filas = []
    e3 = pd.read_csv(K.DATOS_E3 / "tabla_maestra_clasificacion.csv")
    for _, r in e3.iterrows():
        filas.append({"experimento": "E3 (repetición 0)", "modelo": r.modelo, "balanceo": r.balanceo,
                      "metodo": r.metodo, "repeticion": 0, "fold": r.fold, "semilla_particion": K.SEED,
                      "semilla": r.semilla, "presupuesto": r.presupuesto, **{m: r[m] for m in METRICAS},
                      "seg_busqueda": r.seg_busqueda, "seg_ajuste_final": r.seg_ajuste_final,
                      "seg_inferencia": r.seg_inferencia, "params": r.params})
    parts = K.particiones()
    for modelo, balanceo, metodo, grupo in K.benchmarks():
        reps = range(K.N_REPETICIONES) if grupo == "reciente" else range(1, K.N_REPETICIONES)
        for r in reps:
            for f in range(5):
                j = json.loads(K.ruta_benchmark(modelo, r, f).read_text(encoding="utf-8"))
                filas.append({"experimento": "5x5 (" + ("reciente" if grupo == "reciente" else "mejor combinación")
                              + ")", "modelo": modelo, "balanceo": balanceo, "metodo": metodo, "repeticion": r,
                              "fold": f, "semilla_particion": parts[r]["semilla"], "semilla": K.SEED,
                              "presupuesto": K.CF.PRESUPUESTO, **{m: j["metricas"][m] for m in METRICAS},
                              "seg_busqueda": j["seg_busqueda"], "seg_ajuste_final": j["seg_ajuste_final"],
                              "seg_inferencia": j["seg_inferencia"], "params": json.dumps(j["params"])})
    for version, carpeta, clave_met in [("v1", "epl", "metricas_EPL"), ("v2", "epl_v2", "metricas_EPL_v2")]:
        for r in range(K.N_REPETICIONES):
            for f in range(5):
                j = json.loads((K.RES / carpeta / f"r{r}_f{f}.json").read_text(encoding="utf-8"))
                filas.append({"experimento": f"original ({version})", "modelo": f"EPL {version}",
                              "balanceo": f"pesos de clase: {j['params'].get('peso_clase')}",
                              "metodo": "Bayesiana (TPE)", "repeticion": r, "fold": f,
                              "semilla_particion": parts[r]["semilla"], "semilla": K.SEED,
                              "presupuesto": K.CF.PRESUPUESTO, **{m: j[clave_met][m] for m in METRICAS},
                              "seg_busqueda": j["seg_busqueda"] + j.get("seg_cache", 0),
                              "seg_ajuste_final": j["seg_ajuste_final"], "seg_inferencia": j.get("seg_inferencia"),
                              "params": json.dumps(j["params"])})
    tabla = pd.DataFrame(filas)
    destino = K.RAIZ / "resultados" / "tabla_maestra.csv"
    tabla.to_csv(destino, index=False)
    K.log(f"tabla maestra: {len(tabla)} filas → {destino.relative_to(K.RAIZ)}")
    print(tabla.groupby("experimento").size().to_string())


if __name__ == "__main__":
    main()
