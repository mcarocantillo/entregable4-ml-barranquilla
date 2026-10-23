# -*- coding: utf-8 -*-
"""Reúne en ``data/dashboard/`` lo que muestra la página Modelos (el dashboard no recalcula nada de esto).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

- ``modelos.json``: tabla de líderes (25 pliegues + test + tiempos), pruebas de hipótesis de las dos versiones,
  diagrama de diferencia crítica, reglas estándar y ablación, test confirmatorio, diagnósticos por modelo,
  importancia global (media de |valor de Shapley| por variable) y el campo de pesos del EPL (anclas y pesos).
- ``ejemplos.parquet``: 300 viviendas del test (50 por estrato) con sus variables y su estrato real, para la
  demostración de la predicción en vivo.
- ``centro.json``: centro (x0, y0) que reproduce ``dist_centro_km`` a partir de x_km e y_km (mínimos cuadrados), para
  calcular la distancia al centro de una vivienda nueva.

Uso (desde la raíz, después de 09):  python -m experimentos.10_datos_modelos_dashboard
"""
import json
import pickle
import warnings

import joblib
import numpy as np
from scipy.optimize import least_squares

from experimentos import comun as K
from estrato3 import evaluacion as EV
from src.explicacion import shapley_muestreo

DASH = K.RAIZ / "data" / "dashboard"
ORIGINAL = "EPL (original)"


def leer(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))


def importancia_global(modelo, X, fondo, n=80, T=15):
    """Media de |φ| de la probabilidad de la clase predicha, en n viviendas del test."""
    def predecir(df):
        return EV.proba_completa(modelo, df)
    clases = predecir(X).argmax(1)
    tot = {c: 0.0 for c in X.columns}
    for i in range(n):
        phi, _, _ = shapley_muestreo(predecir, X.iloc[[i]], fondo, int(clases[i]), n_permutaciones=T, semilla=i)
        for c, v in phi.items():
            tot[c] += abs(v) / n
    return dict(sorted(tot.items(), key=lambda t: -t[1]))


def main():
    warnings.filterwarnings("ignore")
    meta = leer(K.RAIZ / "artifacts" / "metadata.json")
    version = meta["version_modelo_original"]
    est = {v: leer(K.RES / f"estadistica_{v}.json") for v in ("v1", "v2")}
    tc = leer(K.RAIZ / "resultados" / "test" / "test_confirmatorio.json")
    diag = leer(K.RAIZ / "resultados" / "test" / "diagnosticos.json")
    cols = [c["columna"] for c in meta["esquema_entrada"]]
    d = K.datos()
    sub = d["sub"]
    with open(K.DATOS_E3 / "test_clasificacion.pkl", "rb") as fh:
        test = pickle.load(fh)

    # tabla de líderes
    cv = {x["modelo"]: x for x in est[version]["resumen"]}
    lideres = []
    for nombre, m in meta["modelos"].items():
        lideres.append({"modelo": nombre, "grupo": m.get("grupo", "original"), "balanceo": m.get("balanceo", "—"),
                        "optimizador": m.get("metodo", "Bayesiana"),
                        "f1_cv": cv[nombre]["media"], "f1_cv_sd": cv[nombre]["sd"],
                        "rango_friedman": cv[nombre]["rango_friedman"], "f1_test": tc["f1_macro"][nombre],
                        "ms_inferencia": m["ms_inferencia_por_vivienda"], "tamano_MB": m["tamano_MB"],
                        "archivo": m["archivo"]})

    # importancia global (mismo estimador para todos los modelos)
    rng = np.random.default_rng(K.SEED)
    X_imp = test[cols].iloc[rng.choice(len(test), 80, replace=False)]
    fondo = sub[cols].iloc[rng.choice(len(sub), 50, replace=False)]
    importancia = {}
    for nombre, m in meta["modelos"].items():
        importancia[nombre] = importancia_global(joblib.load(K.RAIZ / "artifacts" / m["archivo"]), X_imp, fondo)
        K.log(f"importancia global: {nombre}")

    # campo de pesos del EPL final
    epl = joblib.load(K.RAIZ / "artifacts" / meta["modelos"][ORIGINAL]["archivo"]).named_steps["epl"]
    campo = {"espacio": epl.espacio, "anclas": epl.anclas_.round(3).tolist(), "pesos": epl.pesos_.round(4).tolist(),
             "pesos_globales": epl.pesos_globales_.round(4).tolist(), "expertos": epl.nombres_expertos_,
             "historia": epl.historia_, "params": meta["modelos"][ORIGINAL]["params"]}

    v2 = [leer(K.RES / "epl_v2" / f"r{r}_f{f}.json") for r in range(5) for f in range(5)]
    salida = {"version_original": version, "lideres": lideres,
              "estadistica": {v: {k: est[v][k] for k in ("friedman", "epl_vs_benchmarks", "epl_vs_variantes",
                                                          "veredicto_validacion", "resumen")} for v in est},
              "espacios_elegidos_v2": {e: sum(x["espacio_elegido"] == e for x in v2)
                                       for e in ("geografico", "variables", "probabilidades")},
              "test": {k: tc[k] for k in ("f1_macro", "ic95_bca_bloques", "epl_menos_mejor", "mcnemar",
                                          "delong_vs_mejor", "mejor_rival_en_validacion", "advertencia")},
              "diagnosticos": diag, "importancia": importancia, "campo_epl": campo}
    K.escribir_json(DASH / "modelos.json", salida)

    # viviendas de ejemplo (50 por estrato) y centro de la ciudad
    # (en pandas 3, groupby().apply() ya no entrega la columna de agrupación: se muestrea con groupby().sample())
    ej = test.groupby("estrato_num").sample(50, random_state=K.SEED)[cols + ["estrato_num"]].copy()
    ej["x_km"], ej["y_km"] = ej.x_km.round(1), ej.y_km.round(1)
    ej.reset_index(drop=True).to_parquet(DASH / "ejemplos.parquet", index=False)
    xy, dist = sub[["x_km", "y_km"]].to_numpy(float), sub.dist_centro_km.to_numpy(float)
    c = least_squares(lambda z: np.hypot(xy[:, 0] - z[0], xy[:, 1] - z[1]) - dist, x0=[0.0, 0.0]).x
    K.escribir_json(DASH / "centro.json", {"x0": float(c[0]), "y0": float(c[1]), "error_medio_km": float(
        np.abs(np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) - dist).mean())})
    K.log("FIN datos del dashboard (Modelos)")


if __name__ == "__main__":
    main()
