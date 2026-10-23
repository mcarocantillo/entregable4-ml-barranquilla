# -*- coding: utf-8 -*-
"""Particiones de la validación anidada repetida 5×5.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

- Repetición 0: los 5 pliegues externos y sus 3 internos del Entregable 3 (los mismos con que se eligieron las
  mejores combinaciones), copiados tal cual.
- Repeticiones 1–4: kNNDM sobre la misma submuestra con semillas 43–46. El escenario del bucle externo es la
  geometría del test (zonas nuevas); el de cada bucle interno, su pliegue externo (mismas funciones del E3).

La guía (sección 6.2) recomienda repetir la validación externa porque con 5 pliegues las pruebas pareadas tienen
muy poca potencia (el p mínimo de un Wilcoxon con 5 pares es 0.0625).

Uso (desde la raíz):  python -m experimentos.01_particiones
"""
import pickle

import numpy as np

from experimentos import comun as K
from estrato3 import datos as DT


def main():
    if K.PARTICIONES.exists():
        K.log(f"ya existe {K.PARTICIONES.name}; no se rehace")
        return
    d = K.datos()
    _, _, _, V = K.CF.importar_e1()
    with open(K.DATOS_E3 / "test_clasificacion.pkl", "rb") as fh:
        test = pickle.load(fh)
    sub = d["sub"]
    reps = [{"semilla": K.SEED, "externos": d["externos"], "internos": d["internos"], "knndm": d["knndm"],
             "origen": "Entregable 3"}]
    for r in range(1, K.N_REPETICIONES):
        semilla = K.SEED + r
        ext = DT.folds_externos(sub, test, V, seed=semilla)
        internos = [DT.folds_internos(sub, itr, iva, V, seed=semilla) for itr, iva in ext["folds"]]
        reps.append({"semilla": semilla, "externos": ext["folds"], "internos": internos,
                     "knndm": {"q": ext["q"], "W_km": ext["W"], "W_aleatoria_km": ext["W_loo"]},
                     "origen": "kNNDM, semilla nueva"})
        K.log(f"repetición {r}: q = {ext['q']}, W = {ext['W']:.3f} km, tamaños de validación "
              f"{[len(v) for _, v in ext['folds']]}")
    # controles: cada repetición cubre la submuestra una vez y los internos están dentro del entrenamiento externo
    for r, rep in enumerate(reps):
        val = np.concatenate([v for _, v in rep["externos"]])
        assert np.array_equal(np.sort(val), np.arange(len(sub))), f"repetición {r}: los pliegues no cubren todo"
        for (itr, _), internos in zip(rep["externos"], rep["internos"]):
            for tr, va in internos:
                assert max(tr.max(), va.max()) < len(itr)
    K.PARTICIONES.parent.mkdir(parents=True, exist_ok=True)
    with open(K.PARTICIONES, "wb") as fh:
        pickle.dump(reps, fh)
    # cuánto se parecen las repeticiones: proporción de pares de viviendas que comparten pliegue en dos repeticiones
    rng = np.random.default_rng(0)
    i, j = rng.integers(len(sub), size=(2, 20000))
    fold = np.zeros((len(reps), len(sub)), int)
    for r, rep in enumerate(reps):
        for f, (_, iva) in enumerate(rep["externos"]):
            fold[r, iva] = f
    juntos = [(fold[r, i] == fold[r, j]) for r in range(len(reps))]
    acuerdo = [[float(np.mean(juntos[a] == juntos[b])) for b in range(len(reps))] for a in range(len(reps))]
    K.escribir_json(K.RES / "particiones_resumen.json", {
        "repeticiones": [{"semilla": rep["semilla"], "origen": rep["origen"], "q": rep["knndm"]["q"],
                          "W_km": rep["knndm"]["W_km"], "n_validacion": [len(v) for _, v in rep["externos"]]}
                         for rep in reps],
        "acuerdo_pares_mismo_pliegue": acuerdo})
    K.log("particiones guardadas")


if __name__ == "__main__":
    main()
