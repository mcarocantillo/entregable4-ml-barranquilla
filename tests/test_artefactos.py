# -*- coding: utf-8 -*-
"""Prueba de ida y vuelta de los modelos serializados (sección 4.2 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Cada archivo de ``artifacts/`` se vuelve a cargar y debe dar EXACTAMENTE las mismas probabilidades que el modelo en
memoria sobre 2 000 viviendas fijas del test (guardadas por ``experimentos/07_artefactos.py``). Se salta si los
artefactos todavía no existen.
"""
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
ART = RAIZ / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "metadata.json").exists(), reason="sin artefactos todavía")


def _datos():
    meta = json.loads((ART / "metadata.json").read_text(encoding="utf-8"))
    ref = np.load(ART / "referencia_ida_vuelta.npz")
    with open(RAIZ / "data" / "e3" / "test_clasificacion.pkl", "rb") as fh:
        test = pickle.load(fh)
    cols = [c["columna"] for c in meta["esquema_entrada"]]
    return meta, ref, test.loc[ref["X_indices_test"], cols]


def test_metadata_completa():
    meta, _, _ = _datos()
    assert meta["semilla"] == 42 and meta["librerias"]["scikit-learn"]
    for nombre, m in meta["modelos"].items():
        for clave in ("archivo", "tamano_MB", "f1_bucle_externo_media", "params", "ida_y_vuelta_identica"):
            assert clave in m, f"{nombre}: falta {clave}"


@pytest.mark.parametrize("archivo", sorted(p.name for p in ART.glob("*.joblib")) if ART.exists() else [])
def test_ida_y_vuelta(archivo):
    from estrato3 import evaluacion as EV
    _, ref, X = _datos()
    modelo = joblib.load(ART / archivo)
    np.testing.assert_array_equal(EV.proba_completa(modelo, X), ref[archivo])
