# -*- coding: utf-8 -*-
"""Carga y caché de los datos y artefactos que usa el dashboard (se leen una sola vez por proceso).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

El dashboard nunca entrena ni busca hiperparámetros: solo lee ``data/dashboard/`` (datos livianos y resultados ya
calculados por ``experimentos/``) y ``artifacts/`` (modelos serializados con su ``metadata.json``).
Las rutas salen de la raíz del repositorio (no hay rutas absolutas en el código).
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DASH = RAIZ / "data" / "dashboard"
ARTEFACTOS = RAIZ / "artifacts"

NUMERICAS = ["area_construida", "area_catastral_terreno", "total_habitaciones", "total_banios", "total_plantas",
             "antiguedad", "planta_ubicacion", "altura", "x_km", "y_km", "dist_centro_km", "knn8_log_area",
             "knn8_banios", "knn8_habitaciones", "knn8_antiguedad", "knn8_piso", "knn8_prop_PH", "knn8_dist_km"]
CATEGORICAS = ["tipo_vivienda", "uso", "condicion_predio"]
SESGADAS = {"area_construida", "area_catastral_terreno", "total_habitaciones", "total_banios", "total_plantas",
            "planta_ubicacion", "antiguedad"}


@lru_cache(maxsize=1)
def viviendas():
    """Submuestra de entrenamiento (10 001 viviendas) con variables, ubicación redondeada y estrato."""
    return pd.read_parquet(DASH / "viviendas.parquet")


@lru_cache(maxsize=1)
def diccionario():
    """Lista de dicts: variable, tipo, unidad, descripción."""
    return json.loads((DASH / "diccionario.json").read_text(encoding="utf-8"))


def unidades():
    return {d["variable"]: d["unidad"] for d in diccionario()}


@lru_cache(maxsize=None)
def json_opcional(nombre):
    """JSON de ``data/dashboard/<nombre>`` o None si todavía no se generó (la página muestra un aviso)."""
    ruta = DASH / nombre
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def metadata_artefactos():
    ruta = ARTEFACTOS / "metadata.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


@lru_cache(maxsize=None)
def artefacto(nombre_archivo):
    """Modelo serializado (joblib), cargado la primera vez que se pide y guardado en memoria."""
    import joblib
    return joblib.load(ARTEFACTOS / nombre_archivo)
