# -*- coding: utf-8 -*-
"""Datos livianos que lee el dashboard (``data/dashboard/``). El dashboard nunca entrena ni busca hiperparámetros.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

- ``viviendas.parquet``: la submuestra de entrenamiento (10 001 viviendas) con las variables del modelo y el estrato.
  Sin número predial y con las coordenadas redondeadas a 0.1 km (~100 m), como en el dashboard del E2.
- ``diccionario.json``: nombre, tipo, unidad y descripción de cada variable (del capítulo 1 del E1).

Uso (desde la raíz):  python -m experimentos.08_datos_dashboard
"""
from experimentos import comun as K

DICCIONARIO = [
    ("area_construida", "numérica", "m²", "Área construida de la vivienda"),
    ("area_catastral_terreno", "numérica", "m²", "Área del terreno (en propiedad horizontal, la cuota del lote)"),
    ("total_habitaciones", "numérica", "—", "Número de habitaciones"),
    ("total_banios", "numérica", "—", "Número de baños"),
    ("total_plantas", "numérica", "—", "Pisos de la propia vivienda (1 en casi todos los apartamentos)"),
    ("antiguedad", "numérica", "años", "2026 − año de construcción registrado"),
    ("planta_ubicacion", "numérica", "piso", "Piso en que está la unidad (hasta 41)"),
    ("altura", "numérica", "m", "Altura registrada (3 m en el 99.7 % de las viviendas)"),
    ("x_km", "espacial", "km", "Coordenada este-oeste respecto al centro"),
    ("y_km", "espacial", "km", "Coordenada sur-norte respecto al centro"),
    ("dist_centro_km", "espacial", "km", "Distancia al centro histórico"),
    ("knn8_log_area", "vecindad", "log m²", "Área construida media (log) de los 8 edificios más cercanos"),
    ("knn8_banios", "vecindad", "—", "Baños medios de los 8 edificios más cercanos"),
    ("knn8_habitaciones", "vecindad", "—", "Habitaciones medias de los 8 edificios más cercanos"),
    ("knn8_antiguedad", "vecindad", "años", "Antigüedad media de los 8 edificios más cercanos"),
    ("knn8_piso", "vecindad", "piso", "Piso medio de los 8 edificios más cercanos"),
    ("knn8_prop_PH", "vecindad", "proporción", "Proporción de los 8 edificios más cercanos en propiedad horizontal"),
    ("knn8_dist_km", "vecindad", "km", "Distancia media a los 8 edificios más cercanos"),
    ("tipo_vivienda", "categórica", "—", "tv_1 VIS, tv_2 VIP, tv_3 no VIS, tv_4 no aplica"),
    ("uso", "categórica", "—", "Uso catastral (casa hasta 3 pisos, apartamentos en PH, ...)"),
    ("condicion_predio", "categórica", "—", "Régimen: NPH, propiedad horizontal (PH) o informal"),
    ("estrato_num", "objetivo", "1–6", "Estrato socioeconómico asignado (variable a predecir)"),
]


def main():
    d = K.datos()
    sub = d["sub"]
    cols = [c for c, *_ in DICCIONARIO]
    df = sub[cols].copy()
    df["x_km"], df["y_km"] = df.x_km.round(1), df.y_km.round(1)
    df["estrato_num"] = df.estrato_num.astype(int)
    destino = K.RAIZ / "data" / "dashboard"
    destino.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino / "viviendas.parquet", index=False)
    K.escribir_json(destino / "diccionario.json",
                    [{"variable": c, "tipo": t, "unidad": u, "descripcion": s} for c, t, u, s in DICCIONARIO])
    K.log(f"viviendas.parquet: {df.shape}, {(destino / 'viviendas.parquet').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
