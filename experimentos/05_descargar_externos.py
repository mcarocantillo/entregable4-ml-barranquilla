# -*- coding: utf-8 -*-
"""Descarga los tres conjuntos de datos adicionales (Cuadro 1 de la guía: evaluación amplia y justa).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Misma tarea que el problema principal (clasificación multiclase ordinal) y naturaleza distinta:

| Conjunto | Fuente | Objetivo | Ubicación |
|---|---|---|---|
| King County (Seattle, EE. UU.) | OpenML 42092 ``house_sales`` | ``grade`` (calidad de construcción, ordinal) | lat, long |
| California (censo de 1990) | OpenML 537 ``houses`` | valor mediano de la vivienda en quintiles (1–5) | latitude, longitude |
| Vino tinto portugués (vinho verde) | OpenML 40691 ``wine-quality-red`` | calidad (ordinal) | no tiene: el EPL pondera en el plano de las 2 primeras componentes principales |

Se guardan como CSV en ``data/externos/<nombre>/datos.csv`` con la fecha de descarga en ``fuente.json``.

Uso (desde la raíz):  python -m experimentos.05_descargar_externos
"""
import datetime as dt
import json

from sklearn.datasets import fetch_openml

from experimentos import comun as K

CONJUNTOS = {"king_county": 42092, "california": 537, "vino_tinto": 40691}


def main():
    for nombre, id_openml in CONJUNTOS.items():
        carpeta = K.RAIZ / "data" / "externos" / nombre
        if (carpeta / "datos.csv").exists():
            K.log(f"{nombre}: ya descargado")
            continue
        carpeta.mkdir(parents=True, exist_ok=True)
        b = fetch_openml(data_id=id_openml, as_frame=True, data_home=str(carpeta / "_cache"))
        df = b.frame
        df.to_csv(carpeta / "datos.csv", index=False)
        K.escribir_json(carpeta / "fuente.json", {
            "openml_id": id_openml, "nombre_openml": b.details.get("name"), "version": b.details.get("version"),
            "url": f"https://www.openml.org/d/{id_openml}", "licencia": b.details.get("licence"),
            "descargado": dt.date.today().isoformat(), "filas": len(df), "columnas": list(df.columns),
            "objetivo_openml": b.target_names})
        K.log(f"{nombre}: {df.shape} — objetivo OpenML {b.target_names}")


if __name__ == "__main__":
    main()
