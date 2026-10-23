# -*- coding: utf-8 -*-
"""Búsqueda en arXiv (API pública export.arxiv.org) de las cadenas del protocolo; guarda conteos y registros.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

arXiv busca con ``all:`` (título, resumen, autores, comentarios); las cadenas son las del protocolo adaptadas a su
sintaxis (sin comodines). Se espera 3 s entre consultas (norma de uso de la API).

Uso (desde la raíz):  python literatura/buscar_arxiv.py
"""
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

AQUI = Path(__file__).resolve().parent
NS = {"a": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/"}


def o(*t):
    return "(" + " OR ".join(f'all:"{x}"' if " " in x else f"all:{x}" for x in t) + ")"


CADENAS = {
    "C1": f'{o("socioeconomic stratum", "socioeconomic strata", "socioeconomic stratification")} AND '
          f'{o("machine learning", "classification", "ensemble")}',
    "C2": f'{o("locally weighted ensemble", "spatially varying weights", "geographically weighted ensemble", "dynamic ensemble selection", "mixture of experts")} AND '
          f'{o("socioeconomic", "housing", "cadastral")}',
    "C3": f'{o("exponentiated gradient", "multiplicative weights", "hedge algorithm")} AND '
          f'{o("ensemble", "mixture weights", "classifier combination")} AND {o("spatial", "geographic", "local")}',
    "C4": f'all:"geographically weighted" AND {o("ensemble", "stacking", "model averaging")} AND '
          f'{o("classification", "classifier")}',
    "C5": f'{o("dynamic ensemble selection", "dynamic classifier selection", "KNORA")} AND '
          f'{o("spatial", "geographic", "remote sensing", "urban", "housing")}',
}


def consultar(q, n=100):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "start": 0, "max_results": n, "sortBy": "relevance"})
    with urllib.request.urlopen(url, timeout=60) as r:
        raiz = ET.fromstring(r.read())
    total = int(raiz.find("os:totalResults", NS).text)
    regs = [{"id": e.find("a:id", NS).text, "titulo": " ".join(e.find("a:title", NS).text.split()),
             "anio": e.find("a:published", NS).text[:4],
             "resumen": " ".join(e.find("a:summary", NS).text.split())} for e in raiz.findall("a:entry", NS)]
    return total, regs


def main():
    salida = {"fecha": dt.date.today().isoformat(), "base": "arXiv (API export.arxiv.org)", "cadenas": {}}
    for c, q in CADENAS.items():
        total, regs = consultar(q)
        salida["cadenas"][c] = {"consulta": q, "resultados": total, "registros": regs}
        print(f"{c}: {total} resultados ({len(regs)} descargados)", flush=True)
        time.sleep(3)
    (AQUI / "arxiv_resultados.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
