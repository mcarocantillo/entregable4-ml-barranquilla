# -*- coding: utf-8 -*-
"""Ejecuta los capítulos (jupytext .py → .ipynb con salidas) y construye el libro con Jupyter Book.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Uso (desde la raíz del repositorio):  python book/construir_libro.py
Los capítulos leen resultados ya calculados por ``experimentos/`` (``resultados/`` y ``artifacts/``) y vuelven a
producir sus tablas y figuras; los cómputos largos (horas) no se repiten al construir el libro.
"""
import subprocess
import sys
from pathlib import Path

import jupytext
from nbclient import NotebookClient

AQUI = Path(__file__).resolve().parent
for py in sorted(p for p in (AQUI / "notebooks").glob("*.py") if not p.name.startswith("_")):
    nb = jupytext.read(py)
    print(f"Ejecutando {py.name} ...", flush=True)
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(py.parent)}}).execute()
    jupytext.write(nb, py.with_suffix(".ipynb"))
subprocess.run([sys.executable, "-c", "from jupyter_book.cli.main import main; main()", "build", str(AQUI)],
               check=True)
print("Libro en", AQUI / "_build" / "html" / "index.html")
