# Estrato socioeconómico en Barranquilla — proyecto final de Machine Learning

¿En qué medida las características físicas y catastrales de una vivienda y su ubicación permiten predecir su
estrato socioeconómico (1 a 6) en Barranquilla? Este repositorio reúne el entregable final: las mejores
combinaciones de los modelos del Entregable 3 reentrenadas y serializadas, un **modelo original** (ensamble con
pesos locales por gradiente exponenciado, EPL) validado con pruebas de hipótesis, un **dashboard** multipágina en
Dash desplegado en Render y el **informe** en Jupyter Book.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329). Profesor Lihki Rubio,
Machine Learning, Universidad del Norte, octubre de 2026.

- Dashboard: *(URL de Render, se completa al desplegar)*
- Libro: https://mcarocantillo.github.io/entregable4-ml-barranquilla/
- Entregables anteriores: [E1](https://mcarocantillo.github.io/entregable1-ml-barranquilla/v3/) ·
  [E2](https://mcarocantillo.github.io/entregable2-ml-barranquilla/v3/) ·
  [E3](https://mcarocantillo.github.io/entregable3-ml-barranquilla/)

## Estructura

```text
app.py                    dashboard (multipágina) — expone server para gunicorn
pages/                    contexto.py, eda.py, modelos.py
src/original_model.py     clase del modelo original (EPL), compatible con scikit-learn
src/explicacion.py        valores de Shapley por permutaciones (explicación local)
src/data.py, src/plots.py carga con caché y figuras del dashboard
artifacts/                modelos serializados (joblib) + metadata.json + referencia de ida y vuelta
data/                     datos livianos (e3/: submuestra, pliegues y test del E3; dashboard/; externos/)
e1/, estrato3/            código heredado del E1 (preprocesador, kNNDM) y del E3 (catálogo, optimizadores, estadística)
experimentos/             scripts numerados que producen todos los resultados (ver abajo)
resultados/               resultados de los experimentos (JSON, CSV, figuras)
tests/                    pytest: modelo original, explicación local, ida y vuelta de los artefactos
book/                     fuentes del Jupyter Book (capítulos 0–9) y references.bib
literatura/               protocolo y resultados de la búsqueda de novedad
requirements.txt          dependencias del dashboard (versiones exactas); requirements-dev.txt: todo lo demás
render.yaml               configuración de despliegue
```

## Instalación

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt     # en Linux/macOS: .venv/bin/pip
```

Semilla global: **42** (``estrato3/configuracion.py``). Todos los lanzadores fijan
``OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`` para que los resultados sean reproducibles.

## Uso

```bash
python app.py                    # dashboard en http://127.0.0.1:8050
python -m pytest -q tests        # pruebas unitarias
python book/construir_libro.py   # ejecuta los capítulos y construye el libro en book/_build/html
```

## Reproducir los resultados (desde la raíz, en este orden)

| Script | Qué hace | Tiempo aprox. (4 núcleos) |
|---|---|---|
| `experimentos/01_particiones.py` | 5 repeticiones × 5 pliegues kNNDM (la 0 es la del E3) | 3 min |
| `experimentos/02_benchmarks_5x5.py` | Mejor combinación de cada modelo, con su búsqueda interna, en las 25 particiones | 1.2 h |
| `experimentos/02b_recientes_5x5.py` | LightGBM y CatBoost con el mismo protocolo | 50 min |
| `experimentos/03_epl_5x5.py` | Modelo original, versión 1 (campo geográfico) | 1 h |
| `experimentos/03b_epl_v2_5x5.py` | Modelo original, versión 2 (espacio de competencia elegido en el bucle interno) | 1–2 h |
| `experimentos/04_estadistica.py v1` / `v2` | Friedman-Nemenyi, Wilcoxon-Holm, t de Nadeau-Bengio, bayesiana, δ de Cliff, BCa | 1 min |
| `experimentos/05_descargar_externos.py`, `06_externos.py`, `06b_externos_v2.py` | Tres conjuntos adicionales (OpenML) | 3–4 h |
| `experimentos/07_artefactos.py` | Reentrenamiento final, joblib, metadata.json, ida y vuelta | 20 min |
| `experimentos/08_datos_dashboard.py`, `10_datos_modelos_dashboard.py` | Datos livianos del dashboard | 15 min |
| `experimentos/09_test_confirmatorio.py` | Test de 65 145 viviendas: McNemar, DeLong, BCa por bloques | 10 min |
| `experimentos/11_robustez.py` | Sensibilidad, semillas, ruido, tamaño y desbalance | 40 min |
| `experimentos/12_complejidad.py` | Tiempos al variar n, p, M y J (con el equipo libre) | 15 min |

Los lanzadores `experimentos/correr_5x5.sh`, `correr_recientes.sh`, `correr_externos.sh` y `correr_v2.sh` encadenan
estos pasos y se pueden interrumpir y relanzar (cada tarea ya hecha se salta).

## Despliegue en Render

*New → Web Service* (o *Blueprint* con `render.yaml`): Build `pip install -r requirements.txt`, Start
`gunicorn app:server --workers 1 --threads 2 --timeout 120`, variable `PYTHON_VERSION=3.11.15`.

## Datos

Catastro abierto de la Alcaldía de Barranquilla (descarga del 15-sep-2026), procesado en el Entregable 1. Los
conjuntos adicionales vienen de OpenML: King County (42092), California (537) y vino tinto (40691).
