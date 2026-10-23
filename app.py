# -*- coding: utf-8 -*-
"""Dashboard predictivo del proyecto final: estrato socioeconómico de las viviendas de Barranquilla.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Aplicación multipágina (``use_pages=True``): Contexto (``/``), EDA (``/eda``) y Modelos (``/modelos``).
Local:   python app.py [puerto]   (por defecto http://127.0.0.1:8050 o la variable PORT)
Render:  gunicorn app:server --workers 1 --threads 2 --timeout 120
"""
import os

# Un hilo en BLAS/OpenMP antes de importar numpy: las predicciones coinciden exactamente con las del entorno en que se
# guardaron los modelos (con varios hilos cambia el redondeo y, en el KNN, el desempate de distancias).
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import dash  # noqa: E402
import dash_bootstrap_components as dbc  # noqa: E402
from dash import html  # noqa: E402

# El tema Flatly de Bootstrap va como archivo local en assets/ (no depende de un CDN).
app = dash.Dash(__name__, use_pages=True, title="Estrato en Barranquilla · Proyecto ML",
                suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

barra = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Estrato en Barranquilla", href="/"),
        dbc.Nav([dbc.NavItem(dbc.NavLink("Contexto", href="/", active="exact")),
                 dbc.NavItem(dbc.NavLink("EDA", href="/eda", active="exact")),
                 dbc.NavItem(dbc.NavLink("Modelos", href="/modelos", active="exact"))], navbar=True, className="ms-auto"),
    ], fluid=True),
    className="barra", dark=True)

app.layout = html.Div([barra, dbc.Container(dash.page_container, fluid=True, className="pb-4")])

if __name__ == "__main__":
    import sys
    puerto = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="127.0.0.1", port=puerto)
