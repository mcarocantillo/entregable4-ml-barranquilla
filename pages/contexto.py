# -*- coding: utf-8 -*-
"""Sección 1: Contexto (problema, pregunta, datos, metodología, limitaciones y autores)."""
import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

from src import data as D

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Estrato en Barranquilla", order=0)

VERSION, FECHA = "1.0", "octubre de 2026"


def cifra(valor, etiqueta):
    return dbc.Col(html.Div([html.Div(valor, className="hero-cifra-valor"),
                             html.Div(etiqueta, className="hero-cifra-etiqueta")], className="hero-cifra"),
                   xs=6, md=3, className="mb-2")


def paso(titulo, texto, original=False):
    return dbc.Col(html.Div([html.B(titulo), texto], className="paso original" if original else "paso"),
                   xs=12, sm=6, lg=True, className="mb-2")


hero = html.Div([
    html.Div("Proyecto final · Machine Learning · Universidad del Norte", className="hero-kicker"),
    html.H1("¿Se puede predecir el estrato de una vivienda con el catastro abierto de Barranquilla?",
            className="hero-titulo"),
    html.P("El estrato (1 a 6) define cuánto paga cada hogar por los servicios públicos: los estratos 1 a 3 reciben "
           "subsidio y el 5 y el 6 pagan una contribución. El DANE lo asigna con información que no es pública. "
           "Este proyecto mide cuánto del estrato se recupera solo con los datos abiertos del catastro y la ubicación.",
           className="hero-sub"),
    dbc.Row([cifra("332 718", "viviendas del catastro abierto (2026)"),
              cifra("96.9 %", "ubicadas en el mapa"),
              cifra("0.92", "I de Moran del estrato: vecinos casi iguales"),
              cifra("hasta 50 %", "de subsidio en energía para el estrato 1 (Ley 1450 de 2011)")],
             className="mt-4", style={"maxWidth": "1000px"}),
    html.Div("Contains modified Copernicus Sentinel data (2021), processed by ESA", className="hero-credito"),
], className="hero")

por_que = html.Div([
    html.H3("¿Por qué importa?"),
    html.P(["La Ley 142 de 1994 (arts. 89 y 99) organiza el cobro de los servicios públicos con ",
            html.B("subsidios cruzados"), ": los estratos 1, 2 y 3 reciben subsidio, el 4 paga el costo y el 5 y el 6 "
            "aportan una contribución. Los topes de subsidio son del 50, 40 y 15 % en energía y gas, y del 70, 40 y "
            "15 % en acueducto, alcantarillado y aseo (Ley 1450 de 2011). El estrato también se usa para focalizar "
            "programas sociales."]),
    html.P(["El estrato clasifica ", html.B("inmuebles, no hogares"), ". El DANE lo calcula con zonas homogéneas y "
            "un puntaje de las edificaciones que no son públicos; según el propio DANE, se institucionalizó «sin que "
            "existiera un modelo similar en otros países». Un modelo con datos abiertos serviría para revisar "
            "estratos desactualizados y para proponer uno de referencia donde falta."]),
    html.Div("Fuentes: Ley 142 de 1994; Ley 1450 de 2011; DANE (2015), Estratificación socioeconómica: enfoque "
             "conceptual.", className="fuente"),
], className="tarjeta")

pregunta = html.Div([
    html.H3("Pregunta de investigación y objetivos"),
    html.P(html.B("¿En qué medida las características físicas y catastrales de una vivienda y su ubicación permiten "
                  "predecir su estrato socioeconómico (1 a 6) en Barranquilla?")),
    html.H4("Objetivo general"),
    html.P("Construir y validar, con una validación espacial honesta, un clasificador del estrato a partir del catastro "
           "abierto, y proponer un modelo original que supere de forma estadísticamente significativa a los modelos "
           "de referencia."),
    html.H4("Objetivos específicos"),
    html.Ul([html.Li("Limpiar y ubicar en el mapa las viviendas del catastro, y describir qué variables se asocian al "
                     "estrato (EDA)."),
             html.Li("Comparar 7 modelos de referencia con 4 técnicas de balanceo y 4 métodos de optimización en "
                     "validación anidada con pliegues espaciales (kNNDM)."),
             html.Li("Diseñar, implementar y validar un modelo original: un ensamble con pesos que cambian según la "
                     "zona de la ciudad (EPL)."),
             html.Li("Poner los modelos a disposición en este dashboard, con predicción en vivo y su explicación.")]),
], className="tarjeta")

dicc = D.diccionario()
datos = html.Div([
    html.H3("Datos"),
    dbc.Row([
        dbc.Col([
            html.P(["Fuente: ", html.B("catastro abierto de la Alcaldía de Barranquilla"), " (servicio ArcGIS REST de "
                    "datos abiertos, Gerencia de Gestión Catastral), descargado el 15 de septiembre de 2026. Es una "
                    "foto transversal: no hay serie de tiempo."]),
            html.Ul([html.Li("382 597 registros → 332 718 viviendas tras la limpieza (87 %)."),
                     html.Li("96.9 % con coordenadas (43 594 ubicadas por el centroide de su manzana; error mediano "
                             "31 m)."),
                     html.Li("Entrenamiento: 257 273 viviendas; prueba: 65 145 en 7 bloques de 2 km."),
                     html.Li("Los modelos se compararon en una submuestra estratificada de 10 001 viviendas de "
                             "entrenamiento (la que se explora en la página EDA)."),
                     html.Li(["Variable objetivo: ", html.B("estrato (1 a 6)"), ". Tarea: ", html.B(
                         "clasificación multiclase ordinal"), " con desbalance 6.2 : 1 (estrato 1 = 30 %, estrato "
                         "6 < 5 %)."])]),
        ], md=5),
        dbc.Col(dash_table.DataTable(
            data=dicc, columns=[{"name": n, "id": i} for n, i in
                                [("Variable", "variable"), ("Tipo", "tipo"), ("Unidad", "unidad"),
                                 ("Descripción", "descripcion")]],
            style_table={"overflowX": "auto", "maxHeight": "380px", "overflowY": "auto"},
            style_cell={"fontSize": 12, "textAlign": "left", "whiteSpace": "normal", "height": "auto",
                        "fontFamily": "Inter, Segoe UI, Arial"},
            style_header={"fontWeight": 700}, fixed_rows={"headers": True}, page_action="none"), md=7),
    ]),
], className="tarjeta")

metodologia = html.Div([
    html.H3("Metodología"),
    dbc.Row([
        paso("1. ETL", "Embudo de 6 pasos; ubicación por manzana."),
        paso("2. EDA", "Asociación, faltantes, atípicos, autocorrelación espacial."),
        paso("3. Partición espacial", "Test en 7 bloques de 2 km; pliegues kNNDM que imitan la distancia real."),
        paso("4. 7 modelos × 4 balanceos × 4 optimizadores", "Validación anidada 5 × 3, 12 evaluaciones."),
        paso("5. Comparación", "Friedman → Nemenyi; mejor combinación por modelo."),
        paso("6. Modelo original (EPL)", "Ensamble con pesos por zona; validación anidada repetida 5 × 5.",
             original=True),
        paso("7. Pruebas de hipótesis", "Wilcoxon-Holm, t de Nadeau-Bengio, Friedman; δ de Cliff.", original=True),
    ], className="g-2"),
], className="tarjeta")

cierre = dbc.Row([
    dbc.Col(html.Div([
        html.H3("Limitaciones"),
        html.Ul([html.Li("El estrato es una etiqueta administrativa: puede estar desactualizado y tiene ajustes "
                         "manuales."),
                 html.Li("Faltan los insumos del DANE (puntaje de las edificaciones, zonas homogéneas); por eso los "
                         "errores se concentran entre estratos vecinos."),
                 html.Li("El test tiene solo 7 bloques y todo el estrato 6 cae en uno; la evidencia principal es la "
                         "validación repetida."),
                 html.Li("3.1 % de las viviendas (casi todas de estrato 1 e informales) no se pudieron ubicar.")]),
    ], className="tarjeta h-100"), md=4),
    dbc.Col(html.Div([
        html.H3("Ética y sesgo"),
        html.Ul([html.Li("El modelo no debe usarse para decidir sobre personas (crédito, seguros, acceso a "
                         "programas): el estrato es del inmueble y el modelo se equivoca con frecuencia entre estratos "
                         "vecinos."),
                 html.Li("Sesgo de cobertura: las viviendas sin ubicar son sobre todo informales de estrato 1."),
                 html.Li("Privacidad: sin número predial y con coordenadas redondeadas a ~100 m.")]),
    ], className="tarjeta h-100"), md=4),
    dbc.Col(html.Div([
        html.H3("Autores"),
        html.P([html.B("María Carolina Cantillo Orozco"), " (200179105)", html.Br(),
                html.B("Juan Camilo Oñoro Araujo"), " (200177329)"]),
        html.P(["Profesor: Lihki Rubio", html.Br(), "Machine Learning, Universidad del Norte"]),
        html.P([f"Versión {VERSION} · {FECHA}"], className="fuente"),
    ], className="tarjeta h-100"), md=4),
], className="g-3")

layout = html.Div([hero, por_que, pregunta, datos, metodologia, cierre])
