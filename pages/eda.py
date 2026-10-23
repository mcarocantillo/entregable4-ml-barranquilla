# -*- coding: utf-8 -*-
"""Sección 2: análisis exploratorio con filtros globales, explorador bivariado e interpretación de cada gráfico."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html
from scipy import stats

from src import data as D
from src import plots as P

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · Estrato en Barranquilla", order=1)

DF = D.viviendas()
UNI = D.unidades()
ETIQ = {d["variable"]: f"{d['variable']} ({d['unidad']})" if d["unidad"] != "—" else d["variable"]
        for d in D.diccionario()}
AREA_MAX = float(np.ceil(DF.area_construida.quantile(0.995) / 50) * 50)
Y_MIN, Y_MAX = float(np.floor(DF.y_km.min())), float(np.ceil(DF.y_km.max()))
CONDICIONES = sorted(DF.condicion_predio.dropna().unique())


def tarjeta(titulo_id, grafico_id, insight_id, ancho=6, extra=None):
    hijos = ([extra] if extra is not None else []) + [dcc.Loading(dcc.Graph(id=grafico_id, config={"displaylogo": False})),
                                                      html.Div(id=insight_id, className="insight")]
    return dbc.Col(html.Div(hijos, className="tarjeta"), xs=12, lg=ancho)


def control(etiqueta, componente, ancho=3):
    return dbc.Col([html.Label(etiqueta), componente], xs=12, sm=6, lg=ancho, className="mb-2")


filtros = html.Div(dbc.Row([
    control("Estratos", dcc.Dropdown(id="f-estrato", options=[{"label": f"Estrato {k}", "value": k} for k in range(1, 7)],
                                     value=list(range(1, 7)), multi=True, clearable=False)),
    control("Régimen del predio", dcc.Dropdown(id="f-condicion", options=CONDICIONES, value=CONDICIONES, multi=True,
                                               clearable=False)),
    control("Área construida (m²)", dcc.RangeSlider(id="f-area", min=0, max=AREA_MAX, step=10, value=[0, AREA_MAX],
                                                    marks={0: "0", int(AREA_MAX): f"{int(AREA_MAX)}"},
                                                    tooltip={"placement": "bottom", "always_visible": False})),
    control("Posición sur → norte (km)", dcc.RangeSlider(id="f-norte", min=Y_MIN, max=Y_MAX, step=0.5,
                                                         value=[Y_MIN, Y_MAX], marks={Y_MIN: "sur", Y_MAX: "norte"},
                                                         tooltip={"placement": "bottom", "always_visible": False})),
], className="g-2 align-items-end"), className="tarjeta filtros")

selector_var = dbc.Row([
    dbc.Col(dcc.Dropdown(id="v-numerica", options=[{"label": ETIQ[c], "value": c} for c in D.NUMERICAS],
                         value="area_construida", clearable=False), md=8),
    dbc.Col(dbc.Checklist(id="v-log", options=[{"label": " escala log", "value": "log"}], value=["log"], switch=True),
            md=4, className="pt-2"),
], className="mb-2")

explorador = dbc.Row([
    dbc.Col([html.Label("Eje x"), dcc.Dropdown(id="b-x", options=[{"label": ETIQ[c], "value": c} for c in D.NUMERICAS],
                                               value="x_km", clearable=False)], md=4),
    dbc.Col([html.Label("Eje y"), dcc.Dropdown(id="b-y", options=[{"label": ETIQ[c], "value": c} for c in D.NUMERICAS],
                                               value="knn8_log_area", clearable=False)], md=4),
    dbc.Col([html.Label("Color"), dcc.Dropdown(id="b-color", options=[{"label": "estrato", "value": "estrato_num"}] +
                                               [{"label": c, "value": c} for c in D.CATEGORICAS] +
                                               [{"label": ETIQ[c], "value": c} for c in D.NUMERICAS],
                                               value="estrato_num", clearable=False)], md=4),
], className="mb-2")

layout = html.Div([
    html.H2("Análisis exploratorio", className="seccion-titulo mt-3"),
    html.P("Submuestra estratificada de 10 001 viviendas de entrenamiento (la misma con la que se compararon los "
           "modelos). Los filtros de arriba actualizan todas las gráficas de la página.", className="fuente"),
    filtros,
    dcc.Store(id="filas-filtradas"),
    html.Div(id="eda-aviso"),
    dbc.Row([tarjeta("t-obj", "g-objetivo", "i-objetivo", 5), tarjeta("t-mapa", "g-mapa", "i-mapa", 7)]),
    dbc.Row([tarjeta("t-hist", "g-hist", "i-hist", 7, extra=selector_var), tarjeta("t-caja", "g-caja", "i-caja", 5)]),
    dbc.Row([tarjeta("t-corr", "g-corr", "i-corr", 7), tarjeta("t-falt", "g-falt", "i-falt", 5)]),
    dbc.Row([tarjeta("t-biv", "g-biv", "i-biv", 12, extra=explorador)]),
])


def filtrar(filas):
    return DF if filas is None else DF.iloc[filas]


@callback(Output("filas-filtradas", "data"), Output("eda-aviso", "children"),
          Input("f-estrato", "value"), Input("f-condicion", "value"), Input("f-area", "value"),
          Input("f-norte", "value"))
def aplicar_filtros(estratos, condiciones, area, norte):
    m = (DF.estrato_num.isin(estratos or []) & DF.condicion_predio.isin(condiciones or []) &
         DF.area_construida.between(area[0], area[1] if area[1] < AREA_MAX else np.inf) &
         DF.y_km.between(norte[0], norte[1]))
    filas = np.flatnonzero(m.to_numpy()).tolist()
    aviso = None
    if len(filas) < 30:
        aviso = dbc.Alert(f"Con estos filtros quedan {len(filas)} viviendas: muy pocas para describir. Amplía los "
                          "filtros.", color="warning")
    return filas, aviso


@callback(Output("g-objetivo", "figure"), Output("i-objetivo", "children"), Input("filas-filtradas", "data"))
def objetivo(filas):
    df = filtrar(filas)
    fig = P.distribucion_objetivo(df)
    if df.empty:
        return fig, ""
    n = df.estrato_num.value_counts()
    mas, menos = n.idxmax(), n.idxmin()
    razon = n.max() / max(n.min(), 1)
    cola = (" Por ese desbalance la métrica principal es el F1 macro, que pesa igual a cada estrato." if razon >= 2
            else " Con estos filtros las clases quedan casi parejas.")
    return fig, [html.B(f"El estrato {mas} es el más frecuente ({100 * n.max() / n.sum():.0f} %)"),
                 f" y el {menos} el menos ({100 * n.min() / n.sum():.1f} %): razón {razon:.1f} : 1.{cola}"]


@callback(Output("g-mapa", "figure"), Output("i-mapa", "children"), Input("filas-filtradas", "data"))
def mapa(filas):
    df = filtrar(filas)
    fig = P.mapa_estratos(df)
    if len(df) < 30:
        return fig, ""
    tercios = np.quantile(df.y_km, [1 / 3, 2 / 3])
    medias = [df.estrato_num[df.y_km <= tercios[0]].mean(),
              df.estrato_num[(df.y_km > tercios[0]) & (df.y_km <= tercios[1])].mean(),
              df.estrato_num[df.y_km > tercios[1]].mean()]
    return fig, [html.B("La ubicación domina:"), f" el estrato medio pasa de {medias[0]:.1f} en el tercio sur a "
                 f"{medias[1]:.1f} en el centro y {medias[2]:.1f} en el norte. Vecinos casi iguales (I de Moran 0.92 en "
                 "todo el catastro): por eso la validación separa zonas, no viviendas. Los huecos cuadrados son los 7 "
                 "bloques de 2 km reservados para la prueba final."]


def eta2_kw(df, var):
    grupos = [g[var].dropna().to_numpy() for _, g in df.groupby("estrato_num") if g[var].notna().sum() > 1]
    if len(grupos) < 2:
        return float("nan")
    h = stats.kruskal(*grupos).statistic
    n, k = sum(len(g) for g in grupos), len(grupos)
    return max(0.0, (h - k + 1) / (n - k))


@callback(Output("g-hist", "figure"), Output("i-hist", "children"), Output("g-caja", "figure"),
          Output("i-caja", "children"), Input("filas-filtradas", "data"), Input("v-numerica", "value"),
          Input("v-log", "value"))
def univariada(filas, var, log):
    df = filtrar(filas)
    log = "log" in (log or []) and DF[var].min() >= 0
    fh = P.histograma(df, var, UNI.get(var, ""), log)
    fc = P.caja_atipicos(df, var, UNI.get(var, ""), log)
    if len(df) < 30:
        return fh, "", fc, ""
    med = df.groupby("estrato_num")[var].median()
    e2 = eta2_kw(df, var)
    tam = "grande" if e2 >= 0.14 else "mediano" if e2 >= 0.06 else "pequeño" if e2 >= 0.01 else "despreciable"
    ih = [html.B(f"η² de Kruskal-Wallis = {e2:.2f} (efecto {tam})."),
          f" La mediana de {var} va de {med.iloc[0]:.3g} en el estrato {med.index[0]} a {med.iloc[-1]:.3g} en el "
          f"{med.index[-1]}."]
    x = df[var].dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    fuera = ((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).mean()
    ic = [html.B(f"{100 * fuera:.1f} % de atípicos de Tukey"),
          " en esta variable. Se conservan (son viviendas reales, sobre todo de estratos 5 y 6) y el modelo solo "
          "recorta el 0.1 % de cada cola (winsorización)." if fuera > 0 else ": ninguno con estos filtros."]
    return fh, ih, fc, ic


@callback(Output("g-corr", "figure"), Output("i-corr", "children"), Output("g-falt", "figure"),
          Output("i-falt", "children"), Input("filas-filtradas", "data"))
def multivariada(filas):
    df = filtrar(filas)
    fc = P.mapa_correlacion(df, D.NUMERICAS)
    ff = P.faltantes(df, D.NUMERICAS + D.CATEGORICAS)
    if len(df) < 30:
        return fc, "", ff, ""
    rho = df[D.NUMERICAS + ["estrato_num"]].corr(method="spearman")["estrato_num"].drop("estrato_num").dropna()
    top = rho.abs().sort_values(ascending=False).index[:3]
    ic = [html.B("Las más asociadas al estrato: "), ", ".join(f"{v} (ρ = {rho[v]:+.2f})" for v in top),
          ". Las variables de vecindad (knn8_*) se asocian más que las de la propia vivienda: el estrato se asigna "
          "por zona."]
    pct = 100 * df[D.NUMERICAS + D.CATEGORICAS].isna().mean()
    if pct.max() > 0:
        if_ = [html.B(f"Pocos faltantes: el máximo es {pct.idxmax()} ({pct.max():.2f} %)."),
               " Se imputan con la mediana del entrenamiento de cada pliegue y una columna indica que faltaba."]
    else:
        if_ = "Ninguna variable tiene faltantes con estos filtros."
    return fc, ic, ff, if_


@callback(Output("g-biv", "figure"), Output("i-biv", "children"), Input("filas-filtradas", "data"),
          Input("b-x", "value"), Input("b-y", "value"), Input("b-color", "value"))
def bivariada(filas, x, y, color):
    df = filtrar(filas)
    fig = P.dispersion(df, x, y, color, ETIQ)
    if len(df) < 30 or x == y:
        return fig, "Elige dos variables distintas." if x == y else ""
    rho = df[[x, y]].corr(method="spearman").iloc[0, 1]
    fuerza = "fuerte" if abs(rho) >= 0.5 else "moderada" if abs(rho) >= 0.3 else "débil"
    return fig, [html.B(f"Spearman ρ = {rho:+.2f} (relación {fuerza})."),
                 f" Cada punto es una vivienda ({min(len(df), 4000):,} de {len(df):,} con estos filtros)."]
