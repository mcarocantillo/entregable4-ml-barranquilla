# -*- coding: utf-8 -*-
"""Sección 3: Modelos (líderes, predicción en vivo con explicación local, diagnóstico, comparación estadística y
modelo original). Solo lee resultados ya calculados y modelos serializados: no entrena nada."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update
from scipy.spatial import cKDTree

from src import data as D
from src import plots as P
from src.explicacion import shapley_muestreo

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Estrato en Barranquilla", order=2)

R = D.json_opcional("modelos.json")
META = D.metadata_artefactos()
ORIGINAL = "EPL (original)"
NOMBRE_ESPACIO = {"geografico": "geográfico", "variables": "de las variables", "probabilidades": "de las probabilidades"}

PSEUDOCODIGO = """Entrada: datos (X, s, y); M expertos; J anclas; ancho h; k anclas vecinas; λ; espacio de competencia
Salida:  expertos ajustados f_1..f_M; anclas a_1..a_J; pesos W = (w_1..w_J), cada w_j en el símplex Δ_M

1  Bloques espaciales de lado b → B pliegues
2  para cada pliegue b y experto m: ajustar f_m sin el pliegue; q_im ← P_m(y_i | x_i)  (fuera de muestra)
3  reajustar cada f_m con todos los datos
4  z_i ← coordenadas de competencia (geográficas | 2 CP de x | clase esperada y desacuerdo)
5  a_1..a_J ← K-medias(z)                    K_ij ← núcleo gaussiano normalizado sobre las k anclas más cercanas
6  π ← argmin_{w∈Δ_M}  −Σ_i α_i log(wᵀq_i)              (gradiente exponenciado, J = 1)
7  W ← π en cada ancla;  η ← η_0
8  repetir
9      g_j ← −Σ_i α_i K_ij q_i / r_i  +  (λ/J)·log(w_j / π)          con  r_i = Σ_j K_ij w_jᵀ q_i
10     w_j ← w_j ⊙ exp(−η g_j) / Σ (w_j ⊙ exp(−η g_j))                 (para cada ancla j)
11     si L(W) no bajó: deshacer y η ← η/2;  si bajó: η ← 1.5 η
12 hasta que la mejora relativa de L(W) < tol  (o T iteraciones)
Predicción: w(z) = Σ_j K_j(z) w_j;   p(y | x) = Σ_m w_m(z) f_m(y | x)"""

DIFERENCIAS = [
    {"Método": "Promedio simple / votación", "Qué hace": "Pesos iguales", "Qué es nuevo en el EPL":
        "Pesos aprendidos y distintos según la región"},
    {"Método": "Pesos globales por gradiente exponenciado (Helmbold et al., 1997)", "Qué hace":
        "Un solo vector de pesos", "Qué es nuevo en el EPL": "Campo de pesos con núcleo y penalización KL hacia los globales"},
    {"Método": "Stacking lineal (Wolpert, 1992)", "Qué hace": "Regresión sobre las salidas", "Qué es nuevo en el EPL":
        "Pesos en el símplex (interpretables), problema convexo, dependen de la región"},
    {"Método": "Selección dinámica (KNORA, META-DES)", "Qué hace": "Elige modelos por su acierto en vecinos",
     "Qué es nuevo en el EPL": "Pesos suaves que minimizan una pérdida global convexa; ajuste cruzado por bloques"},
    {"Método": "Mezcla de expertos (Jacobs et al., 1991)", "Qué hace":
        "Compuerta y expertos entrenados juntos (no convexo)", "Qué es nuevo en el EPL":
        "Expertos fijos ya validados; óptimo único; compuerta no neuronal"},
    {"Método": "Ensamble con ponderación espacial (Liu, 2024)", "Qué hace":
        "Ponderación geográfica local + explicabilidad", "Qué es nuevo en el EPL":
        "Espacio de competencia elegido por validación; clasificación ordinal multiclase; regla derivada"},
    {"Método": "Combinación convexa por regiones (Zhou y Lin, 2026)", "Qué hace":
        "Mezcla convexa distinta por región (regresión) y diagnóstico de cuándo paga", "Qué es nuevo en el EPL":
        "Clasificación ordinal; regiones suaves con núcleo; encogimiento KL; espacio de competencia elegido por validación"},
    {"Método": "XGBoost geográfico (E2/E3)", "Qué hace": "Mezcla fija α entre global y locales",
     "Qué es nuevo en el EPL": "Pesos aprendidos por ancla y por experto, con cualquier familia de expertos"},
]


def pendiente():
    return html.Div([html.H2("Modelos", className="seccion-titulo mt-3"),
                     dbc.Alert("Los resultados de los modelos se están calculando: esta página se completa cuando "
                               "existan data/dashboard/modelos.json y artifacts/metadata.json.", color="info")])


if R is None or META is None:
    layout = pendiente
else:
    LIDERES = pd.DataFrame(R["lideres"])
    MODELOS = list(LIDERES.sort_values("f1_cv", ascending=False).modelo)
    ESQUEMA = {c["columna"]: c for c in META["esquema_entrada"]}
    VIV = D.viviendas()
    EJEMPLOS = pd.read_parquet(D.DASH / "ejemplos.parquet")
    CENTRO = D.json_opcional("centro.json")
    ARBOL = cKDTree(VIV[["x_km", "y_km"]].to_numpy())
    FONDO = VIV[[c for c in ESQUEMA]].sample(40, random_state=42)
    FISICAS = ["area_construida", "area_catastral_terreno", "total_habitaciones", "total_banios", "total_plantas",
               "antiguedad", "planta_ubicacion", "altura"]
    DERIVADAS = ["dist_centro_km", "knn8_log_area", "knn8_banios", "knn8_habitaciones", "knn8_antiguedad",
                 "knn8_piso", "knn8_prop_PH", "knn8_dist_km"]
    TIPICA = {c: (VIV[c].mode().iloc[0] if ESQUEMA[c]["tipo"] == "categórica" else float(VIV[c].median()))
              for c in ESQUEMA}

    def tabla_lideres():
        df = LIDERES.copy()
        df["F1 (25 pliegues)"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(df.f1_cv, df.f1_cv_sd)]
        df = df.sort_values("f1_cv", ascending=False)
        cols = [("Modelo", "modelo"), ("Grupo", "grupo"), ("Balanceo", "balanceo"), ("Optimizador", "optimizador"),
                ("F1 (25 pliegues)", "F1 (25 pliegues)"), ("Rango medio", "rango_friedman"),
                ("F1 test", "f1_test"), ("ms por vivienda", "ms_inferencia"), ("MB", "tamano_MB")]
        datos = df.assign(rango_friedman=df.rango_friedman.round(2), f1_test=df.f1_test.round(3),
                          ms_inferencia=df.ms_inferencia.round(3)).to_dict("records")
        return dash_table.DataTable(
            data=datos, columns=[{"name": n, "id": i} for n, i in cols], sort_action="native",
            filter_action="native", style_table={"overflowX": "auto"},
            style_cell={"fontSize": 13, "fontFamily": "Inter, Segoe UI, Arial", "padding": "6px"},
            style_data_conditional=[{"if": {"filter_query": f'{{modelo}} = "{ORIGINAL}"'},
                                     "backgroundColor": "#fdeee7", "fontWeight": 700}])

    def campo(c):
        e = ESQUEMA[c]
        etiqueta = html.Label(f"{c} ({D.unidades().get(c, '')})" if D.unidades().get(c, "—") != "—" else c)
        if e["tipo"] == "categórica":
            comp = dcc.Dropdown(id=f"in-{c}", options=e["categorias"], value=TIPICA[c], clearable=False)
        else:
            comp = dbc.Input(id=f"in-{c}", type="number", value=round(TIPICA[c], 2), min=e["minimo"],
                             max=e["maximo"], step="any", size="sm")
        return dbc.Col([etiqueta, comp], xs=6, md=4, lg=3, className="mb-2")

    def veredicto_alerta(version):
        v = R["estadistica"][version]["veredicto_validacion"]
        ok = v["R3_en_validacion"]
        texto = (f"Versión {version}: frente al mejor rival ({v['mejor_benchmark']}), "
                 f"{v['cond2_n_rechazos']} de 3 pruebas rechazan la igualdad con p ajustado < 0.05 y el δ de Cliff "
                 f"es {v['cond3_delta_cliff_vs_mejor']:+.3f}. ")
        texto += ("Se cumple la superioridad estadística (R3)." if ok else
                  "NO se cumple la superioridad estadística (R3): la ventaja media existe pero no es significativa "
                  "en las tres pruebas o su tamaño es despreciable.")
        return dbc.Alert(texto, color="success" if ok else "warning", className="veredicto")

    def tabla_pruebas(version):
        e = R["estadistica"][version]
        mejor = e["veredicto_validacion"]["mejor_benchmark"]
        x = next(c for c in e["epl_vs_benchmarks"] if c["rival"] == mejor)
        fr = e["friedman"]
        filas = [
            {"Prueba": "Friedman → Nemenyi", "H0": "Todos los modelos tienen el mismo rango medio",
             "Supuestos": "Bloques (pliegues) pareados; rangos", "Estadístico": f"χ² = {fr['chi2']:.1f}",
             "p crudo": f"{fr['p']:.1e}", "p ajustado": f"Nemenyi {x['nemenyi_p']:.3f}",
             "Efecto": f"Δ rango vs CD {fr['cd']:.2f}", "Conclusión": "rechaza" if x["nemenyi_p"] < 0.05 else
             "no rechaza (EPL vs mejor)"},
            {"Prueba": "Wilcoxon de rangos con signo", "H0": "Mediana de la diferencia = 0",
             "Supuestos": f"Simetría (asimetría {x['supuestos']['asimetria']:.2f})", "Estadístico": f"W = {x['wilcoxon_W']:.0f}",
             "p crudo": f"{x['wilcoxon_p']:.4f}", "p ajustado": f"Holm {x['wilcoxon_p_holm']:.4f}",
             "Efecto": f"δ Cliff {x['delta_cliff']:+.3f} ({x['delta_magnitud']})",
             "Conclusión": "rechaza" if x["wilcoxon_p_holm"] < 0.05 else "no rechaza"},
            {"Prueba": "t corregida de Nadeau-Bengio", "H0": "Media de la diferencia = 0",
             "Supuestos": f"Normalidad (Shapiro p = {x['supuestos']['shapiro_p']:.2f}); pliegues dependientes",
             "Estadístico": f"t = {x['t_NB']:.2f} (gl {x['t_NB_gl']})", "p crudo": f"{x['t_NB_p']:.4f}",
             "p ajustado": f"Holm {x['t_NB_p_holm']:.4f}", "Efecto": f"d pareada {x['d_cohen_pareada']:.2f}",
             "Conclusión": "rechaza" if x["t_NB_p_holm"] < 0.05 else "no rechaza"},
            {"Prueba": "Bayesiana de rangos con signo (ROPE 0.01)", "H0": "—",
             "Supuestos": "Previa Dirichlet; pliegues intercambiables", "Estadístico": "—",
             "p crudo": "—", "p ajustado": "—",
             "Efecto": f"P(mejor) {x['bayes']['P(EPL mejor)']:.2f}, P(equiv.) {x['bayes']['P(equivalentes)']:.2f}",
             "Conclusión": "probable mejora" if x["bayes"]["P(EPL mejor)"] > 0.95 else "incierta"},
        ]
        return html.Div([
            html.P(f"EPL frente al mejor rival en la validación ({mejor}): diferencia media {x['dif_media']:+.4f}, "
                   f"IC 95 % BCa [{x['IC95_BCa'][0]:+.4f}, {x['IC95_BCa'][1]:+.4f}], gana en {x['pliegues_a_favor']} "
                   f"de {x['n_pliegues']} pliegues. Comparaciones por familia: {len(e['epl_vs_benchmarks'])}.",
                   className="fuente"),
            dash_table.DataTable(data=filas, columns=[{"name": c, "id": c} for c in filas[0]],
                                 style_table={"overflowX": "auto"}, style_cell={"fontSize": 12, "whiteSpace": "normal",
                                                                                "height": "auto", "textAlign": "left",
                                                                                "fontFamily": "Inter, Segoe UI, Arial"})])

    def figura_campo():
        c = R["campo_epl"]
        anclas, pesos = np.array(c["anclas"]), np.array(c["pesos"])
        experto = c["expertos"][int(np.argmax(np.ptp(pesos, axis=0)))]
        fig = P.mapa_pesos(anclas, pesos, c["expertos"], experto)
        if c["espacio"] != "geografico":
            ejes = {"variables": ("Componente principal 1 (escalada)", "Componente principal 2 (escalada)"),
                    "probabilidades": ("Clase esperada por los expertos (escalada)", "Desacuerdo entre expertos")}
            fig.update_layout(xaxis_title=ejes[c["espacio"]][0], yaxis_title=ejes[c["espacio"]][1],
                              yaxis=dict(scaleanchor=None))
        return fig, experto

    fig_campo, experto_variable = figura_campo()
    ver = R["version_original"]
    est_v = R["estadistica"][ver]
    variantes = {x["rival"]: x for x in est_v["epl_vs_variantes"]}
    medias_var = {ORIGINAL: next(x["media"] for x in est_v["resumen"] if x["modelo"] == ORIGINAL)}
    medias_var.update({k: medias_var[ORIGINAL] - v["dif_media"] for k, v in variantes.items()})

    layout = html.Div([
        html.H2("Modelos", className="seccion-titulo mt-3"),
        html.Div([html.H3("Tabla de líderes"),
                  html.P("Mejor combinación de cada modelo (balanceo y optimizador del E3), los recientes LightGBM y "
                         "CatBoost y el modelo original. F1 macro en la validación anidada repetida 5 × 5 (media ± "
                         "desviación de los 25 pliegues externos) y en el test de 7 bloques. Se puede ordenar y filtrar.",
                         className="fuente"),
                  tabla_lideres()], className="tarjeta"),
        html.Div(dbc.Row([dbc.Col([html.Label("Modelo para la predicción y el diagnóstico"),
                                   dcc.Dropdown(id="m-modelo", options=MODELOS, value=ORIGINAL, clearable=False)],
                                  md=5)]), className="tarjeta"),
        html.Div([
            html.H3("Predicción en vivo"),
            dbc.Row([dbc.Col([html.Label("Cargar una vivienda de ejemplo del test (con su estrato real)"),
                              dcc.Dropdown(id="m-ejemplo", options=[
                                  {"label": f"Ejemplo {i + 1}: estrato real {int(e)}", "value": i}
                                  for i, e in enumerate(EJEMPLOS.estrato_num)], placeholder="(opcional)")], md=6)],
                    className="mb-2"),
            dbc.Row([campo(c) for c in FISICAS + D.CATEGORICAS + ["x_km", "y_km"]]),
            html.P("Las variables de vecindad (los 8 edificios más cercanos) y la distancia al centro se calculan solas a "
                   "partir de la ubicación; también se puede elegir la ubicación haciendo clic en el mapa.",
                   className="fuente"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="m-mapa-clic", figure=P.mapa_estratos(VIV.sample(2500, random_state=1))
                                  .update_layout(height=360, title="Clic para ubicar la vivienda",
                                                 showlegend=False), config={"displaylogo": False}), md=5),
                dbc.Col([dbc.Button("Predecir", id="m-predecir", color="primary", className="mb-2"),
                         html.Div(id="m-alerta"), dcc.Loading(html.Div(id="m-salida"))], md=7),
            ]),
            dcc.Store(id="m-ejemplo-fila"),
        ], className="tarjeta"),
        html.Div([html.H3("Diagnóstico en el test"), html.Div(id="m-diag-texto", className="fuente"),
                  dbc.Row([dbc.Col(dcc.Graph(id="m-confusion"), lg=6), dbc.Col(dcc.Graph(id="m-calibracion"), lg=6)]),
                  dbc.Row([dbc.Col(dcc.Graph(id="m-roc"), lg=6), dbc.Col(dcc.Graph(id="m-pr"), lg=6)]),
                  html.H4("Interpretabilidad global"), dcc.Graph(id="m-importancia"),
                  html.Div(id="m-importancia-texto", className="insight")], className="tarjeta"),
        html.Div([html.H3("Comparación estadística"),
                  dbc.RadioItems(id="m-version", options=[{"label": " versión 2 (espacio elegido en el bucle interno)",
                                                           "value": "v2"},
                                                          {"label": " versión 1 (campo geográfico)", "value": "v1"}],
                                 value=ver, inline=True, className="mb-2"),
                  dbc.Row([dbc.Col(dcc.Graph(id="m-cd"), lg=6), dbc.Col(html.Div(id="m-pruebas"), lg=6)]),
                  html.Div(id="m-veredicto")], className="tarjeta"),
        html.Div([
            html.H3("El modelo original: ensamble con pesos locales (EPL)"),
            dbc.Row([
                dbc.Col([
                    html.P(["Combina las probabilidades de los 7 modelos base con ", html.B("pesos que cambian según "
                            "la región"), ". Los pesos de cada región (ancla) se aprenden minimizando una log-pérdida "
                            "convexa —con una penalización que los acerca a los pesos globales— mediante gradiente "
                            "exponenciado, sobre predicciones fuera de muestra por bloques espaciales. Por ser "
                            "convexa y estrictamente convexa con λ > 0, el óptimo es único y la pérdida baja en cada "
                            "iteración."]),
                    html.P([f"Configuración final: espacio {NOMBRE_ESPACIO[R['campo_epl']['espacio']]}, "
                            f"J = {R['campo_epl']['params']['n_anclas']} anclas, λ = "
                            f"{R['campo_epl']['params']['lam']:.3g}. En la versión 2, el bucle interno eligió el "
                            "espacio así en los 25 pliegues: " + ", ".join(
                                f"{NOMBRE_ESPACIO[k]} {v}" for k, v in R["espacios_elegidos_v2"].items()) + "."]),
                    html.Div(PSEUDOCODIGO, className="pseudocodigo"),
                ], lg=7),
                dbc.Col([dcc.Graph(figure=P.convergencia(R["campo_epl"]["historia"])),
                         html.Div("La pérdida baja de forma monótona (el paso se reduce a la mitad si una iteración no "
                                  "mejora) y se detiene cuando la mejora relativa es menor que 10⁻⁷.", className="insight")],
                        lg=5),
            ]),
            html.H4("Diferencias con la literatura"),
            dash_table.DataTable(data=DIFERENCIAS, columns=[{"name": c, "id": c} for c in DIFERENCIAS[0]],
                                 style_cell={"fontSize": 12, "whiteSpace": "normal", "height": "auto",
                                             "textAlign": "left", "fontFamily": "Inter, Segoe UI, Arial"}),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=P.barras_f1(medias_var, {k: 0 for k in medias_var},
                                                      "Reglas estándar y ablación (F1 medio en los 25 pliegues)")),
                         html.Div("Cada barra es el EPL con un componente retirado o una regla de combinación estándar "
                                  "sobre los mismos expertos. La que más cae es «sin ajuste cruzado»: aprender los pesos "
                                  "con las predicciones de los expertos sobre sus propios datos premia al que memoriza.",
                                  className="insight")], lg=6),
                dbc.Col([dcc.Graph(figure=fig_campo),
                         html.Div(f"Peso de {experto_variable} (el experto cuyo peso más varía) en cada ancla del "
                                  "espacio de competencia del modelo final.", className="insight")], lg=6),
            ], className="mt-3"),
        ], className="tarjeta"),
    ])

    # -- callbacks ---------------------------------------------------------
    ENTRADAS = FISICAS + D.CATEGORICAS + ["x_km", "y_km"]

    @callback([Output(f"in-{c}", "value") for c in ENTRADAS] + [Output("m-ejemplo-fila", "data")],
              Input("m-ejemplo", "value"), prevent_initial_call=True)
    def cargar_ejemplo(i):
        if i is None:
            return [no_update] * (len(ENTRADAS) + 1)
        fila = EJEMPLOS.iloc[int(i)]
        valores = [fila[c] if ESQUEMA[c]["tipo"] == "categórica" else (None if pd.isna(fila[c]) else float(fila[c]))
                   for c in ENTRADAS]
        return valores + [fila.to_dict()]

    @callback(Output("in-x_km", "value", allow_duplicate=True), Output("in-y_km", "value", allow_duplicate=True),
              Input("m-mapa-clic", "clickData"), prevent_initial_call=True)
    def ubicar(click):
        if not click:
            return no_update, no_update
        p = click["points"][0]
        return round(p["x"], 2), round(p["y"], 2)

    def derivar(fila, ejemplo):
        """Distancia al centro y vecindad de los 8 vecinos más cercanos de la submuestra (o las del ejemplo)."""
        if ejemplo and abs(ejemplo["x_km"] - fila["x_km"]) < 1e-9 and abs(ejemplo["y_km"] - fila["y_km"]) < 1e-9:
            for c in DERIVADAS:
                fila[c] = ejemplo[c]
            return fila, "las del ejemplo"
        dist, idx = ARBOL.query([fila["x_km"], fila["y_km"]], k=8)
        v = VIV.iloc[idx]
        fila.update({"dist_centro_km": float(np.hypot(fila["x_km"] - CENTRO["x0"], fila["y_km"] - CENTRO["y0"])),
                     "knn8_log_area": float(np.log1p(v.area_construida).mean()),
                     "knn8_banios": float(v.total_banios.mean()), "knn8_habitaciones": float(v.total_habitaciones.mean()),
                     "knn8_antiguedad": float(v.antiguedad.mean()), "knn8_piso": float(v.planta_ubicacion.mean()),
                     "knn8_prop_PH": float((v.condicion_predio == "PH_Unidad_Predial").mean()),
                     "knn8_dist_km": float(np.mean(dist))})
        return fila, "aproximadas con la submuestra"

    @callback(Output("m-salida", "children"), Output("m-alerta", "children"), Input("m-predecir", "n_clicks"),
              State("m-modelo", "value"), State("m-ejemplo-fila", "data"),
              [State(f"in-{c}", "value") for c in ENTRADAS], prevent_initial_call=True)
    def predecir(_, modelo, ejemplo, *valores):
        fila = dict(zip(ENTRADAS, valores))
        errores = []
        for c in ENTRADAS:
            e = ESQUEMA[c]
            v = fila[c]
            if e["tipo"] == "categórica":
                if v not in e["categorias"]:
                    errores.append(f"{c}: elija una categoría de la lista.")
            elif v is None or not np.isfinite(float(v)):
                errores.append(f"{c}: falta el valor o no es un número.")
            elif not e["minimo"] <= float(v) <= e["maximo"]:
                errores.append(f"{c}: {v} está fuera del rango válido [{e['minimo']:g}, {e['maximo']:g}].")
        if errores:
            return no_update, dbc.Alert([html.B("Revise los datos: "), html.Ul([html.Li(x) for x in errores])],
                                        color="danger")
        try:
            fila, origen = derivar(fila, ejemplo)
            X = pd.DataFrame([fila])[list(ESQUEMA)]
            for c in ESQUEMA:
                if ESQUEMA[c]["tipo"] != "categórica":
                    X[c] = X[c].astype(float)
            est = D.artefacto(LIDERES.set_index("modelo").loc[modelo, "archivo"])
            from estrato3 import evaluacion as EV

            def f(df):
                return EV.proba_completa(est, df)
            proba = f(X)[0]
            k = int(np.argmax(proba))
            phi, base, fx = shapley_muestreo(f, X, FONDO, k, n_permutaciones=25, semilla=0)
        except Exception as e:  # el dashboard no se cae: muestra el error
            return no_update, dbc.Alert(f"No se pudo predecir: {type(e).__name__}: {e}", color="danger")
        real = (f" · estrato real del ejemplo: {int(ejemplo['estrato_num'])}" if ejemplo and origen == "las del ejemplo"
                else "")
        return html.Div([
            dbc.Row([dbc.Col(html.Div([html.Div(f"Estrato {k + 1}", className="kpi-valor" +
                                                (" original" if modelo == ORIGINAL else "")),
                                       html.Div(f"predicción de {modelo} (probabilidad {proba[k]:.0%}){real}",
                                                className="kpi-etiqueta")], className="kpi"), md=12)]),
            dcc.Graph(figure=P.barras_probabilidad(proba, range(1, 7))),
            dcc.Graph(figure=P.shap_local(phi, base, fx, f"Por qué: aporte de cada variable a la probabilidad del "
                                                         f"estrato {k + 1} (valores de Shapley)")),
            html.Div(f"Variables de vecindad y distancia al centro: {origen}. Barras azules suben la probabilidad del "
                     f"estrato predicho y naranjas la bajan; suman {fx - base:+.3f} (de {base:.2f} a {fx:.2f}).",
                     className="insight")]), None

    @callback(Output("m-confusion", "figure"), Output("m-calibracion", "figure"), Output("m-roc", "figure"),
              Output("m-pr", "figure"), Output("m-importancia", "figure"), Output("m-diag-texto", "children"),
              Output("m-importancia-texto", "children"), Input("m-modelo", "value"))
    def diagnostico(modelo):
        d = R["diagnosticos"][modelo]
        imp = R["importancia"][modelo]
        top = list(imp.items())[:12]
        fig_imp = P.barras_f1({k: v for k, v in top}, {k: 0 for k, _ in top},
                              f"Importancia global de {modelo}: media de |valor de Shapley|", resaltar=None)
        fig_imp.update_layout(xaxis_title="Aporte medio a la probabilidad de la clase predicha")
        f1 = R["test"]["f1_macro"][modelo]
        ic = R["test"]["ic95_bca_bloques"][modelo]
        cm = np.array(d["confusion"])
        cerca = np.trace(cm) + np.trace(cm, 1) + np.trace(cm, -1)
        return (P.matriz_confusion(cm, range(1, 7), f"Matriz de confusión de {modelo}"),
                P.calibracion(d["calibracion"], "Calibración"),
                P.curvas(d["roc"], "Curvas ROC (uno contra el resto)", "Tasa de falsos positivos",
                         "Tasa de verdaderos positivos"),
                P.curvas(d["pr"], "Curvas de precisión-recall", "Recall", "Precisión", diagonal=False), fig_imp,
                f"Test de 65 145 viviendas en 7 bloques: F1 macro {f1:.3f}, IC 95 % BCa por bloques [{ic[0]:.3f}, "
                f"{ic[1]:.3f}]. El {100 * cerca / cm.sum():.0f} % de las predicciones cae en el estrato real o en uno "
                "vecino.",
                [html.B(f"Lo que más pesa: {top[0][0]}, {top[1][0]} y {top[2][0]}."),
                 " Mismo estimador para todos los modelos (valores de Shapley por permutaciones sobre 80 viviendas del "
                 "test), así que las importancias son comparables entre modelos."])

    @callback(Output("m-cd", "figure"), Output("m-pruebas", "children"), Output("m-veredicto", "children"),
              Input("m-version", "value"))
    def comparacion(version):
        fr = R["estadistica"][version]["friedman"]
        fig = P.diagrama_cd(fr["rangos"], fr["cd"], f"Diferencia crítica de Nemenyi, versión {version} "
                                                    f"(25 pliegues, Friedman p = {fr['p']:.1e})")
        return fig, tabla_pruebas(version), veredicto_alerta(version)
