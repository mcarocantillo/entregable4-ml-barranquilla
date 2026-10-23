# -*- coding: utf-8 -*-
"""Funciones que construyen las figuras del dashboard (una plantilla Plotly para todas).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Paleta de cinco colores (sección 7.4 de la guía): azul oscuro, azul, dorado, naranja (solo el modelo original) y
gris. Los estratos usan una rampa de un solo tono (azul claro = 1, azul oscuro = 6), que también se distingue sin
color porque cada gráfico rotula el estrato en el eje o en la leyenda.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

AZUL_OSCURO, AZUL, DORADO, NARANJA, GRIS = "#0d366b", "#2a78d6", "#f5b83d", "#eb6834", "#6b7280"
ESTRATOS = ["#c6dbf5", "#93bdee", "#5f9de4", "#2a78d6", "#1a56a8", "#0d366b"]
ORIGINAL = "EPL (original)"

pio.templates["proyecto"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color="#101418"),
    # el título va arriba y la leyenda horizontal justo debajo (antes se montaban)
    title=dict(font=dict(size=15, color=AZUL_OSCURO), x=0.0, xanchor="left", y=0.98, yanchor="top"),
    colorway=[AZUL, DORADO, AZUL_OSCURO, GRIS, NARANJA], paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor="#eef0f3", zerolinecolor="#d5d9df", linecolor="#d5d9df", ticks="outside"),
    yaxis=dict(gridcolor="#eef0f3", zerolinecolor="#d5d9df", linecolor="#d5d9df", ticks="outside"),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    margin=dict(l=60, r=20, t=95, b=50), hoverlabel=dict(font_size=12)))
pio.templates.default = "proyecto"


def _vacia(texto="Sin datos para los filtros elegidos"):
    fig = go.Figure()
    fig.add_annotation(text=texto, showarrow=False, font=dict(size=14, color=GRIS), x=0.5, y=0.5, xref="paper",
                       yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def distribucion_objetivo(df):
    """Barras: viviendas por estrato, con el porcentaje en la etiqueta."""
    if df.empty:
        return _vacia()
    n = df.estrato_num.value_counts().reindex(range(1, 7), fill_value=0)
    pct = 100 * n / n.sum()
    fig = go.Figure(go.Bar(x=[f"Estrato {k}" for k in n.index], y=n.values, marker_color=ESTRATOS,
                           text=[f"{p:.1f} %" for p in pct], textposition="outside",
                           hovertemplate="%{x}: %{y:,} viviendas (%{text})<extra></extra>"))
    fig.update_layout(title="Viviendas por estrato (variable objetivo)", yaxis_title="Viviendas", height=360,
                      showlegend=False)
    return fig


def histograma(df, var, unidad, log=False):
    """Histograma superpuesto por estrato (densidad), en escala log si se pide."""
    if df.empty:
        return _vacia()
    fig = go.Figure()
    x_todo = df[var].dropna()
    if log:
        x_todo = np.log1p(x_todo)
    bordes = np.histogram_bin_edges(x_todo, bins=40)
    for k in range(1, 7):
        x = df.loc[df.estrato_num == k, var].dropna()
        if x.empty:
            continue
        x = np.log1p(x) if log else x
        fig.add_trace(go.Histogram(x=x, name=f"Estrato {k}", marker_color=ESTRATOS[k - 1], opacity=0.75,
                                   histnorm="probability density", xbins=dict(start=bordes[0], end=bordes[-1],
                                                                             size=bordes[1] - bordes[0])))
    eje = f"log(1 + {var})" if log else f"{var} ({unidad})"
    fig.update_layout(barmode="overlay", title=f"Distribución de {var} por estrato", xaxis_title=eje,
                      yaxis_title="Densidad", height=380)
    return fig


def mapa_correlacion(df, numericas):
    """Mapa de calor de correlaciones de Spearman (robustas a colas y relaciones monótonas)."""
    if len(df) < 10:
        return _vacia()
    c = df[numericas + ["estrato_num"]].corr(method="spearman")
    etiquetas = [v.replace("_", " ") for v in c.columns]
    fig = go.Figure(go.Heatmap(z=c.values, x=etiquetas, y=etiquetas, zmin=-1, zmax=1,
                               colorscale=[[0, NARANJA], [0.5, "#ffffff"], [1, AZUL_OSCURO]],
                               text=np.round(c.values, 2), texttemplate="%{text}", textfont=dict(size=9),
                               colorbar=dict(title="ρ"), hovertemplate="%{y} – %{x}: ρ = %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Correlación de Spearman entre variables numéricas y el estrato", height=560,
                      xaxis=dict(tickangle=-45), yaxis=dict(autorange="reversed"), margin=dict(l=140, b=140))
    return fig


def faltantes(df, variables):
    """Porcentaje de valores faltantes por variable (solo las que tienen alguno)."""
    pct = 100 * df[variables].isna().mean()
    pct = pct[pct > 0].sort_values()
    if pct.empty:
        return _vacia("Ninguna variable tiene faltantes con estos filtros")
    fig = go.Figure(go.Bar(x=pct.values, y=pct.index, orientation="h", marker_color=AZUL, cliponaxis=False,
                           text=[f"{p:.2f} %" for p in pct.values], textposition="outside",
                           hovertemplate="%{y}: %{x:.2f} % faltante<extra></extra>"))
    fig.update_layout(title="Valores faltantes por variable", xaxis_title="Viviendas sin dato (%)", height=300,
                      margin=dict(l=150))
    return fig


def caja_atipicos(df, var, unidad, log=False):
    """Diagrama de caja por estrato (puntos fuera de 1.5·RIC = atípicos de Tukey)."""
    if df.empty:
        return _vacia()
    fig = go.Figure()
    for k in range(1, 7):
        x = df.loc[df.estrato_num == k, var].dropna()
        if x.empty:
            continue
        fig.add_trace(go.Box(y=np.log1p(x) if log else x, name=f"E{k}", marker_color=ESTRATOS[k - 1],
                             boxpoints="outliers", marker=dict(size=3)))
    fig.update_layout(title=f"Atípicos de {var} por estrato (regla de Tukey)", showlegend=False, height=380,
                      yaxis_title=f"log(1 + {var})" if log else f"{var} ({unidad})", xaxis_title="Estrato")
    return fig


def mapa_estratos(df):
    """Viviendas en el plano (km respecto al centro), coloreadas por estrato."""
    if df.empty:
        return _vacia()
    fig = go.Figure()
    for k in range(1, 7):
        s = df[df.estrato_num == k]
        fig.add_trace(go.Scattergl(x=s.x_km, y=s.y_km, mode="markers", name=f"Estrato {k}",
                                   marker=dict(size=4, color=ESTRATOS[k - 1], line=dict(width=0)),
                                   hovertemplate=f"Estrato {k}<br>x %{{x}} km, y %{{y}} km<extra></extra>"))
    fig.update_layout(title="Dónde está cada estrato (coordenadas redondeadas a 100 m)", xaxis_title="Este-oeste (km)",
                      yaxis_title="Sur-norte (km)", height=480, yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def dispersion(df, x, y, color, etiquetas, max_puntos=4000, semilla=0):
    """Explorador bivariado: x, y y color elegidos por el usuario (muestra fija si hay muchas viviendas)."""
    if df.empty:
        return _vacia()
    d = df.sample(max_puntos, random_state=semilla) if len(df) > max_puntos else df
    fig = go.Figure()
    if color == "estrato_num":
        for k in range(1, 7):
            s = d[d.estrato_num == k]
            fig.add_trace(go.Scattergl(x=s[x], y=s[y], mode="markers", name=f"Estrato {k}",
                                       marker=dict(size=5, color=ESTRATOS[k - 1], opacity=0.75)))
    elif d[color].dtype == object or str(d[color].dtype) in ("category", "str", "string"):
        paleta = [AZUL, DORADO, AZUL_OSCURO, GRIS, NARANJA]
        for i, (cat, s) in enumerate(d.groupby(color)):
            fig.add_trace(go.Scattergl(x=s[x], y=s[y], mode="markers", name=str(cat)[:28],
                                       marker=dict(size=5, color=paleta[i % 5], opacity=0.7)))
    else:
        fig.add_trace(go.Scattergl(x=d[x], y=d[y], mode="markers", name=color,
                                   marker=dict(size=5, color=d[color], colorscale=[[0, "#c6dbf5"], [1, AZUL_OSCURO]],
                                               colorbar=dict(title=color[:14]), opacity=0.8)))
    fig.update_layout(title=f"{etiquetas.get(y, y)} frente a {etiquetas.get(x, x)}", height=480,
                      xaxis_title=etiquetas.get(x, x), yaxis_title=etiquetas.get(y, y))
    return fig


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def barras_probabilidad(proba, clases):
    """Probabilidad que el modelo asigna a cada estrato para la vivienda del formulario."""
    fig = go.Figure(go.Bar(x=[f"Estrato {k}" for k in clases], y=proba, marker_color=ESTRATOS[:len(clases)],
                           text=[f"{100 * p:.1f} %" for p in proba], textposition="outside",
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_layout(title="Probabilidad por estrato", yaxis=dict(title="Probabilidad", tickformat=".0%",
                                                                  range=[0, 1.1]), height=320, showlegend=False)
    return fig


def shap_local(contribuciones, base, prediccion, titulo):
    """Cascada horizontal de contribuciones (valores de Shapley) a la probabilidad de la clase predicha."""
    orden = sorted(contribuciones.items(), key=lambda t: abs(t[1]))[-10:]
    nombres = [n for n, _ in orden]
    valores = [v for _, v in orden]
    fig = go.Figure(go.Bar(x=valores, y=nombres, orientation="h",
                           marker_color=[AZUL if v >= 0 else NARANJA for v in valores],
                           text=[f"{v:+.3f}" for v in valores], textposition="outside",
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_layout(title=titulo, xaxis_title=f"Aporte a la probabilidad (base {base:.2f} → {prediccion:.2f})",
                      height=380, margin=dict(l=170), showlegend=False)
    return fig


def matriz_confusion(cm, clases, titulo):
    """Matriz de confusión normalizada por fila (recall de cada estrato en la diagonal)."""
    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    etiquetas = [f"E{k}" for k in clases]
    fig = go.Figure(go.Heatmap(z=norm, x=etiquetas, y=etiquetas, zmin=0, zmax=1,
                               colorscale=[[0, "#ffffff"], [1, AZUL_OSCURO]], text=cm.astype(int),
                               texttemplate="%{z:.2f}<br>(%{text})", textfont=dict(size=10),
                               colorbar=dict(title="Proporción"),
                               hovertemplate="Real %{y}, predicho %{x}: %{z:.1%}<extra></extra>"))
    fig.update_layout(title=titulo, xaxis_title="Estrato predicho", yaxis_title="Estrato real",
                      yaxis=dict(autorange="reversed"), height=420)
    return fig


def curvas(curvas_por_clase, titulo, eje_x, eje_y, diagonal=True):
    """Curvas uno contra el resto (ROC o precisión-recall) de los 6 estratos."""
    fig = go.Figure()
    for k, c in curvas_por_clase.items():
        fig.add_trace(go.Scatter(x=c["x"], y=c["y"], mode="lines", name=f"E{k} ({c['area']:.2f})",
                                 line=dict(color=ESTRATOS[int(k) - 1], width=2)))
    if diagonal:
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                                 line=dict(color=GRIS, dash="dot", width=1)))
    # leyenda debajo: con 6 estratos + azar, arriba se montaba sobre el título
    fig.update_layout(title=titulo, xaxis_title=eje_x, yaxis_title=eje_y, height=460,
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
                      legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
                      margin=dict(l=60, r=20, t=50, b=110))
    return fig


def calibracion(puntos, titulo):
    """Diagrama de calibración (confianza de la clase predicha frente a su acierto observado)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                             line=dict(color=GRIS, dash="dot")))
    fig.add_trace(go.Scatter(x=puntos["confianza"], y=puntos["acierto"], mode="lines+markers", name="Modelo",
                             marker=dict(size=[max(5, min(18, n / 400)) for n in puntos["n"]], color=AZUL),
                             line=dict(color=AZUL), customdata=puntos["n"],
                             hovertemplate="Confianza %{x:.2f}: acierto %{y:.2f} (%{customdata:,} viv.)<extra></extra>"))
    fig.update_layout(title=titulo, xaxis_title="Probabilidad de la clase predicha",
                      yaxis_title="Proporción de aciertos", height=420, xaxis=dict(range=[0, 1]),
                      yaxis=dict(range=[0, 1]))
    return fig


def diagrama_cd(rangos, cd, titulo="Diagrama de diferencia crítica (Nemenyi)"):
    """Diagrama de diferencia crítica: rango medio de cada modelo; la barra dorada mide CD desde el mejor."""
    r = sorted(rangos.items(), key=lambda t: t[1])
    nombres = [n for n, _ in r]
    valores = [v for _, v in r]
    colores = [NARANJA if n == ORIGINAL else AZUL for n in nombres]
    fig = go.Figure(go.Scatter(x=valores, y=nombres, mode="markers+text", marker=dict(size=12, color=colores),
                               text=[f"{v:.2f}" for v in valores], textposition="middle right",
                               hovertemplate="%{y}: rango medio %{x:.2f}<extra></extra>"))
    fig.add_shape(type="rect", x0=valores[0], x1=valores[0] + cd, y0=-0.5, y1=len(nombres) - 0.5,
                  fillcolor="rgba(245,184,61,.18)", line=dict(width=0), layer="below")
    fig.add_annotation(x=valores[0] + cd, y=len(nombres) - 0.5, text=f"CD = {cd:.2f}", showarrow=False,
                       yanchor="bottom", font=dict(color="#8a6510"))
    fig.update_layout(title=titulo, xaxis_title="Rango medio en los pliegues (1 = mejor)",
                      yaxis=dict(autorange="reversed"), height=60 + 38 * len(nombres), margin=dict(l=170),
                      showlegend=False)
    return fig


def convergencia(historia, titulo="Convergencia del gradiente exponenciado"):
    """Pérdida del EPL por iteración (monótona no creciente por construcción)."""
    fig = go.Figure(go.Scatter(x=list(range(len(historia))), y=historia, mode="lines+markers",
                               line=dict(color=NARANJA), marker=dict(size=6),
                               hovertemplate="Iteración %{x}: pérdida %{y:.5f}<extra></extra>"))
    fig.update_layout(title=titulo, xaxis_title="Iteración", yaxis_title="Log-pérdida ponderada + KL", height=340)
    return fig


def mapa_pesos(anclas, pesos, nombres, experto):
    """Peso de un experto en cada ancla del EPL (campo de pesos en el plano de la ciudad)."""
    j = nombres.index(experto)
    fig = go.Figure(go.Scatter(x=anclas[:, 0], y=anclas[:, 1], mode="markers",
                               marker=dict(size=16, color=pesos[:, j], cmin=0, cmax=1,
                                           colorscale=[[0, "#ffffff"], [1, NARANJA]], line=dict(width=1, color=GRIS),
                                           colorbar=dict(title="Peso")),
                               hovertemplate="x %{x:.1f} km, y %{y:.1f} km: peso %{marker.color:.2f}<extra></extra>"))
    fig.update_layout(title=f"Peso de {experto} en cada ancla", xaxis_title="Este-oeste (km)",
                      yaxis_title="Sur-norte (km)", height=440, yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def barras_f1(medias, sds, titulo, resaltar=ORIGINAL):
    """F1 macro medio ± desviación de cada modelo (o variante)."""
    orden = sorted(medias, key=medias.get)
    fig = go.Figure(go.Bar(x=[medias[m] for m in orden], y=orden, orientation="h",
                           error_x=dict(type="data", array=[sds[m] for m in orden], color=GRIS, thickness=1),
                           marker_color=[NARANJA if m == resaltar else AZUL for m in orden],
                           text=[f"{medias[m]:.3f}" for m in orden], textposition="inside",
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title=titulo, xaxis_title="F1 macro (media ± desviación en los pliegues)",
                      height=60 + 34 * len(orden), margin=dict(l=190), showlegend=False)
    return fig
