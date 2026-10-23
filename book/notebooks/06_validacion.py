# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 6. Validación estadística
#
# :::{admonition} Qué responde este capítulo
# :class: tip
# ¿El modelo original supera a los benchmarks de forma estadísticamente significativa y con un efecto no
# despreciable (condición R3)? ¿Se confirma en el test, en otros conjuntos de datos y bajo perturbaciones?
# :::
#
# ## 6.1 Protocolo
#
# - **Pareado estricto.** Los 11 modelos (7 de la guía, el XGBoost geográfico, LightGBM, CatBoost y el EPL) se evalúan
#   en las **mismas 25 particiones externas** (5 repeticiones × 5 pliegues kNNDM), con la misma semilla (42) y el mismo
#   preprocesamiento; cada uno con su búsqueda de 12 evaluaciones en los mismos 3 pliegues internos.
# - **Potencia.** Validación externa repetida (la guía la recomienda: con 5 pliegues el p mínimo de Wilcoxon es 0.0625)
#   y corrección de Nadeau y Bengio para la dependencia entre pliegues {cite}`nadeau2003,bouckaert2004`.
# - **Comparaciones múltiples.** Holm {cite}`holm1979` sobre las 10 comparaciones del EPL con cada rival; se reportan p
#   crudo y ajustado.
# - **Datos de corte transversal.** Se usan las pruebas de la columna izquierda de la Tabla 3 de la guía (no las de
#   series de tiempo).
# - **Superioridad (sección 6.1 de la guía).** (1) media mayor que la de cada benchmark; (2) al menos tres pruebas
#   distintas rechazan la igualdad frente al mejor benchmark con p ajustado < 0.05; (3) |δ de Cliff| ≥ 0.147.
# - **Versiones.** El modelo final es la versión 2 (espacio de competencia elegido en el bucle interno), fijado antes
#   de ver sus resultados externos; la versión 1 se reporta igual.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio

RAIZ = Path.cwd().resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import plots as P  # noqa: E402

pio.renderers.default = "notebook"
E = {v: json.loads((RAIZ / "resultados" / "cv5x5" / f"estadistica_{v}.json").read_text(encoding="utf-8"))
     for v in ("v1", "v2")}
pd.DataFrame(E["v2"]["resumen"]).round(4).style.hide(axis="index")

# %% [markdown]
# ## 6.2 Prueba global: Friedman y Nemenyi

# %%
fr = E["v2"]["friedman"]
print(f"Friedman (versión 2): χ² = {fr['chi2']:.1f}, gl = {fr['gl']}, p = {fr['p']:.1e}; "
      f"diferencia crítica de Nemenyi = {fr['cd']:.2f} rangos con 11 modelos y 25 pliegues")
P.diagrama_cd(fr["rangos"], fr["cd"], "Figura 6.1. Diferencia crítica de Nemenyi (versión 2, 25 pliegues)").show()

# %% [markdown]
# **Lectura.** Los modelos no rinden igual (Friedman p ≈ 10⁻²⁴) y el EPL tiene el mejor rango medio, pero queda a menos
# de una diferencia crítica de CatBoost, Random Forest, XGBoost y LightGBM: Nemenyi no los separa. Sí separa al EPL de
# los modelos débiles (Naive Bayes, árbol, KNN y SVM).
#
# ## 6.3 El EPL frente a cada benchmark

# %%
def tabla_pruebas(v):
    filas = []
    for x in E[v]["epl_vs_benchmarks"]:
        filas.append({"rival": x["rival"], "EPL − rival": round(x["dif_media"], 4),
                      "IC 95 % BCa": f"[{x['IC95_BCa'][0]:+.4f}, {x['IC95_BCa'][1]:+.4f}]",
                      "pliegues a favor": f"{x['pliegues_a_favor']}/25",
                      "Wilcoxon p (Holm)": f"{x['wilcoxon_p']:.4f} ({x['wilcoxon_p_holm']:.4f})",
                      "t N-B p (Holm)": f"{x['t_NB_p']:.4f} ({x['t_NB_p_holm']:.4f})",
                      "Nemenyi p": round(x["nemenyi_p"], 4), "δ de Cliff": f"{x['delta_cliff']:+.3f} ({x['delta_magnitud']})",
                      "d pareada": round(x["d_cohen_pareada"], 2),
                      "P(EPL mejor) bayes": round(x["bayes"]["P(EPL mejor)"], 2)})
    return pd.DataFrame(filas)


tabla_pruebas("v2").style.hide(axis="index")

# %% [markdown]
# **Tabla de pruebas frente al mejor benchmark** (el de mayor F1 medio en la validación: CatBoost).

# %%
v = E["v2"]
mejor = v["veredicto_validacion"]["mejor_benchmark"]
x = next(c for c in v["epl_vs_benchmarks"] if c["rival"] == mejor)
pd.DataFrame([
    {"prueba": "Friedman → Nemenyi", "H0": "igual rango medio de los 11 modelos",
     "supuestos verificados": "bloques pareados (mismos pliegues); rangos sin supuesto de normalidad",
     "estadístico": f"χ² = {v['friedman']['chi2']:.1f}", "p crudo": f"{v['friedman']['p']:.1e}",
     "p ajustado": f"Nemenyi {x['nemenyi_p']:.3f}", "efecto": f"rango EPL {v['friedman']['rangos']['EPL (original)']:.2f} "
     f"vs {v['friedman']['rangos'][mejor]:.2f} (CD {v['friedman']['cd']:.2f})",
     "conclusión": "rechaza igualdad global; no separa EPL y " + mejor},
    {"prueba": "Wilcoxon de rangos con signo", "H0": "mediana de la diferencia por pliegue = 0",
     "supuestos verificados": f"diferencias pareadas; simetría aprox. (asimetría {x['supuestos']['asimetria']:.2f}); "
                              f"{x['supuestos']['ceros']} ceros",
     "estadístico": f"W = {x['wilcoxon_W']:.0f}", "p crudo": f"{x['wilcoxon_p']:.4f}",
     "p ajustado": f"Holm {x['wilcoxon_p_holm']:.4f}", "efecto": f"δ de Cliff {x['delta_cliff']:+.3f}",
     "conclusión": "rechaza" if x["wilcoxon_p_holm"] < 0.05 else "no rechaza"},
    {"prueba": "t corregida de Nadeau y Bengio", "H0": "media de la diferencia = 0",
     "supuestos verificados": f"normalidad de las diferencias (Shapiro p = {x['supuestos']['shapiro_p']:.2f}); "
                              "corrección por solapamiento de entrenamientos",
     "estadístico": f"t = {x['t_NB']:.2f}, gl = {x['t_NB_gl']}", "p crudo": f"{x['t_NB_p']:.4f}",
     "p ajustado": f"Holm {x['t_NB_p_holm']:.4f}", "efecto": f"d pareada {x['d_cohen_pareada']:.2f}",
     "conclusión": "rechaza" if x["t_NB_p_holm"] < 0.05 else "no rechaza"},
    {"prueba": "Bayesiana de rangos con signo (ROPE ±0.01)", "H0": "—",
     "supuestos verificados": "previa de Dirichlet con pseudo-observación en 0", "estadístico": "—", "p crudo": "—",
     "p ajustado": "—", "efecto": f"P(EPL mejor) {x['bayes']['P(EPL mejor)']:.2f}; P(equivalentes) "
     f"{x['bayes']['P(equivalentes)']:.2f}", "conclusión": "probable ventaja, no concluyente"},
]).style.hide(axis="index")

# %% [markdown]
# **Lectura.** El EPL tiene la **mayor media** de los 11 modelos (condición 1 cumplida) y supera a cada benchmark en la
# mayoría de los pliegues. Frente al mejor benchmark, la diferencia es de +0.013 de F1: Wilcoxon con Holm la declara
# significativa, pero la t corregida de Nadeau-Bengio y Nemenyi no, y el δ de Cliff (≈ 0.10) es **despreciable**.
# Solo 1 de 3 pruebas rechaza: **la condición R3 no se cumple en la validación.** Frente a los modelos débiles (KNN,
# árbol, SVM, Naive Bayes) la superioridad sí es clara en todas las pruebas.
#
# La versión 1 da la misma conclusión:

# %%
for ver in ("v1", "v2"):
    ve = E[ver]["veredicto_validacion"]
    print(f"{ver}: mejor benchmark {ve['mejor_benchmark']}; pruebas que rechazan {ve['cond2_n_rechazos']}/3 "
          f"{ve['cond2_pruebas_que_rechazan_vs_mejor']}; δ de Cliff {ve['cond3_delta_cliff_vs_mejor']:+.3f}; "
          f"R3 en validación: {ve['R3_en_validacion']}")

# %% [markdown]
# ### 6.3.1 Frente al modelo base del Entregable 1
#
# El modelo base del proyecto (logística multinomial B del E1: C = 0.01, sin pesos, **sin vecindad física**) no es un
# benchmark de R3, pero es el punto de partida. Se reprodujo en las mismas 25 particiones (control: la repetición 0
# da exactamente los F1 que publicó el E3) y en el test (``experimentos/04b_modelo_base.py``).

# %%
b = json.loads((RAIZ / "resultados" / "cv5x5" / "base" / "base_vs_epl.json").read_text(encoding="utf-8"))
x, fb, te = b["epl_vs_base"], b["friedman_con_base"], b["test"]
pd.DataFrame([
    {"prueba": "Friedman → Nemenyi (12 modelos)", "estadístico": f"χ² = {fb['chi2']:.1f}",
     "p": f"Nemenyi {fb['nemenyi_p_epl_vs_base']:.1e}",
     "efecto": f"rango {fb['rango_epl']:.2f} frente a {fb['rango_base']:.2f} (CD {fb['cd']:.2f})"},
    {"prueba": "Wilcoxon de rangos con signo", "estadístico": f"W = {x['wilcoxon_W']:.0f}", "p": f"{x['wilcoxon_p']:.1e}",
     "efecto": f"δ de Cliff {x['delta_cliff']:+.2f} ({x['delta_magnitud']})"},
    {"prueba": "t corregida de Nadeau-Bengio", "estadístico": f"t = {x['t_NB']:.2f}", "p": f"{x['t_NB_p']:.1e}",
     "efecto": f"d pareada {x['d_cohen_pareada']:.2f}"},
    {"prueba": "McNemar en el test", "estadístico": f"χ² = {te['mcnemar']['estadistico']:.0f}",
     "p": f"{te['mcnemar']['p']:.1e}",
     "efecto": f"EPL acierta y el base no: {te['mcnemar']['b']:,}; al revés: {te['mcnemar']['c']:,}"},
]).style.hide(axis="index")

# %% [markdown]
# **Lectura.** Frente al modelo base, el EPL es **claramente superior**: +0.19 de F1 en la validación (IC BCa
# [+0.16, +0.22], 25 de 25 pliegues) y +0.16 en el test (IC por bloques [+0.09, +0.22]), con las cuatro pruebas
# rechazando la igualdad y un efecto grande (δ ≈ 0.84). Pero esa mejora la comparten los benchmarks flexibles con las
# mismas variables de vecindad (CatBoost y Random Forest llegan a ≈ 0.605): viene sobre todo de la **vecindad física** y
# de los modelos no lineales, no de la regla de combinación. Por eso la pregunta de R3 es la de la sección 6.3.
#
# ## 6.4 Reglas estándar, ablación y versión 1 frente a versión 2

# %%
filas = []
for x in E["v2"]["epl_vs_variantes"]:
    filas.append({"variante": x["rival"], "EPL v2 − variante": round(x["dif_media"], 4),
                  "IC 95 % BCa": f"[{x['IC95_BCa'][0]:+.4f}, {x['IC95_BCa'][1]:+.4f}]",
                  "pliegues a favor": f"{x['pliegues_a_favor']}/25", "Wilcoxon p (Holm)": round(x["wilcoxon_p_holm"], 3)})
pd.DataFrame(filas).style.hide(axis="index")

# %% [markdown]
# **Lectura.** (i) El EPL **no supera a las reglas estándar** de combinación: frente al promedio simple, los pesos por
# error global y el stacking lineal las diferencias están dentro del ruido. (ii) El único componente con un aporte
# claro es el **ajuste cruzado por bloques**: sin él el F1 cae unos 0.054 (23 de 25 pliegues). (iii) El campo local,
# la penalización KL y el suavizado no aportan de forma medible. (iv) La versión 2 (0.618) empata con la versión 1
# (0.618); el bucle interno eligió el espacio de las variables en 18 de 25 pliegues. Fijar ese espacio da 0.626, pero
# esa variante se calculó como análisis de sensibilidad y **no se puede elegir mirando el bucle externo** sin sesgo.
#
# Esto coincide con {cite:t}`zhou2026`: la ganancia de una combinación por regiones depende del desplazamiento entre
# entrenamiento y destino; en «zonas nuevas», sin datos etiquetados cerca, los pesos aprendidos en una región no se
# trasladan a otra.
#
# ## 6.5 Test confirmatorio (65 145 viviendas, 7 bloques)

# %%
tc = json.loads((RAIZ / "resultados" / "test" / "test_confirmatorio.json").read_text(encoding="utf-8"))
filas = [{"modelo": n, "F1 macro": round(f, 4), "IC 95 % BCa por bloques":
          f"[{tc['ic95_bca_bloques'][n][0]:.3f}, {tc['ic95_bca_bloques'][n][1]:.3f}]",
          "exactitud": round(tc["metricas"][n]["accuracy"], 4), "MAE ordinal": round(tc["metricas"][n]["mae_ordinal"], 3),
          "McNemar: EPL acierta y el rival no / al revés": "—" if n == "EPL (original)" else
          f"{tc['mcnemar'][n]['b']:,} / {tc['mcnemar'][n]['c']:,} (p Holm {tc['mcnemar'][n]['p_holm']:.1e})"}
         for n, f in sorted(tc["f1_macro"].items(), key=lambda t: -t[1])]
pd.DataFrame(filas).style.hide(axis="index")

# %%
d = tc["epl_menos_mejor"]
print(f"EPL − {tc['mejor_rival_en_validacion']} en el test: {d['dif_f1']:+.4f}, IC 95 % BCa por bloques "
      f"[{d['ic95_bca_bloques'][0]:+.4f}, {d['ic95_bca_bloques'][1]:+.4f}]; permutación por signo en 6 bloques: "
      f"p = {d['p_permutacion_signo_bloques']:.3f}")
print("DeLong (AUC uno contra el resto, EPL − " + tc["mejor_rival_en_validacion"] + "): " + ", ".join(
    f"E{k} {x['diferencia']:+.4f} (p Holm {x['p_holm']:.1e})" for k, x in tc["delong_vs_mejor"].items()))

# %% [markdown]
# **Lectura.** En el test el EPL empata con CatBoost en F1 macro (diferencia ≈ −0.002, IC por bloques que contiene el
# 0, permutación p ≈ 0.84). **McNemar es significativo pero en contra del EPL**: CatBoost acierta más viviendas que el
# EPL falla que al revés (su exactitud es ≈ 0.01 mayor); lo mismo ocurre frente al Random Forest, XGBoost y la
# logística. El EPL cambia un poco de exactitud por un F1 macro más parejo entre estratos. DeLong da AUC algo mayores
# para el EPL en los estratos 1 a 4 y menores en 5 y 6. Con 65 145 viviendas y dependencia espacial fuerte (ICC 0.99)
# McNemar y DeLong detectan diferencias minúsculas: sus p-valores son optimistas, y por eso pesan más el IC y la
# permutación por bloques. **El test no confirma superioridad.**
#
# ## 6.6 Tres conjuntos de datos adicionales

# %%
ext = json.loads((RAIZ / "resultados" / "externos" / "resumen_externos_v2.json").read_text(encoding="utf-8"))
filas = []
for nombre, r in ext.items():
    medias = {k: v for k, v in r["media"].items() if not k.startswith("[")}
    rivales = [c for c in r["epl_v2_vs_rivales"]]
    mejor = max(rivales, key=lambda c: c["media_rival"])
    rango = sorted(medias, key=medias.get, reverse=True).index("EPL (original)") + 1
    filas.append({"conjunto": nombre, "F1 EPL v2": round(medias["EPL (original)"], 3), "puesto del EPL": f"{rango} de 10",
                  "mejor rival": f"{mejor['rival']} ({mejor['media_rival']:.3f})",
                  "diferencia": round(mejor["dif_media"], 4), "Wilcoxon p (Holm)": round(mejor["wilcoxon_p_holm"], 3),
                  "t N-B p (Holm)": round(mejor["t_NB_p_holm"], 3),
                  "Nemenyi p": round(r["nemenyi_p_epl"][mejor["rival"]], 3), "δ de Cliff": round(mejor["delta_cliff"], 3),
                  "espacio elegido": r["espacio_elegido"]})
pd.DataFrame(filas).style.hide(axis="index")

# %% [markdown]
# **Lectura.** Mismo protocolo en conjuntos de naturaleza distinta (King County, California y vino tinto; 3 × 5
# pliegues, mismos 9 rivales y presupuesto). El EPL **empata** con el mejor en King County, **gana** en California
# (+0.025 frente a la logística, δ grande, pero solo Wilcoxon rechaza) y **pierde** en el vino frente al Random Forest.
# El resultado es coherente con la 5 × 5: el EPL está siempre entre los mejores, pero no domina.
#
# ## 6.7 Robustez y sensibilidad

# %%
rob = RAIZ / "resultados" / "cv5x5" / "robustez"
if (rob / "resumen.json").exists():
    r = json.loads((rob / "resumen.json").read_text(encoding="utf-8"))
    display(pd.DataFrame(r["condiciones"]).T.round(3))
    print("F1 medio del EPL por semilla (5 pliegues de la repetición 0):",
          {k: round(v, 4) for k, v in r["semillas_EPL"].items()}, "| desviación entre semillas:",
          round(r["sd_entre_semillas"], 4))
    s = json.loads((rob / "sensibilidad.json").read_text(encoding="utf-8"))
    display(pd.DataFrame({k: v["media"] for k, v in s.items()}, index=["F1 medio (25 pliegues)"]).T.round(4))
else:
    print("Pendiente: experimentos/11_robustez.py")

# %% [markdown]
# **Lectura.** Las condiciones alteran solo el entrenamiento (la validación no se toca) y usan los hiperparámetros de
# cada pliegue sin nueva búsqueda (5 pliegues de la repetición 0). La sensibilidad cambia un hiperparámetro a la vez
# desde la configuración elegida (25 pliegues).
#
# - **Ruido, menos datos y desbalance.** El EPL queda por encima del Random Forest y de CatBoost en las cuatro
#   condiciones (por ejemplo, con 25 % del entrenamiento: 0.648 frente a 0.628 y 0.610; con 20 % de etiquetas
#   cambiadas: 0.652 frente a 0.641 y 0.630): combinar expertos amortigua el daño. Pero la regla simple de pesos por
#   error global lo iguala o lo supera (0.668 frente a 0.658 con más desbalance): la ventaja es del ensamble, no del
#   campo local.
# - **Semillas.** F1 medio 0.680, 0.680 y 0.677 con tres semillas (desviación ≈ 0.002): el resultado no depende de la
#   semilla del ajuste cruzado ni de K-medias.
# - **Hiperparámetros propios.** El F1 se mueve entre 0.615 y 0.624 al variar J (1 a 80), h (0.25 a 4 km), k (1 a 10),
#   λ (0 a 10) y el espacio de competencia: el modelo es estable, y la misma estabilidad confirma que el campo local
#   aporta poco (J = 1, pesos globales, da 0.616).
#
# ## 6.8 Interpretabilidad: EPL frente al mejor benchmark

# %%
mod = json.loads((RAIZ / "data" / "dashboard" / "modelos.json").read_text(encoding="utf-8"))
imp = pd.DataFrame({m: pd.Series(mod["importancia"][m]) for m in ("EPL (original)", "CatBoost")}).fillna(0)
imp.sort_values("EPL (original)", ascending=False).head(12).round(4)

# %% [markdown]
# Mismo estimador para los dos modelos (valores de Shapley por permutaciones {cite}`strumbelj2014`, media de |φ| de la
# probabilidad de la clase predicha en 80 viviendas del test). **Lectura.** Los dos modelos se apoyan en la ubicación
# (distancia al centro, posición sur–norte) y en la vecindad física; el EPL, al promediar expertos, reparte el peso
# entre más variables.
#
# ## 6.9 Veredicto
#
# :::{admonition} R3 no se cumple
# :class: warning
# El EPL tiene el mayor F1 medio en la validación repetida y gana en la mayoría de los pliegues a cada benchmark, pero
# frente al mejor (CatBoost) solo **una** de las tres pruebas rechaza la igualdad con p ajustado < 0.05 y el tamaño
# del efecto es **despreciable**; el test y los conjuntos adicionales no confirman una ventaja. Por la regla de la
# sección 12.1 de la guía, con R1 y R2 cumplidos y R3 no, el tope de la calificación es **4.0**.
# :::
#
# Causas probables: (i) los expertos ya están cerca del techo que permiten los datos abiertos (los cinco mejores
# modelos quedan a 0.02 de F1), porque el DANE asigna el estrato con información que no es pública; (ii) en «zonas
# nuevas» la competencia local no se traslada de una región a otra; (iii) con 25 pliegues dependientes, una diferencia
# de 0.013 no alcanza la potencia de la t corregida.
