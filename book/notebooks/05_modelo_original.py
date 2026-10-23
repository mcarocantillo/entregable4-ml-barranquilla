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
# # 5. El modelo original: ensamble con pesos locales (EPL)
#
# :::{admonition} Qué responde este capítulo
# :class: tip
# Qué es el modelo propuesto, cómo se formula y se optimiza, qué propiedades tiene (con demostración), cuánto
# cuesta, cómo está implementado y probado, y qué aporta cada uno de sus componentes.
# :::
#
# ## 5.1 Motivación
#
# Los entregables anteriores dejaron tres hechos. (i) Los modelos base se equivocan en lugares distintos: el KNN
# acierta donde la ciudad es homogénea, la logística prolonga el gradiente sur–norte y los árboles aprenden el mapa
# donde hay datos cerca (E2, cap. 7). (ii) La información de la vecindad es la que más pesa, y ningún modelo
# individual domina en todos los pliegues espaciales (E3). (iii) El XGBoost geográfico del E2/E3, que mezclaba un
# modelo global con modelos locales con un peso fijo, no ganaba en zonas nuevas y además envolvía una librería
# (no cumple la condición R2 de la guía).
#
# La idea del EPL es combinar **modelos completos** ya validados (los 7 modelos base, con su mejor combinación) con
# **pesos que cambian según la región**, y aprender esos pesos con una regla propia, convexa y con garantías. Así el
# ensamble puede creerle más al KNN en zonas homogéneas y más a la logística donde los vecinos no informan.
#
# ## 5.2 Formulación
#
# Sean $M$ expertos con probabilidades $P_m(y\mid x)$ sobre $K$ clases, y $z(x)\in\mathbb{R}^d$ las coordenadas de
# cada vivienda en un **espacio de competencia** (sección 5.2.1). Se fijan $J$ anclas $a_1,\dots,a_J$ (centros de
# K-medias de las $z_i$ de entrenamiento) y, en cada ancla, un vector de pesos $w_j\in\Delta_M$ (el símplex). El peso
# de una vivienda es la mezcla suavizada de las anclas cercanas:
#
# $$
# w(z) \;=\; \sum_{j\in N_k(z)} \kappa_j(z)\,w_j, \qquad
# \kappa_j(z) \;=\; \frac{\exp\!\big(-\lVert z-a_j\rVert^2/2h^2\big)}{\sum_{l\in N_k(z)}\exp\!\big(-\lVert z-a_l\rVert^2/2h^2\big)},
# $$ (eq-pesos)
#
# donde $N_k(z)$ son las $k$ anclas más cercanas y $h$ el ancho de banda. La predicción es
#
# $$
# p(y\mid x) \;=\; \sum_{m=1}^{M} w_m\big(z(x)\big)\,P_m(y\mid x).
# $$ (eq-mezcla)
#
# Como cada $w(z)$ está en el símplex y cada $P_m$ es una distribución, $p(\cdot\mid x)$ también lo es.
#
# **Objetivo.** Con $q_{im} = \max\{P_m^{\text{oof}}(y_i\mid x_i), \varepsilon\}$ la probabilidad que el experto
# $m$ da a la clase observada **fuera de muestra** (ajuste cruzado por bloques espaciales de 1 km), $K_{ij}$ el peso
# de núcleo de la vivienda $i$ en el ancla $j$ y $\alpha_i$ el peso de su clase (inverso de la frecuencia si
# ``peso_clase="balanced"``), los pesos de las anclas $W=(w_1,\dots,w_J)$ minimizan
#
# $$
# L(W) \;=\; -\frac{1}{\sum_i\alpha_i}\sum_{i=1}^{n}\alpha_i\log r_i(W) \;+\; \frac{\lambda}{J}\sum_{j=1}^{J}
# \mathrm{KL}(w_j\,\Vert\,\pi), \qquad r_i(W)=\sum_{j}K_{ij}\,w_j^{\top}q_i ,
# $$ (eq-objetivo)
#
# donde $\pi$ son los pesos globales (el mismo problema con $J=1$ y $\lambda=0$). La penalización KL encoge hacia
# $\pi$ las anclas con pocos datos: con el factor $1/J$ uniforme, la relación penalización/datos de un ancla crece
# cuando su masa $\sum_i K_{ij}$ es pequeña.
#
# **Gradiente y regla de actualización.**
#
# $$
# g_{jm} \;=\; \frac{\partial L}{\partial w_{jm}} \;=\; -\frac{1}{\sum_i\alpha_i}\sum_{i}\alpha_i\frac{K_{ij}\,q_{im}}{r_i}
# \;+\; \frac{\lambda}{J}\Big(\log\frac{w_{jm}}{\pi_m}+1\Big),
# \qquad
# w_{jm}^{(t+1)} \;=\; \frac{w_{jm}^{(t)}\,e^{-\eta g_{jm}}}{\sum_{m'}w_{jm'}^{(t)}\,e^{-\eta g_{jm'}}} .
# $$ (eq-eg)
#
# Es el **gradiente exponenciado** {cite}`kivinen1997`: descenso espejo con la entropía como función de distancia
# {cite}`beck2003`, que mantiene cada $w_j$ en el símplex sin proyecciones (la constante $+1$ se cancela al
# normalizar). Si una iteración no baja $L$, el paso se divide por 2 y se repite; si la baja, el siguiente intento
# usa $1.5\,\eta$. Se detiene cuando la mejora relativa es menor que $10^{-7}$ o tras $T$ iteraciones.
#
# ### 5.2.1 Espacio de competencia (versión 2)
#
# La versión 1 usaba solo $z=s$, la ubicación en km. La ablación (sección 5.8) mostró que en «zonas nuevas» ese campo
# casi no aportaba, y la versión 2 deja que el bucle interno elija entre tres espacios:
#
# - **geográfico**: $z = s$ (km, sin reescalar; la versión 1);
# - **variables**: $z$ = 2 primeras componentes principales de las variables de los expertos (PCA por SVD, ajustada
#   con el entrenamiento);
# - **probabilidades**: $z = \big(\bar e(x), \operatorname{sd}_m e_m(x)\big)$ con $e_m(x)=\sum_k k\,P_m(k\mid x)$ la
#   clase esperada por el experto $m$: dónde cae la vivienda en la escala ordinal y cuánto discrepan los expertos,
#   la información con que la selección dinámica decide a quién creerle {cite}`cruz2018`. En el entrenamiento se usan
#   las probabilidades fuera de muestra; al predecir, las de los expertos finales.
#
# Los dos espacios nuevos se reescalan para que la caja entre los percentiles 1 y 99 tenga una diagonal de 20
# unidades (≈ la ciudad en km), de modo que $h$ y $J$ signifiquen lo mismo en los tres.
#
# ## 5.3 Propiedades (con demostración)
#
# :::{admonition} Proposición 5.1 (convexidad, existencia y unicidad)
# :class: note
# :name: prop-convexidad
# $L$ es convexa en $\mathcal{W}=\Delta_M^J$, alcanza su mínimo y, si $\lambda>0$, el minimizador es único.
# :::
#
# *Demostración.* Cada $r_i(W)$ es lineal en $W$ y, como $q_{im}\ge\varepsilon$ y $\sum_j K_{ij}=1$,
# $r_i(W)\ge\varepsilon>0$ en todo $\mathcal{W}$. $-\log$ es convexa y decreciente, y su composición con una función
# lineal positiva es convexa; una suma con pesos $\alpha_i\ge 0$ conserva la convexidad. $\mathrm{KL}(w\Vert\pi)=
# \sum_m w_m\log w_m-\sum_m w_m\log\pi_m$ es la entropía negativa (estrictamente convexa en el símplex) más un término
# lineal, luego es estrictamente convexa; la suma con $\lambda/J>0$ hace a $L$ estrictamente convexa. $\mathcal{W}$
# es compacto y $L$ es continua en él (está acotada porque $r_i\in[\varepsilon,1]$), así que el mínimo existe
# (Weierstrass), y una función estrictamente convexa tiene a lo sumo un minimizador. $\square$
#
# :::{admonition} Proposición 5.2 (descenso monótono y convergencia)
# :class: note
# :name: prop-convergencia
# La sucesión $L(W^{(t)})$ que produce el algoritmo es no creciente y converge. Con paso fijo $\eta$ y gradientes
# acotados $\lVert g\rVert_\infty\le G$, el promedio $\bar W_T$ de las iteraciones cumple
# $L(\bar W_T)-L^\star \le \dfrac{J\log M}{\eta T}+\dfrac{\eta G^2}{2}$.
# :::
#
# *Demostración.* La regla acepta una iteración solo si no aumenta $L$ (si no, reduce $\eta$ hasta lograrlo o hasta
# $\eta<10^{-8}$, en cuyo caso se queda en $W^{(t)}$), así que $L(W^{(t+1)})\le L(W^{(t)})$. Como $r_i\le 1$, el
# término de datos es $\ge 0$ y el KL es $\ge 0$: la sucesión es monótona y acotada inferiormente, luego converge. La
# cota con paso fijo es la del descenso espejo con la entropía {cite}`beck2003`: en cada símplex el diámetro de
# Bregman desde el centro es $\log M$, se suman $J$ símplex, y la norma dual de la entropía es $\lVert\cdot\rVert_\infty$;
# $G$ es finito porque $q_{im}/r_i\le 1/\varepsilon$ (el piso $\varepsilon$ acota el gradiente). Además, $L$ es
# relativamente suave respecto de la entropía en el interior del símplex, lo que da tasa $O(1/t)$ sin promediar
# {cite}`lu2018`. $\square$
#
# :::{admonition} Proposición 5.3 (nunca peor que el mejor experto en la pérdida fuera de muestra)
# :class: note
# :name: prop-anidamiento
# Sea $L_{\text{datos}}$ el primer término de {eq}`eq-objetivo`. Entonces $L(W^\star)\le L_{\text{datos}}(\pi)\le
# \min_m L_{\text{datos}}(e_m)$, donde $e_m$ es el vértice del símplex que pone todo el peso en el experto $m$.
# :::
#
# *Demostración.* $W=(\pi,\dots,\pi)$ es factible, tiene KL nulo y da $r_i=\pi^\top q_i$ (porque $\sum_j K_{ij}=1$),
# así que $L(W^\star)\le L(\pi,\dots,\pi)=L_{\text{datos}}(\pi)$. $\pi$ minimiza $L_{\text{datos}}$ sobre todo el
# símplex, que contiene a los vértices $e_m$. $\square$
#
# La proposición 5.3 es una garantía sobre las predicciones **fuera de muestra del entrenamiento**: el
# combinador no puede empeorar la log-pérdida ponderada del mejor experto ni la de los pesos globales. No garantiza
# un mejor F1 en datos nuevos; eso es lo que contrasta el capítulo 6.
#
# :::{admonition} Proposición 5.4 (complejidad)
# :class: note
# :name: prop-complejidad
# Con $B$ pliegues del ajuste cruzado y $C_m(n,p)$, $I_m(p)$ los costos de entrenar y predecir del experto $m$:
# entrenamiento $O\big((B+1)\sum_m C_m(n,p) + n\,d\,J\,R + T\,n\,k\,M\big)$, con $R$ iteraciones de K-medias y $T$
# del gradiente exponenciado; inferencia por vivienda $O\big(\sum_m I_m(p)+\log J + kM\big)$ (árbol k-d sobre las
# anclas); memoria adicional $O(nM+JM)$.
# :::
#
# *Demostración.* El ajuste cruzado entrena cada experto $B$ veces sobre $\approx n(B-1)/B$ filas y una vez con todo.
# K-medias cuesta $O(nJd)$ por iteración. Cada iteración del gradiente exponenciado calcula $r_i$ ($k$ anclas × $M$
# expertos por fila) y acumula el gradiente con el mismo costo; la normalización cuesta $O(JM)$. Al predecir se
# consultan las $k$ anclas más cercanas en un árbol k-d ($O(\log J)$ en promedio) y se mezclan $M$ vectores. $\square$
#
# En la práctica domina el primer término (los expertos): el combinador cuesta segundos (sección 5.7).
#
# ## 5.4 Pseudocódigo
#
# ```text
# Entrada:  X, s, y (entrenamiento); M expertos; J, h, k, λ, η0, T, tol, ε; espacio de competencia; B, b
# Salida:   expertos ajustados f_1..f_M; transformación del espacio; anclas a_1..a_J; pesos W y π
#
# 1  celdas ← floor(s / b);  asignar cada celda a uno de B pliegues al azar (semilla)
# 2  para cada pliegue f y experto m:  ajustar f_m sin f;  P_oof[f, m] ← f_m(x) en f
# 3  para cada m: reajustar f_m con todos los datos
# 4  q_im ← max(P_oof[i, m, y_i], ε);  α_i ← n / (K · n_{y_i})  (o 1)
# 5  π ← GE(K = 1 columna de unos, λ = 0, w0 = uniforme)                     # pesos globales
# 6  z ← coordenadas de competencia (geográfico | 2 CP de x | clase esperada y desacuerdo); escalar
# 7  a ← K-medias++(z, J);  (idx, K) ← k anclas más cercanas y núcleo gaussiano normalizado
# 8  W ← GE(K, λ, w0 = π en cada ancla)
#
# GE(K, λ, w0):  W ← w0;  η ← η0;  L ← L(W)
#    repetir t = 1..T:
#       G ← ∇L(W)                                            # ecuación (eq-eg)
#       repetir:  W' ← normalizar_filas(W ⊙ exp(−η G));  si L(W') ≤ L o η < 1e−8: salir;  η ← η/2
#       mejora ← (L − L(W')) / |L|;  W, L ← W', L(W');  si mejora < tol: parar;  η ← 1.5 η
#
# Predicción(x):  z ← coordenadas(x);  w ← Σ_{j∈N_k(z)} K_j(z) w_j;  devolver Σ_m w_m f_m(x)
# ```
#
# ## 5.5 Implementación
#
# La clase ``EnsamblePesosLocales`` hereda de ``BaseEstimator`` y ``ClassifierMixin`` (``fit``, ``predict``,
# ``predict_proba``), así que funciona dentro de ``Pipeline``, con ``clone`` y en la validación anidada. El núcleo
# (K-medias, núcleo, pérdida, gradiente, gradiente exponenciado, espacio de competencia) usa solo numpy y scipy; los
# expertos son estimadores cualesquiera. Las piezas principales, comentadas en el código:
#
# ```{literalinclude} ../../src/original_model.py
# :language: python
# :pyobject: gradiente
# ```
#
# ```{literalinclude} ../../src/original_model.py
# :language: python
# :pyobject: gradiente_exponenciado
# ```
#
# ```{literalinclude} ../../src/original_model.py
# :language: python
# :pyobject: coordenadas_crudas
# ```
#
# El archivo completo es ``src/original_model.py`` (≈ 600 líneas con documentación).
#
# ## 5.6 Pruebas unitarias
#
# ``tests/test_original_model.py`` verifica lo que pide la guía y las propiedades de la sección 5.3:

# %%
import subprocess
import sys
from pathlib import Path

RAIZ = Path.cwd().resolve().parents[1]
r = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", "tests/test_original_model.py",
                    "tests/test_explicacion.py"], cwd=RAIZ, capture_output=True, text=True)
for linea in r.stdout.splitlines():
    if "::" in linea and ("PASSED" in linea or "FAILED" in linea):
        nombre, estado = linea.split("::", 1)[1].split()[0], ("pasa" if "PASSED" in linea else "FALLA")
        print(f"{estado:5s}  {nombre}")
    elif " passed" in linea or " failed" in linea:
        print(linea.strip("= "))

# %% [markdown]
# | Requisito de la guía | Prueba |
# |---|---|
# | (i) aprende donde la solución es conocida | ``test_aprende_el_campo_de_pesos_conocido`` (ciudad con dos mitades: el EPL acierta > 90 % y pone > 0.9 de peso en el experto correcto de cada mitad); ``test_espacio_variables_encuentra_la_competencia`` (la competencia depende de una variable, no del lugar: el espacio de variables gana al geográfico) |
# | (ii) reproducibilidad con la misma semilla | ``test_misma_semilla_mismo_modelo`` |
# | (iii) entradas inválidas | ``test_parametros_invalidos`` (8 casos), ``test_datos_invalidos``, ``test_predecir_sin_ajustar`` |
# | (iv) compatibilidad con ``clone`` | ``test_clone_y_parametros``, ``test_pipeline_y_validacion_cruzada``, ``test_etiquetas_no_numericas`` |
# | Propiedades de la sección 5.3 | ``test_perdida_monotona_no_creciente``, ``test_gradiente_analitico_igual_al_numerico``, ``test_converge_al_optimo_de_un_caso_cerrado``, ``test_lambda_grande_lleva_a_los_pesos_globales``, ``test_probabilidades_y_pesos_en_el_simplex`` |
#
# ## 5.7 Complejidad empírica
#
# Tiempos medidos con el equipo libre al variar el número de viviendas $n$ y el de variables $p$ (``experimentos/12_complejidad.py``):

# %%
import json

import pandas as pd

ruta = RAIZ / "resultados" / "complejidad" / "complejidad.json"
if ruta.exists():
    c = json.loads(ruta.read_text(encoding="utf-8"))
    display(pd.DataFrame(c["tabla"]))
    print(c.get("lectura", ""))
else:
    print("Pendiente: correr experimentos/12_complejidad.py")

# %% [markdown]
# **Lectura.** El combinador crece casi linealmente con $n$ (pendiente log-log ≈ 0.9; la proposición 5.4 predice 1) y
# tarda menos de medio segundo con 8 000 viviendas; el número de expertos $M$ y de anclas $J$ casi no pesa a esta
# escala (dominan el núcleo y la normalización). El EPL completo tarda ≈ 90 s con 8 000 viviendas: más del 99 % es
# entrenar los 7 expertos 6 veces (5 pliegues del ajuste cruzado y el ajuste final), que crece con $n$ y con $p$ como
# los propios expertos. La inferencia cuesta ≈ 0.3 ms por vivienda.
#
# ## 5.8 Ablación
#
# Cada componente se retira por separado, con el resto de la configuración que eligió el bucle interno, en los mismos
# 25 pliegues externos y con los mismos expertos (las probabilidades se reutilizan, así que la única diferencia es el
# componente). También se contrasta con las reglas estándar de combinación que pide el Cuadro 1 de la guía.

# %%
filas = []
for v in ("v1", "v2"):
    ruta = RAIZ / "resultados" / "cv5x5" / f"estadistica_{v}.json"
    if not ruta.exists():
        continue
    e = json.loads(ruta.read_text(encoding="utf-8"))
    media_epl = next(x["media"] for x in e["resumen"] if x["modelo"] == "EPL (original)")
    for x in e["epl_vs_variantes"]:
        filas.append({"versión": v, "variante": x["rival"], "F1 medio": round(media_epl - x["dif_media"], 4),
                      "EPL − variante": round(x["dif_media"], 4),
                      "IC 95 % BCa": f"[{x['IC95_BCa'][0]:+.4f}, {x['IC95_BCa'][1]:+.4f}]",
                      "pliegues a favor": f"{x['pliegues_a_favor']}/25", "Wilcoxon p (Holm)": round(x["wilcoxon_p_holm"], 4)})
pd.DataFrame(filas) if filas else print("Pendiente: estadística de la 5×5")
