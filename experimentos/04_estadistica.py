# -*- coding: utf-8 -*-
"""Pruebas de hipótesis del modelo original frente a los benchmarks en la validación repetida 5×5 (bloque D).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Unidad de análisis: los 25 pliegues externos (5 repeticiones × 5 pliegues kNNDM), pareados: todos los modelos se
evaluaron en las mismas particiones, con la misma semilla y el mismo preprocesamiento.

Ruta de la guía (Figura 4, clasificación, datos de corte transversal):

1. **Friedman + Nemenyi** (procedimiento global) con el modelo original dentro del universo: 7 benchmarks de la
   guía, el XGBoost geográfico (referencia), LightGBM y CatBoost (recientes) y el EPL. Diagrama de diferencia
   crítica.
2. **Wilcoxon de rangos con signo** pareado, EPL frente a cada benchmark, con Holm sobre las 10 comparaciones.
3. **t corregida de Nadeau y Bengio** para validación cruzada repetida (Bouckaert y Frank, 2004:
   var × (1/(r·k) + n_val/n_train), gl = r·k − 1), con Holm sobre las 10 comparaciones.
4. Complementos: prueba bayesiana de rangos con signo con región de equivalencia práctica (ROPE = 0.01 de F1;
   Benavoli et al., 2017), δ de Cliff, d de Cohen pareada e IC 95 % BCa de la diferencia media (remuestreando
   pliegues).

Una segunda familia compara el EPL con las reglas estándar de combinación (promedio simple, pesos por error global,
stacking lineal) y con sus variantes de ablación.

Superioridad (sección 6.1 de la guía): (1) media mayor que la de cada benchmark; (2) al menos tres pruebas distintas
rechazan la igualdad con p ajustado < 0.05 frente al mejor benchmark; (3) |δ de Cliff| ≥ 0.147. McNemar y DeLong en
el test los agrega ``05_test_confirmatorio.py``.

Uso (desde la raíz):  python -m experimentos.04_estadistica [v1|v2]      (v2 por defecto)

Versiones del modelo original: ``v1`` = campo de pesos geográfico (``03_epl_5x5.py``); ``v2`` = espacio de
competencia elegido en el bucle interno (``03b_epl_v2_5x5.py``). Cada una se compara por separado con los mismos
rivales y deja ``estadistica_<v>.json``, ``f1_por_pliegue_<v>.csv`` y ``figuras/cd_5x5_<v>.png``.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from estrato3 import estadistica as ES
from experimentos import comun as K

ROPE = 0.01
ORIGINAL = "EPL (original)"
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
CARPETA, CLAVE = {"v1": ("epl", "EPL"), "v2": ("epl_v2", "EPL v2")}[VERSION]


def tabla_f1():
    """F1 macro por pliegue: filas (r, f), columnas = 8 benchmarks + EPL; y las variantes del EPL aparte."""
    rivales = [m for m, _, _, _ in K.benchmarks()
               if all(K.ruta_benchmark(m, r, f).exists() for r in range(K.N_REPETICIONES) for f in range(5))]
    faltan = [m for m, _, _, _ in K.benchmarks() if m not in rivales]
    if faltan:
        K.log(f"AVISO: sin los 25 pliegues todavía, se dejan fuera: {faltan}")
    filas, variantes, n_val, n_tr = [], [], [], []
    for r in range(K.N_REPETICIONES):
        for f in range(5):
            fila = {"rep": r, "fold": f}
            for m in rivales:
                j, _, _ = K.leer_benchmark(m, r, f)
                fila[m] = j["metricas"]["f1_macro"]
            e = json.loads((K.RES / CARPETA / f"r{r}_f{f}.json").read_text(encoding="utf-8"))
            fila[ORIGINAL] = e["f1"][CLAVE]
            filas.append(fila)
            variantes.append({"rep": r, "fold": f, **{(ORIGINAL if k == CLAVE else k): v for k, v in e["f1"].items()}})
            n_val.append(e["n_val"])
            n_tr.append(e["n_train"])
    T = pd.DataFrame(filas).set_index(["rep", "fold"])
    V = pd.DataFrame(variantes).set_index(["rep", "fold"])
    return T, V, float(np.mean(n_val)), float(np.mean(n_tr))


def t_corregida_repetida(d, n_val, n_tr):
    """t corregida de Nadeau-Bengio para r×k pliegues (Bouckaert y Frank, 2004). Devuelve t, gl y p de dos colas."""
    d = np.asarray(d, float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"t": float("nan"), "gl": J - 1, "p": 1.0}
    t = d.mean() / np.sqrt((1 / J + n_val / n_tr) * var)
    return {"t": float(t), "gl": J - 1, "p": float(2 * stats.t.sf(abs(t), J - 1))}


def wilcoxon(d):
    """Wilcoxon de rangos con signo (dos colas, método de Pratt para ceros) y su estadístico."""
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return {"W": float("nan"), "p": 1.0}
    res = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided")
    return {"W": float(res.statistic), "p": float(res.pvalue)}


def bayes_rangos_signo(d, rope=ROPE, prior=0.5, n_muestras=50_000, seed=K.SEED):
    """Prueba bayesiana de rangos con signo (Benavoli et al., 2017) con una pseudo-observación previa en 0.

    Devuelve las probabilidades a posteriori de que la diferencia esté a la izquierda de la ROPE (el EPL es peor),
    dentro de la ROPE (equivalencia práctica) o a la derecha (el EPL es mejor).
    """
    rng = np.random.default_rng(seed)
    z = np.concatenate([[0.0], np.asarray(d, float)])
    alfa = np.ones(len(z))
    alfa[0] = prior
    w = rng.dirichlet(alfa, n_muestras)
    s = z[:, None] + z[None, :]
    izq, der = (s < -2 * rope).astype(float), (s > 2 * rope).astype(float)
    p_izq = np.einsum("si,ij,sj->s", w, izq, w)
    p_der = np.einsum("si,ij,sj->s", w, der, w)
    p_rope = 1 - p_izq - p_der
    gana = np.argmax(np.column_stack([p_izq, p_rope, p_der]), axis=1)
    return {"P(EPL peor)": float(np.mean(gana == 0)), "P(equivalentes)": float(np.mean(gana == 1)),
            "P(EPL mejor)": float(np.mean(gana == 2))}


def ic_bca(d, seed=K.SEED):
    """IC 95 % BCa de la diferencia media remuestreando pliegues."""
    d = np.asarray(d, float)
    if np.allclose(d, d[0]):
        return [float(d[0]), float(d[0])]
    res = stats.bootstrap((d,), np.mean, n_resamples=9999, method="BCa", random_state=seed)
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def supuestos(d):
    """Supuestos que se revisan: normalidad de las diferencias (t) y simetría (Wilcoxon)."""
    d = np.asarray(d, float)
    return {"shapiro_p": float(stats.shapiro(d).pvalue) if np.ptp(d) > 0 else None,
            "asimetria": float(stats.skew(d)), "ceros": int(np.sum(np.isclose(d, 0)))}


def comparar(T, a, rivales, n_val, n_tr):
    """EPL frente a cada rival: tres pruebas con p crudo y Holm, tamaños del efecto, IC e inferencia bayesiana."""
    filas = []
    for b in rivales:
        d = (T[a] - T[b]).to_numpy()
        w, t = wilcoxon(d), t_corregida_repetida(d, n_val, n_tr)
        delta, mag = ES.delta_cliff(T[a].to_numpy(), T[b].to_numpy())
        filas.append({"rival": b, "media_rival": float(T[b].mean()), "dif_media": float(d.mean()),
                      "IC95_BCa": ic_bca(d), "pliegues_a_favor": int((d > 0).sum()), "n_pliegues": len(d),
                      "wilcoxon_W": w["W"], "wilcoxon_p": w["p"], "t_NB": t["t"], "t_NB_gl": t["gl"],
                      "t_NB_p": t["p"], "delta_cliff": delta, "delta_magnitud": mag,
                      "d_cohen_pareada": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"),
                      "bayes": bayes_rangos_signo(d), "supuestos": supuestos(d)})
    for prueba in ("wilcoxon", "t_NB"):
        ajust = ES.holm([x[f"{prueba}_p"] for x in filas])
        for x, p in zip(filas, ajust):
            x[f"{prueba}_p_holm"] = float(p)
    return filas


def diagrama(rangos, cd, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = ES.diagrama_cd(rangos, cd, titulo=f"Diferencia crítica de Nemenyi (25 pliegues, CD = {cd:.2f})")
    ax.figure.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(ax.figure)


def main():
    T, V, n_val, n_tr = tabla_f1()
    benchmarks = [c for c in T.columns if c != ORIGINAL]
    guia = [m for m, _, _, g in K.benchmarks() if g == "guia" and m in benchmarks]
    K.log(f"tabla {T.shape}; n_val medio {n_val:.0f}, n_train medio {n_tr:.0f}")

    # 1. Friedman + Nemenyi (universo completo, EPL incluido)
    fr = ES.friedman(T)
    cd = ES.diferencia_critica(T.shape[1], T.shape[0])
    nem = ES.nemenyi(T)
    figs = K.RAIZ / "resultados" / "figuras"
    figs.mkdir(parents=True, exist_ok=True)
    diagrama(fr["rangos"], cd, figs / f"cd_5x5_{VERSION}.png")

    # 2–4. EPL frente a cada benchmark
    comp = comparar(T, ORIGINAL, benchmarks, n_val, n_tr)
    for x in comp:
        x["nemenyi_p"] = float(nem.loc[ORIGINAL, x["rival"]])

    # Veredicto (sección 6.1) frente al mejor benchmark por media y frente a los 7 de la guía
    medias = T.mean()
    mejor = medias[benchmarks].idxmax()
    c_mejor = next(x for x in comp if x["rival"] == mejor)
    pruebas_rechazan = {"Friedman→Nemenyi": fr["p"] < 0.05 and c_mejor["nemenyi_p"] < 0.05,
                        "Wilcoxon-Holm": c_mejor["wilcoxon_p_holm"] < 0.05,
                        "t Nadeau-Bengio-Holm": c_mejor["t_NB_p_holm"] < 0.05}
    veredicto = {"mejor_benchmark": mejor,
                 "cond1_media_mayor_que_todos": bool(all(medias[ORIGINAL] > medias[b] for b in benchmarks)),
                 "cond1_media_mayor_que_7_guia": bool(all(medias[ORIGINAL] > medias[b] for b in guia)),
                 "cond2_pruebas_que_rechazan_vs_mejor": pruebas_rechazan,
                 "cond2_n_rechazos": int(sum(pruebas_rechazan.values())),
                 "cond3_delta_cliff_vs_mejor": c_mejor["delta_cliff"],
                 "cond3_no_despreciable": bool(abs(c_mejor["delta_cliff"]) >= 0.147)}
    veredicto["R3_en_validacion"] = bool(veredicto["cond1_media_mayor_que_todos"] and
                                         veredicto["cond2_n_rechazos"] >= 3 and veredicto["cond3_no_despreciable"]
                                         and c_mejor["dif_media"] > 0)

    # Segunda familia: reglas estándar y ablación (EPL frente a cada variante)
    W = V
    otras = [c for c in W.columns if c != ORIGINAL]
    familia2 = comparar(W, ORIGINAL, otras, n_val, n_tr)

    resumen = pd.DataFrame({"media": T.mean(), "sd": T.std(ddof=1), "rango_friedman": fr["rangos"]}).sort_values(
        "media", ascending=False)
    salida = {"n_pliegues": len(T), "n_comparaciones_familia1": len(benchmarks),
              "n_comparaciones_familia2": len(otras), "rope": ROPE, "n_val_medio": n_val, "n_train_medio": n_tr,
              "friedman": {"chi2": fr["chi2"], "gl": fr["gl"], "p": fr["p"], "cd": cd,
                           "rangos": fr["rangos"].to_dict()},
              "nemenyi_p": nem.round(6).to_dict(), "resumen": resumen.reset_index(names="modelo").to_dict("records"),
              "epl_vs_benchmarks": comp, "epl_vs_variantes": familia2, "veredicto_validacion": veredicto,
              "tabla_f1": T.reset_index().to_dict("records"), "tabla_variantes": V.reset_index().to_dict("records")}
    salida["version"] = VERSION
    K.escribir_json(K.RES / f"estadistica_{VERSION}.json", salida)
    T.reset_index().to_csv(K.RES / f"f1_por_pliegue_{VERSION}.csv", index=False)
    print(resumen.round(4).to_string())
    print(f"\nFriedman χ² = {fr['chi2']:.1f}, p = {fr['p']:.2e}, CD = {cd:.2f}")
    cols = ["rival", "dif_media", "IC95_BCa", "pliegues_a_favor", "wilcoxon_p", "wilcoxon_p_holm", "t_NB_p",
            "t_NB_p_holm", "nemenyi_p", "delta_cliff"]
    print(pd.DataFrame(comp)[cols].round(4).to_string(index=False))
    print("\nVariantes:")
    print(pd.DataFrame(familia2)[cols[:-2] + ["delta_cliff"]].round(4).to_string(index=False))
    print("\nVeredicto:", json.dumps(veredicto, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
