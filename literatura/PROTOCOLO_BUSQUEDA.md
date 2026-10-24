# Protocolo de búsqueda: evidencia de novedad del modelo original (sección 5.2 de la guía)

Modelo propuesto: **ensamble con pesos locales por gradiente exponenciado (EPL)**. Combina las probabilidades de
varios clasificadores con un campo de pesos que varía en el espacio geográfico (anclas + núcleo gaussiano),
aprendido minimizando una log-pérdida convexa con penalización KL mediante gradiente exponenciado, sobre
predicciones fuera de muestra por bloques espaciales.

La guía pide: **≥ 3 bases de datos**, **≥ 5 cadenas** que combinen (i) el modelo o su familia, (ii) el problema y
(iii) sinónimos en inglés y en español; la **fecha** de cada consulta; y una **matriz de novedad** (resultados,
revisados, relevantes, observación).

Reparto: **tú** corres Scopus, Web of Science y Google Scholar (acceso institucional; Scholar pide CAPTCHA). **Yo**
corro arXiv por su API y reviso títulos y resúmenes de lo que exportes.

---

## Cómo registrar cada búsqueda

1. Copia la cadena tal cual (sin refinar con filtros, como en la búsqueda de WoS del 19-oct).
2. Anota la **fecha** y el **número de resultados**.
3. Toma una **captura** de la página de resultados (guárdala en `e4/literatura/capturas/<base>_<C#>.png`).
4. **Exporta** los resultados a CSV (Scopus: *Export → CSV*, con título, autores, año, resumen y palabras clave;
   WoS: *Export → Excel* o *Tab delimited*, registro completo; Scholar: no exporta, copia los títulos de las 3
   primeras páginas a un .txt). Si hay más de 200 resultados, exporta los 200 más relevantes. Guárdalos en
   `e4/literatura/exportes/`.
5. Llena la fila en la tabla del final (o pásame los números y yo la lleno).

---

## Cadenas

### C1. Aplicación (estrato con aprendizaje automático y datos de vivienda)
- **Scopus:** `TITLE-ABS-KEY ( ( "socioeconomic stratum" OR "socioeconomic strata" OR "socioeconomic stratification" OR "socio-economic stratum" ) AND ( "machine learning" OR classifier OR classification OR ensemble ) AND ( cadastr* OR "property data" OR housing OR dwelling* ) )`
- **WoS:** `TS=( ("socioeconomic stratum" OR "socioeconomic strata" OR "socioeconomic stratification" OR "socio-economic stratum") AND ("machine learning" OR classifier OR classification OR ensemble) AND (cadastr* OR "property data" OR housing OR dwelling*) )`
- **Scholar:** `"socioeconomic stratum" "machine learning" cadastral housing classification`

### C2. Modelo propuesto aplicado al problema
- **Scopus:** `TITLE-ABS-KEY ( ( "locally weighted ensemble" OR "local ensemble weights" OR "spatially varying weights" OR "geographically weighted ensemble" OR "dynamic ensemble selection" OR "mixture of experts" ) AND ( socioeconomic OR "social stratum" OR housing OR cadastr* ) )`
- **WoS:** `TS=( ("locally weighted ensemble" OR "local ensemble weights" OR "spatially varying weights" OR "geographically weighted ensemble" OR "dynamic ensemble selection" OR "mixture of experts") AND (socioeconomic OR "social stratum" OR housing OR cadastr*) )`
- **Scholar:** `"geographically weighted ensemble" OR "locally weighted ensemble" socioeconomic housing classification`

### C3. Familia algorítmica (regla de combinación por gradiente exponenciado), sin restringir la aplicación
- **Scopus:** `TITLE-ABS-KEY ( ( "exponentiated gradient" OR "multiplicative weights" OR "hedge algorithm" ) AND ( ensemble OR "mixture weights" OR "classifier combination" OR "combining classifiers" ) AND ( spatial OR geographic* OR local* ) )`
- **WoS:** `TS=( ("exponentiated gradient" OR "multiplicative weights" OR "hedge algorithm") AND (ensemble OR "mixture weights" OR "classifier combination" OR "combining classifiers") AND (spatial OR geographic* OR local*) )`
- **Scholar:** `"exponentiated gradient" ensemble weights spatial classification`

### C4. Ensambles geográficamente ponderados (métodos afines de otras aplicaciones)
- **Scopus:** `TITLE-ABS-KEY ( "geographically weighted" AND ( ensemble OR stacking OR "model averaging" OR "model combination" ) AND ( classification OR classifier ) )`
- **WoS:** `TS=( "geographically weighted" AND (ensemble OR stacking OR "model averaging" OR "model combination") AND (classification OR classifier) )`
- **Scholar:** `"geographically weighted" stacking OR ensemble classification spatially varying weights`

### C5. Selección dinámica de clasificadores con información espacial
- **Scopus:** `TITLE-ABS-KEY ( ( "dynamic ensemble selection" OR "dynamic classifier selection" OR KNORA OR "META-DES" ) AND ( spatial OR geographic* OR "remote sensing" OR urban OR housing ) )`
- **WoS:** `TS=( ("dynamic ensemble selection" OR "dynamic classifier selection" OR KNORA OR "META-DES") AND (spatial OR geographic* OR "remote sensing" OR urban OR housing) )`
- **Scholar:** `"dynamic ensemble selection" spatial housing urban classification`

### C6. Español (estrato y aprendizaje automático)
- **Scopus:** `TITLE-ABS-KEY ( ( "estrato socioeconómico" OR "estratificación socioeconómica" OR "estratos socioeconómicos" ) AND ( "aprendizaje automático" OR "machine learning" OR clasificación OR ensamble ) )`
- **WoS:** `TS=( ("estrato socioeconómico" OR "estratificación socioeconómica" OR "estratos socioeconómicos") AND ("aprendizaje automático" OR "machine learning" OR clasificación OR ensamble) )`
- **Scholar:** `"estrato socioeconómico" "aprendizaje automático" catastro vivienda`
- Opcional, si tienes tiempo: la misma cadena en **SciELO** o **Redalyc** (el E2 reconoció que faltaban fuentes
  latinoamericanas).

---

## Matriz de novedad

Llenada el 23-oct-2026 con las búsquedas de la usuaria (Scopus y Google Scholar) y las de arXiv. La matriz completa
está en el capítulo 4 del libro (`book/notebooks/04_literatura.py`) y el cribado registro por registro, con el motivo
de cada decisión, en `cribado.csv`. Exportes de Scopus en `exportes/`; captura del error de Web of Science en
`capturas/`.

C3 en Google Scholar se repitió el 23-oct-2026 (376 resultados, 10 revisados, 4 relevantes). Web of Science se repitió el mismo 23-oct-2026 con las seis cadenas (5, 13, 1, 7, 16 y 0 resultados).

## Tabla de métodos afines (se completa con la revisión; 5–10 por familia algorítmica)

| Método | Familia | Qué hace | Qué conserva el EPL | Qué es nuevo en el EPL |
|---|---|---|---|---|
| Promedio simple / votación | Combinación fija | Pesos iguales | Mezcla de probabilidades | Pesos aprendidos y locales |
| Pesos por error global / EG global (Helmbold et al., 1997) | Combinación aprendida global | Un vector de pesos para todo el espacio | Gradiente exponenciado sobre el símplex | Campo de pesos que varía en el espacio, con suavizado por núcleo y penalización KL hacia los globales |
| Stacking lineal (Wolpert, 1992) | Meta-aprendiz | Regresión sobre las salidas de los modelos | Uso de predicciones fuera de muestra | Pesos en el símplex (interpretables, convexos), dependientes de la ubicación |
| Selección dinámica (KNORA, META-DES) | Selección local | Elige modelos por su acierto en vecinos del espacio de variables | Competencia local | Optimiza una pérdida global convexa; pesos suaves en el espacio geográfico; ajuste cruzado por bloques |
| Mezcla de expertos (Jacobs et al., 1991) | Compuerta aprendida | Compuerta (red o softmax) entrena expertos y pesos a la vez, problema no convexo (EM) | Pesos que dependen de la entrada | Expertos fijos ya entrenados; problema convexo con óptimo único; compuerta no neuronal |
| Regresión geográficamente ponderada (GWR) / GW-RF | Modelos locales | Ajusta un modelo por ubicación | Núcleo espacial y ancho de banda | Pondera modelos completos en lugar de reajustar coeficientes; clasificación multiclase |
| XGBoost geográfico (E2/E3) | Mezcla global + locales | Promedio fijo α entre un XGBoost global y uno local por ancla | Anclas y vecindad | Pesos aprendidos por ancla y por experto; cualquier familia de expertos |
