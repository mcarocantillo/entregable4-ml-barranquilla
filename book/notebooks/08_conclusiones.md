# 8. Conclusiones y limitaciones

:::{admonition} Qué responde este capítulo
:class: tip
Qué se encontró, cómo queda el modelo original frente a las condiciones R1, R2 y R3, qué limita los resultados, qué
sigue y cómo se usaron herramientas de inteligencia artificial.
:::

## 8.1 Hallazgos

1. **El estrato se recupera en buena parte, pero no del todo, con el catastro abierto.** Los mejores modelos llegan a un
   F1 macro de 0.60–0.62 en zonas nuevas (validación repetida) y de 0.56–0.59 en el test; el 99 % de las predicciones del
   modelo original cae en el estrato real o en uno vecino. Lo que falta (el puntaje de las edificaciones y las zonas
   homogéneas del DANE) no es público.
2. **La ubicación y la vecindad física dominan** sobre las características de la propia vivienda (capítulo 2): el
   estrato se asigna por zona.
3. **Varios modelos quedan en un empate técnico en la cabeza.** Random Forest, CatBoost, XGBoost y el modelo original
   están a menos de 0.02 de F1; Nemenyi no los separa.
4. **Elegir y reportar con los mismos pliegues infla el desempeño.** En particiones nuevas todos los modelos bajan
   entre 0.04 y 0.10 de F1 frente a lo que reportó el E3.
5. **El modelo original (EPL) tiene el mayor F1 medio** de los 11 modelos en la validación repetida (0.618 frente a
   0.605 de CatBoost y Random Forest), pero la ventaja es pequeña. Lo que realmente aporta es combinar expertos con
   pesos aprendidos sobre **predicciones fuera de muestra por bloques**; el campo de pesos local no aporta de forma
   medible en zonas nuevas, ni en la versión 1 (geográfica) ni en la 2 (espacio elegido por validación).

## 8.2 Veredicto sobre R1, R2 y R3

| Condición | Veredicto | Evidencia |
|---|---|---|
| **R1. Novedad en la aplicación** | **Sí** | Búsqueda en Scopus, Google Scholar y arXiv (6 cadenas, 23-oct-2026) y Web of Science (E2): nadie aplica un ensamble con pesos locales —ni otro modelo— para predecir el estrato de viviendas con el catastro. La idea de pesos por región existe en otras aplicaciones (Zhou y Lin, 2026; Liu, 2024): la novedad declarada es la regla de aprendizaje, el ajuste cruzado espacial, la elección del espacio de competencia y la aplicación (capítulo 4). |
| **R2. Implementación algorítmica propia** | **Sí** | Clase propia compatible con scikit-learn (``src/original_model.py``): K-medias, núcleo, pérdida convexa, gradiente y gradiente exponenciado programados con numpy y scipy; formulación, pseudocódigo, cuatro proposiciones demostradas, 30 pruebas unitarias y ablación (capítulo 5). |
| **R3. Superioridad estadísticamente verificada** | **No** | Mayor media que todos los benchmarks, pero frente al mejor (CatBoost) solo Wilcoxon-Holm rechaza; la t de Nadeau-Bengio y Nemenyi no, y el δ de Cliff es despreciable. El test y los conjuntos adicionales no confirman la ventaja (capítulo 6). |

Por la regla de la sección 12.1 de la guía, **el tope de la calificación es 4.0**. Reportar este resultado desfavorable
es lo que pide la guía; manipular particiones, semillas o pruebas para obtener significancia no lo sería.

## 8.3 Limitaciones

- **Datos.** El estrato es una etiqueta administrativa, puede estar desactualizado y tiene ajustes manuales; faltan los
  insumos del DANE; el 3.1 % de las viviendas (casi todas informales de estrato 1) no se pudo ubicar.
- **Validación.** El test tiene 7 bloques (uno con 3 viviendas) y todo el estrato 6 cae en uno; los modelos se comparan
  en una submuestra de 10 001 viviendas por costo de cómputo; el test se reutilizó en los cuatro entregables.
- **Modelo original.** Su costo está dominado por entrenar los expertos 6 veces (ajuste cruzado); el campo local no se
  traslada a zonas sin datos cerca; con 12 evaluaciones, la búsqueda que incluye el espacio de competencia explora
  menos cada espacio que una búsqueda con el espacio fijo.
- **Revisión de literatura.** Web of Science no estuvo disponible el día de la búsqueda; en Google Scholar solo se
  revisaron los 10 primeros resultados cuando había más de 50.
- **Uso.** El modelo no debe usarse para decisiones sobre personas (crédito, seguros, programas sociales).

## 8.4 Trabajo futuro

- Evaluar el EPL en el escenario de **zonas conocidas** (interpolar dentro de barrios con datos), donde el E2 mostró que
  la información local sí ayuda y donde, según Zhou y Lin (2026), una combinación por regiones tiene más posibilidades.
- Agregar información que hoy falta: imágenes de fachadas o satelitales, variables del barrio, y un modelo en dos etapas
  como el del DANE (zonas homogéneas y luego viviendas atípicas).
- Usar más bloques de test y una validación con todo el entrenamiento en lugar de la submuestra.
- Ampliar la búsqueda bibliográfica a Web of Science, SciELO y Redalyc.

## 8.5 Declaración de autoría y uso de herramientas de inteligencia artificial

:::{admonition} Borrador: revisar, completar y firmar por el equipo
:class: danger
Esta sección debe reflejar exactamente cómo trabajó el equipo. El texto siguiente es un punto de partida.
:::

Los autores, María Carolina Cantillo Orozco y Juan Camilo Oñoro Araujo, son responsables de todo el contenido de este
entregable y declaran que pueden explicar y modificar cualquier parte del código y del análisis.

Se usó **Claude (Anthropic)**, un asistente de inteligencia artificial, como apoyo para: revisar y resumir los entregables
anteriores; proponer, implementar y probar el código del modelo original, de los experimentos y del dashboard; ejecutar
las corridas de cómputo; diseñar las pruebas estadísticas; buscar en arXiv; y redactar borradores de los capítulos. Los
autores definieron el problema y los datos (entregables 1 a 3), tomaron las decisiones de diseño y de alcance, hicieron
las búsquedas bibliográficas en Scopus, Web of Science y Google Scholar, revisaron los resultados y el texto, y
corrigieron lo necesario. Ninguna implementación se copió de repositorios de terceros: las librerías usadas se citan y
el núcleo del modelo original está programado con numpy y scipy.

Firmas: ______________________ (María Carolina Cantillo Orozco) ______________________ (Juan Camilo Oñoro Araujo)

Fecha: ____ de octubre de 2026
