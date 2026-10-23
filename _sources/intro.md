# Estrato socioeconómico en Barranquilla — Proyecto final de Machine Learning

**María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)**
Profesor Lihki Rubio · Machine Learning · Universidad del Norte · octubre de 2026

**Dashboard:** *(URL de Render: se completa al desplegar)* · **Repositorio:** [github.com/mcarocantillo/entregable4-ml-barranquilla](https://github.com/mcarocantillo/entregable4-ml-barranquilla) ·
Entregables anteriores: [E1](https://mcarocantillo.github.io/entregable1-ml-barranquilla/v3/),
[E2](https://mcarocantillo.github.io/entregable2-ml-barranquilla/v3/),
[E3](https://mcarocantillo.github.io/entregable3-ml-barranquilla/)

## En una página

**Problema.** El estrato (1 a 6) clasifica las viviendas para cobrar los servicios públicos con subsidios cruzados
(Ley 142 de 1994): los estratos 1 a 3 reciben subsidio y el 5 y el 6 contribuyen. El DANE lo asigna con información que no
es pública.

**Pregunta.** ¿En qué medida las características físicas y catastrales de una vivienda y su ubicación permiten predecir
su estrato en Barranquilla?

**Objetivos.** (1) Comparar, con una validación espacial honesta, las mejores combinaciones de 7 modelos de referencia y
dos métodos recientes; (2) diseñar, implementar y validar estadísticamente un modelo original; (3) ponerlos a disposición
en un dashboard con predicción en vivo y su explicación.

**Respuesta.**

- Con el catastro abierto se recupera buena parte del estrato: **F1 macro ≈ 0.60–0.62** en zonas nuevas y ≈ 0.58 en el
  test, con casi todos los errores entre estratos vecinos. La ubicación y la vecindad física dominan.
- El modelo original, un **ensamble con pesos locales aprendidos por gradiente exponenciado (EPL)**, supera con
  claridad al modelo base del E1 (0.62 frente a 0.43 de F1 en las mismas particiones; las cuatro pruebas lo confirman)
  y tiene el **mayor F1 medio** de los 11 modelos en la validación repetida 5 × 5 (0.618 frente a 0.605 del mejor
  benchmark), pero frente a los benchmarks la ventaja **no es estadísticamente concluyente**: solo una de tres pruebas la confirma y el tamaño del efecto es despreciable. El
  test y tres conjuntos de datos adicionales tampoco la confirman.
- Veredicto de la guía: **R1 sí, R2 sí, R3 no** (tope 4.0). Lo que aporta el modelo es combinar expertos con pesos
  aprendidos sobre predicciones fuera de muestra por bloques espaciales; el campo de pesos local no se traslada a zonas
  sin datos cerca, un límite que también describe la literatura reciente.

## Mapa del libro

| Capítulo | Qué responde |
|---|---|
| 1. Trazabilidad de correcciones | Qué se corrigió del Entregable 3 y dónde está la evidencia |
| 2. Datos: ETL y EDA | De dónde salen los datos y qué variables se asocian al estrato |
| 3. Modelos benchmark | Cuál es la mejor combinación de cada modelo y cómo se serializó |
| 4. Revisión de literatura y novedad | Si alguien ya usó un modelo así para el estrato y en qué se diferencia la propuesta |
| 5. Modelo original | Formulación, pseudocódigo, propiedades demostradas, código, pruebas y ablación |
| 6. Validación estadística | Si el modelo original supera a los benchmarks con significancia (R3) |
| 7. Dashboard y despliegue | Cómo está construido el dashboard y cómo se publica |
| 8. Conclusiones y limitaciones | Veredicto sobre R1, R2 y R3, limitaciones, trabajo futuro y declaración de uso de IA |
| 9. Reproducibilidad y bibliografía | Entorno, semillas, controles, tabla maestra y referencias |
