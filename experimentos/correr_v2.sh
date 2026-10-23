#!/usr/bin/env bash
# Versión 2 del modelo original: 5×5 (caché + búsqueda con el espacio de competencia), estadística de las dos
# versiones y, cuando terminen los rivales y la versión 1 en los conjuntos extra, la versión 2 en ellos.
# Reanudable: cada pliegue que ya tiene su archivo se salta.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY=${PYTHON:-../venv_e2/Scripts/python.exe}
LOG=resultados/logs
echo "=== INICIO v2 $(date)" >> $LOG/correr_5x5.txt
N_TRABAJOS=${N_V2:-3} $PY -u -m experimentos.03b_epl_v2_5x5 >> $LOG/03b_epl_v2.txt 2>&1
echo "epl v2: código $? $(date)" >> $LOG/correr_5x5.txt
$PY -u -m experimentos.04_estadistica v1 > $LOG/04_estadistica_v1.txt 2>&1
$PY -u -m experimentos.04_estadistica v2 > $LOG/04_estadistica_v2.txt 2>&1
echo "estadística: código $? $(date)" >> $LOG/correr_5x5.txt
until [ -f resultados/externos/resumen_externos.json ]; do sleep 60; done
N_TRABAJOS=${N_EXT:-6} $PY -u -m experimentos.06b_externos_v2 >> $LOG/06b_externos_v2.txt 2>&1
echo "externos v2: código $? $(date)" >> $LOG/correr_5x5.txt
echo "=== FIN v2 $(date)" >> $LOG/correr_5x5.txt
# robustez (después de todo lo anterior)
N_TRABAJOS=${N_ROB:-4} $PY -u -m experimentos.11_robustez >> $LOG/11_robustez.txt 2>&1
echo "robustez: código $? $(date)" >> $LOG/correr_5x5.txt
echo "=== FIN robustez $(date)" >> $LOG/correr_5x5.txt
# complejidad (al final, con el equipo libre)
$PY -u -m experimentos.12_complejidad >> $LOG/12_complejidad.txt 2>&1
echo "complejidad: código $? $(date)" >> $LOG/correr_5x5.txt
echo "=== FIN complejidad $(date)" >> $LOG/correr_5x5.txt
