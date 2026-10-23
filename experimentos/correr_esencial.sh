#!/usr/bin/env bash
# Cadena recortada (lo esencial para la guía): versión 2 en los conjuntos extra, robustez mínima y complejidad.
# Espera la estadística de la versión 2; la complejidad espera además a que terminen los modelos finales.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY=${PYTHON:-../venv_e2/Scripts/python.exe}
LOG=resultados/logs
until [ -f resultados/cv5x5/estadistica_v2.json ]; do sleep 60; done
echo "=== INICIO esencial $(date)" >> $LOG/correr_5x5.txt
N_TRABAJOS=${N_EXT:-4} $PY -u -m experimentos.06b_externos_v2 >> $LOG/06b_externos_v2.txt 2>&1
echo "externos v2: código $? $(date)" >> $LOG/correr_5x5.txt
N_TRABAJOS=${N_ROB:-4} $PY -u -m experimentos.11_robustez >> $LOG/11_robustez.txt 2>&1
echo "robustez: código $? $(date)" >> $LOG/correr_5x5.txt
until grep -q "=== FIN finales" $LOG/correr_5x5.txt 2>/dev/null; do sleep 60; done
$PY -u -m experimentos.12_complejidad >> $LOG/12_complejidad.txt 2>&1
echo "complejidad: código $? $(date)" >> $LOG/correr_5x5.txt
echo "=== FIN esencial $(date)" >> $LOG/correr_5x5.txt
