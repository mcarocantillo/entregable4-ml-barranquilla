#!/usr/bin/env bash
# Modelos finales, test confirmatorio y datos de la página Modelos. Espera la estadística de la versión 2.
# El modelo original final es la versión 2 (fijado antes de ver sus resultados externos).
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY=${PYTHON:-../venv_e2/Scripts/python.exe}
LOG=resultados/logs
until [ -f resultados/cv5x5/estadistica_v2.json ]; do sleep 60; done
echo "=== INICIO finales $(date)" >> $LOG/correr_5x5.txt
$PY -u -m experimentos.07_artefactos v2 >> $LOG/07_artefactos.txt 2>&1
echo "artefactos: código $? $(date)" >> $LOG/correr_5x5.txt
$PY -u -m experimentos.09_test_confirmatorio >> $LOG/09_test.txt 2>&1
echo "test: código $? $(date)" >> $LOG/correr_5x5.txt
$PY -u -m experimentos.10_datos_modelos_dashboard >> $LOG/10_dashboard.txt 2>&1
echo "dashboard: código $? $(date)" >> $LOG/correr_5x5.txt
echo "=== FIN finales $(date)" >> $LOG/correr_5x5.txt
