#!/usr/bin/env bash
# LightGBM y CatBoost en la 5×5. Espera a que termine correr_5x5.sh (para no competir por los núcleos).
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY=${PYTHON:-../venv_e2/Scripts/python.exe}
LOG=resultados/logs
until grep -q "=== FIN" $LOG/correr_5x5.txt 2>/dev/null; do sleep 60; done
echo "=== INICIO recientes $(date)" >> $LOG/correr_5x5.txt
N_TRABAJOS=${N_BENCH:-6} $PY -u -m experimentos.02b_recientes_5x5 >> $LOG/02b_recientes.txt 2>&1
echo "recientes: código $? $(date)" >> $LOG/correr_5x5.txt
echo "=== FIN recientes $(date)" >> $LOG/correr_5x5.txt
