#!/usr/bin/env bash
# Conjuntos de datos adicionales (3 × 3 × 5). Espera a que terminen la 5×5 y los métodos recientes.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY=${PYTHON:-../venv_e2/Scripts/python.exe}
LOG=resultados/logs
until grep -q "=== FIN recientes" $LOG/correr_5x5.txt 2>/dev/null; do sleep 60; done
echo "=== INICIO externos $(date)" >> $LOG/correr_5x5.txt
N_TRABAJOS=${N_EXT:-6} $PY -u -m experimentos.06_externos >> $LOG/06_externos.txt 2>&1
echo "externos: código $? $(date)" >> $LOG/correr_5x5.txt
echo "=== FIN externos $(date)" >> $LOG/correr_5x5.txt
