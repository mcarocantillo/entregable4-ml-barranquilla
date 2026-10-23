#!/usr/bin/env bash
# Validación anidada repetida 5×5: particiones → benchmarks → modelo original. Reanudable (salta lo ya hecho).
# Uso (desde cualquier carpeta):  bash experimentos/correr_5x5.sh
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY=${PYTHON:-../venv_e2/Scripts/python.exe}
LOG=resultados/logs
echo "=== INICIO $(date)" >> $LOG/correr_5x5.txt
$PY -u -m experimentos.01_particiones >> $LOG/01_particiones.txt 2>&1
N_TRABAJOS=${N_BENCH:-6} $PY -u -m experimentos.02_benchmarks_5x5 >> $LOG/02_benchmarks.txt 2>&1
echo "benchmarks: código $? $(date)" >> $LOG/correr_5x5.txt
N_TRABAJOS=${N_EPL:-4} $PY -u -m experimentos.03_epl_5x5 >> $LOG/03_epl.txt 2>&1
echo "epl: código $? $(date)" >> $LOG/correr_5x5.txt
echo "=== FIN $(date)" >> $LOG/correr_5x5.txt
