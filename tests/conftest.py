# -*- coding: utf-8 -*-
"""Configuración común de las pruebas.

Fija un solo hilo en BLAS/OpenMP ANTES de que se importe numpy: con varios hilos, las sumas en coma flotante cambian
de orden y aparecen diferencias de redondeo (hasta 0.02 en algunas probabilidades del KNN, por empates de distancia
que se rompen distinto). Los artefactos se guardaron con un hilo, y la prueba de ida y vuelta exige igualdad exacta.
"""
import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
