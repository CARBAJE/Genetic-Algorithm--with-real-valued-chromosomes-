import numpy as np
from libs.functions import langermann, drop_wave
# ---------------------------
# Parámetros del algoritmo
# ---------------------------
POP_SIZE = 100            # Número de individuos en la población
NUM_GENERATIONS = 200     # Número de generaciones
NUM_RUNS = 10             # Número de ejecuciones completas (ciclos)

# Parámetros de la función de Langermann
a = np.array([3, 5, 2, 1, 7])
b = np.array([5, 2, 1, 4, 9])
c = np.array([1, 2, 5, 2, 3])

# Parámetros del torneo
TOURNAMENT_SIZE = 3  # Número de individuos participantes en cada torneo

# Parámetros del cruzamiento SBX
CROSSOVER_PROB = 0.9  # Probabilidad de aplicar cruzamiento
ETA_C = 15            # Índice de distribución para SBX

# Parámetros de la mutación polinomial
MUTATION_PROB = 1.0 / 2  # Probabilidad de mutar cada gen
ETA_MUT = 20                         # Índice de distribución para mutación polinomial

best_solutions_list = [] 
all_runs_history = []  # Para graficar luego

FUNCTIONS = {
    "langermann": {
        "func": langermann,
        "lb": np.array([0, 0]),
        "ub": np.array([10, 10]),
        "name": "Langermann",
        "num_runs": NUM_RUNS
    },
    "drop_wave": {
        "func": drop_wave,
        "lb": np.array([-5.12, -5.12]),
        "ub": np.array([5.12, 5.12]),
        "name": "Drop-Wave",
        "num_runs": NUM_RUNS
    }
}