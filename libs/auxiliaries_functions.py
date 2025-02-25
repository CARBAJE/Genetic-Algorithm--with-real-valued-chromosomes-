import numpy as np
# ---------------------------
# Funciones auxiliares del GA
# ---------------------------
def initialize_population(pop_size, num_variables, lower_bound, upper_bound):
    """Inicializa la población uniformemente en el espacio de búsqueda."""
    return np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, num_variables))
