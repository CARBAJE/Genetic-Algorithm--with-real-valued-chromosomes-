import numpy as np
# ---------------------------
# Definición de la función de Langermann
# ---------------------------
def langermann(x):
    """
    Función de Langermann.
    x: array-like, [x1, x2]
    """
    a = np.array([3, 5, 2, 1, 7])
    b = np.array([5, 2, 1, 4, 9])
    c = np.array([1, 2, 5, 2, 3])
    x1, x2 = x
    valor = 0.0
    for i in range(len(a)):
        dist_cuadrada = (x1 - a[i])**2 + (x2 - b[i])**2
        valor += c[i] * np.cos(np.pi * dist_cuadrada) * np.exp(-dist_cuadrada / np.pi)
    return -valor  # Se minimiza, por eso el signo negativo.

def drop_wave(x):
    """
    Función Drop-Wave.
    x: array-like, [x1, x2]
    
    f(x, y) = (1 + cos(12 * sqrt(x^2 + y^2))) / (0.5*(x^2 + y^2) + 2)
    
    *Nota: A veces se ve con un signo negativo adelante. Aquí usamos
    la versión positiva tal como se muestra en tu imagen.
    """
    x1, x2 = x
    r = np.sqrt(x1**2 + x2**2)
    return - (1 + np.cos(12*r)) / (0.5*r**2 + 2)