import numpy as np

def sbx_crossover(parent1, parent2, lower_bound, upper_bound, eta, crossover_prob):
    """Realiza el cruzamiento SBX para dos padres y devuelve dos hijos."""
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)
    
    if np.random.rand() <= crossover_prob:
        for i in range(len(parent1)):
            u = np.random.rand()
            if u <= 0.5:
                beta = (2*u)**(1/(eta+1))
            else:
                beta = (1/(2*(1-u)))**(1/(eta+1))
            
            # Genera los dos hijos
            child1[i] = 0.5*((1+beta)*parent1[i] + (1-beta)*parent2[i])
            child2[i] = 0.5*((1-beta)*parent1[i] + (1+beta)*parent2[i])
            
            # Asegurar que los hijos estén dentro de los límites
            child1[i] = np.clip(child1[i], lower_bound[i], upper_bound[i])
            child2[i] = np.clip(child2[i], lower_bound[i], upper_bound[i])
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()
    
    return child1, child2

def sbx_crossover_with_boundaries(parent1, parent2, lower_bound, upper_bound,
                                  eta, crossover_prob, use_global_u=False):
    """
    Realiza el cruzamiento SBX con límites, usando las fórmulas que ajustan beta
    en función de la cercanía a las fronteras. Permite usar un único 'u' para todos
    los genes o uno por cada gen.
    
    Parámetros
    ----------
    parent1, parent2 : array-like
        Padres a cruzar (misma dimensión).
    lower_bound, upper_bound : array-like
        Límites inferiores y superiores para cada variable.
    eta : float
        Índice de distribución (eta) para SBX.
    crossover_prob : float
        Probabilidad de realizar el crossover (entre 0 y 1).
    use_global_u : bool
        Si True, se usa el mismo valor de 'u' para todas las variables en un cruce.
        Si False, se genera un 'u' distinto por cada variable (comportamiento estándar).

    Retorna
    -------
    child1, child2 : arrays
        Hijos resultantes del cruzamiento SBX con límites.
    """
    parent1 = np.asarray(parent1)
    parent2 = np.asarray(parent2)
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)
    
    # Se decide si hay crossover según crossover_prob
    if np.random.rand() > crossover_prob:
        # No hay cruce, los hijos son copias de los padres
        return parent1.copy(), parent2.copy()
    
    # Si se quiere un solo 'u' para todo el vector
    if use_global_u:
        u_global = np.random.rand()
    
    for i in range(len(parent1)):
        x1 = parent1[i]
        x2 = parent2[i]
        lb = lower_bound[i]
        ub = upper_bound[i]
        
        # Aseguramos x1 <= x2
        if x1 > x2:
            x1, x2 = x2, x1
        
        dist = x2 - x1
        if dist < 1e-14:
            # Los padres son prácticamente iguales
            child1[i] = x1
            child2[i] = x2
            continue
        
        # =======================
        # Cálculo de beta y alpha
        # =======================
        # min_dist es la mínima distancia a la frontera desde x1 o x2
        min_val = min(x1 - lb, ub - x2)
        
        # Evitar valores negativos si, por error, x1 < lb o x2 > ub
        # (asumimos que los padres ya están dentro de [lb, ub])
        if min_val < 0:
            min_val = 0
        
        beta = 1.0 + (2.0 * min_val / dist)
        alpha = 2.0 - beta**(-(eta+1))
        
        # =======================
        # Cálculo de u y beta_q
        # =======================
        if use_global_u:
            u = u_global
        else:
            u = np.random.rand()
        
        if u <= (1.0 / alpha):
            betaq = (alpha * u)**(1.0/(eta+1))
        else:
            betaq = (1.0 / (2.0 - alpha*u))**(1.0/(eta+1))
        
        # =======================
        # Cálculo de los hijos
        # =======================
        c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
        c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
        
        # Ajustar a [lb, ub]
        c1 = np.clip(c1, lb, ub)
        c2 = np.clip(c2, lb, ub)
        
        # Ubicar en child1, child2 respetando si originalmente x1>x2
        # (aunque arriba ya reordenamos x1<=x2, pero si prefieres
        #  respetar el orden original, puedes hacer un if).
        child1[i] = c1
        child2[i] = c2
    
    return child1, child2