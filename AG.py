from AG_confs import *
from libs.selection import vectorized_tournament_selection
from libs.crossover import sbx_crossover_with_boundaries
from libs.mutation import polynomial_mutation_with_boundaries
from libs.auxiliaries_functions import initialize_population


# ---------------------------
# Función principal del GA
# ---------------------------
def genetic_algorithm(objective_func, lower_bound, upper_bound,
                      pop_size=POP_SIZE, num_generations=NUM_GENERATIONS,
                      tournament_size=TOURNAMENT_SIZE,
                      crossover_prob=CROSSOVER_PROB, eta_c=ETA_C,
                      mutation_prob=MUTATION_PROB, eta_mut=ETA_MUT):
    """
    Ejecuta el GA para la función objetivo dada y retorna:
      - best_solution, best_value
      - worst_solution, worst_value
      - avg_solution, avg_value
      - std_value (fitness)
      - best_fitness_history, best_x1_history, best_x2_history
      - population (final), fitness (final)
    """
    num_variables = len(lower_bound)
    
    # 1) Inicializar población
    population = initialize_population(pop_size, num_variables, lower_bound, upper_bound)
    fitness = np.array([objective_func(ind) for ind in population])
    
    best_fitness_history = []
    best_x1_history = []
    best_x2_history = []
    
    # Para animación: almacenamos el mejor (x1, x2) en cada generación
    best_solutions_over_time = np.zeros((num_generations, num_variables))
    
    for gen in range(num_generations):
        # Elitismo
        best_index = np.argmin(fitness)
        best_fitness = fitness[best_index]
        elite = population[best_index].copy()
        
        best_fitness_history.append(best_fitness)
        best_x1_history.append(elite[0])
        best_x2_history.append(elite[1])
        best_solutions_over_time[gen, :] = elite
        
        new_population = []
        
        # Cuántos padres se necesitan
        num_parents_needed = 2 * (pop_size - 1)
        winners, _ = vectorized_tournament_selection(fitness, num_parents_needed,
                                                     tournament_size, pop_size,
                                                     unique_in_column=True, unique_in_row=False)
        
        # Generar nueva población
        for i in range(0, len(winners), 2):
            parent1 = population[winners[i]].copy()
            if i + 1 < len(winners):
                parent2 = population[winners[i+1]].copy()
            else:
                parent2 = parent1.copy()
            
            child1, child2 = sbx_crossover_with_boundaries(
                parent1, parent2, lower_bound, upper_bound,
                eta_c, crossover_prob
            )
            child1 = polynomial_mutation_with_boundaries(
                child1, lower_bound, upper_bound,
                mutation_prob, eta_mut
            )
            child2 = polynomial_mutation_with_boundaries(
                child2, lower_bound, upper_bound,
                mutation_prob, eta_mut
            )
            
            new_population.append(child1)
            if len(new_population) < pop_size - 1:
                new_population.append(child2)
        
        # Convertir a array y evaluar
        new_population = np.array(new_population)
        new_fitness = np.array([objective_func(ind) for ind in new_population])
        
        # Agregar el élite
        new_population = np.vstack([new_population, elite])
        new_fitness = np.append(new_fitness, best_fitness)
        
        # Actualizar
        population = new_population.copy()
        fitness = new_fitness.copy()
    
    # Al final, mejor, peor, media, std
    best_index = np.argmin(fitness)
    worst_index = np.argmax(fitness)
    best_solution = population[best_index]
    best_value = fitness[best_index]
    worst_solution = population[worst_index]
    worst_value = fitness[worst_index]
    avg_solution = np.mean(population, axis=0)
    avg_value = np.mean(fitness)
    std_value = np.std(fitness)
    
    return (best_solution, best_value,
            worst_solution, worst_value,
            avg_solution, avg_value,
            std_value,
            best_fitness_history,
            best_x1_history,
            best_x2_history,
            population,
            fitness,
            best_solutions_over_time)