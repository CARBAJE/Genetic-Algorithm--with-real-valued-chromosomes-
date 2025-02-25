import os
import pandas as pd
from AG_confs import *

from AG import genetic_algorithm
from libs.plot import *

def main():
    # Crear carpetas de salida generales
    os.makedirs("outputs", exist_ok=True)
    
    for func_key, func_data in FUNCTIONS.items():
        f_obj = func_data["func"]
        lb = func_data["lb"]
        ub = func_data["ub"]
        func_name = func_data["name"]
        num_runs = func_data["num_runs"]
        
        # Carpetas específicas de cada función
        func_folder = f"outputs/{func_key}"
        os.makedirs(func_folder, exist_ok=True)
        hist_folder = os.path.join(func_folder, "historiales")
        res_folder = os.path.join(func_folder, "resumenes")
        os.makedirs(hist_folder, exist_ok=True)
        os.makedirs(res_folder, exist_ok=True)
        
        print(f"\n==============================================")
        print(f"  FUNCIÓN: {func_name}")
        print(f"==============================================")
        
        all_runs_history = []
        best_solutions_all_runs = []
        
        for run in range(num_runs):
            print(f"\nEjecución {run+1}/{num_runs}")
            
            (best_sol, best_val,
             worst_sol, worst_val,
             avg_sol,  avg_val,
             std_val,
             best_fitness_history,
             best_x1_history,
             best_x2_history,
             population_final,
             fitness_final,
             best_solutions_over_time) = genetic_algorithm(
                 f_obj, lb, ub,
                 pop_size=POP_SIZE,
                 num_generations=NUM_GENERATIONS,
                 tournament_size=TOURNAMENT_SIZE,
                 crossover_prob=CROSSOVER_PROB,
                 eta_c=ETA_C,
                 mutation_prob=MUTATION_PROB,
                 eta_mut=ETA_MUT
             )
            
            # 1) Guardar historial
            df_historial = pd.DataFrame({
                "Generacion": np.arange(1, NUM_GENERATIONS + 1),
                "Mejor x1": best_x1_history,
                "Mejor x2": best_x2_history,
                "Mejor Fitness": best_fitness_history
            })
            historial_filename = os.path.join(hist_folder, f"historial_run_{run+1}.csv")
            df_historial.to_csv(historial_filename, index=False)
            
            # 2) Guardar resumen
            data_resumen = [
                ["Mejor", best_sol[0], best_sol[1], best_val],
                ["Media", avg_sol[0], avg_sol[1], avg_val],
                ["Peor", worst_sol[0], worst_sol[1], worst_val],
                ["Desv. estándar", np.nan, np.nan, std_val]
            ]
            df_resumen = pd.DataFrame(data_resumen, columns=["Indicador", "x1", "x2", "Fitness"])
            resumen_filename = os.path.join(res_folder, f"resumen_run_{run+1}.csv")
            df_resumen.to_csv(resumen_filename, index=False)
            
            print(df_resumen.to_string(index=False))
            
            all_runs_history.append(best_fitness_history)
            best_solutions_all_runs.append(best_sol)
        
        # Opcional: Graficar evolución del fitness de todas las corridas
        plot_evolucion_fitness(all_runs_history, func_key, func_name)
        
        # Opcional: Graficar superficie 3D (solo si es 2 variables)
        if len(lb) == 2:
            plot_surface_3d(f_obj, lb, ub, best_solutions_all_runs, func_key, func_name)

if __name__ == "__main__":
    # Llamamos a main() para correr el GA en todas las funciones definidas
    main()