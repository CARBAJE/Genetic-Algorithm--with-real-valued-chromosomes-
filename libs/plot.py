import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # para gráficos 3D
from matplotlib.animation import FuncAnimation

def plot_evolucion_fitness(all_runs_history, func_key, func_name):
    """Grafica la evolución del fitness (original y normalizado) de varias corridas."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot Izquierdo: Curvas originales
    for idx, history in enumerate(all_runs_history):
        axs[0].plot(history, label=f'Ejecución {idx+1}')
    axs[0].set_xlabel('Generación')
    axs[0].set_ylabel('Mejor Fitness')
    axs[0].set_title(f'Evolución del Fitness (Original) - {func_name}')
    axs[0].legend()
    axs[0].grid(True)
    
    # Subplot Derecho: Curvas normalizadas
    for idx, history in enumerate(all_runs_history):
        h = np.array(history)
        h_min, h_max = h.min(), h.max()
        if h_max == h_min:
            norm_history = np.zeros_like(h)
        else:
            norm_history = (h - h_min) / (h_max - h_min)
        axs[1].plot(norm_history, label=f'Ejecución {idx+1}')
    axs[1].set_xlabel('Generación')
    axs[1].set_ylabel('Fitness Normalizado')
    axs[1].set_title(f'Evolución del Fitness (Normalizado) - {func_name}')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"outputs/{func_key}/evolucion_fitness_{func_key}.png")
    plt.show()

def plot_surface_3d(objective_func, lower_bound, upper_bound, best_solutions_list, func_key, func_name):
    """Grafica la superficie 3D de la función y su proyección en el plano XY."""
    num_points = 100
    x_vals = np.linspace(lower_bound[0], upper_bound[0], num_points)
    y_vals = np.linspace(lower_bound[1], upper_bound[1], num_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_func([X[i, j], Y[i, j]])
    
    colors = [plt.cm.jet(i/len(best_solutions_list)) for i in range(len(best_solutions_list))]
    
    fig = plt.figure(figsize=(12, 6))
    
    # Subplot 1: Vista 3D
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
    ax3d.set_title(f'Superficie 3D - {func_name}')
    ax3d.set_xlabel('x1')
    ax3d.set_ylabel('x2')
    ax3d.set_zlabel('Fitness')
    
    for idx, sol in enumerate(best_solutions_list):
        bx, by = sol
        f_val = objective_func([bx, by])
        col = colors[idx]
        ax3d.scatter(bx, by, f_val, color=col, s=100, marker='o')
    ax3d.view_init(elev=30, azim=30)
    
    # Subplot 2: XY
    ax_xy = fig.add_subplot(1, 2, 2)
    cont = ax_xy.contourf(X, Y, Z, cmap='viridis')
    fig.colorbar(cont, ax=ax_xy)
    for idx, sol in enumerate(best_solutions_list):
        bx, by = sol
        col = colors[idx]
        ax_xy.scatter(bx, by, color=col, s=100, marker='o', label=f'Run {idx+1}')
    ax_xy.set_title('Proyección XY')
    ax_xy.set_xlabel('x1')
    ax_xy.set_ylabel('x2')
    ax_xy.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/{func_key}/surface_3d_{func_key}.png")
    plt.show()