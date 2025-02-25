# Genetic Algorithm for Function Minimization with Real Valued Chromosomes

This project implements a Genetic Algorithm (GA) to minimize benchmark functions. Two objective functions are provided by default: **Langermann** and **Drop-Wave**. The GA uses a combination of tournament selection, Simulated Binary Crossover (SBX) with boundary handling, and polynomial mutation with boundaries to explore the search space.

## Overview

The script is organized into several components:
- **Objective Functions:**  
  - **Langermann Function:** A multimodal function defined with cosine and exponential components.
  - **Drop-Wave Function:** A 2-dimensional function with a wave-like surface.
- **Genetic Algorithm Components:**  
  - **Population Initialization:** Generates an initial population uniformly in the search space.
  - **Selection:** Uses vectorized tournament selection to choose parents based on fitness.
  - **Crossover:** Implements SBX (Simulated Binary Crossover) with boundary control.
  - **Mutation:** Applies polynomial mutation with boundary control.
  - **Elitism:** Keeps the best solution from each generation.
- **Output and Visualization:**  
  - Saves run histories and summary results (best, worst, average, and standard deviation) into CSV files.
  - Generates plots for the evolution of fitness over generations (both original and normalized).
  - If the objective is 2-dimensional, creates a 3D surface plot and its 2D projection with the best solutions marked.

## Project Structure

```
├── AG_confs.py                # Configuration parameters (e.g., population size, generations, etc.)
├── AG.py                      # Main genetic algorithm function implementation
├── libs
│   ├── auxiliaries_functions.py   # Helper functions (e.g., population initialization)
│   ├── crossover.py               # Crossover (SBX) functions
│   ├── mutation.py                # Mutation functions (polynomial mutation)
│   └── selection.py               # Tournament selection function
├── outputs                    # Folder where run histories, summaries, and plots are saved (auto-created)
└── main_script.py             # Main entry point that executes the GA for each defined function
```

*Note:* The provided code is self-contained in the script below, but it references modularized functions (e.g., from `libs/` and `AG_confs`). Make sure these modules are in place or adjust the imports as necessary.

## Prerequisites

- **Python 3.6+**  
- **Required Python packages:**
  - `numpy`
  - `matplotlib`
  - `pandas`

Install the required packages using pip:

```bash
pip install numpy matplotlib pandas
```

## Usage

1. **Configure Parameters:**  
   Modify the configuration parameters in `AG_confs.py` or within the script (population size, number of generations, tournament size, crossover and mutation probabilities, etc.).

2. **Add/Modify Objective Functions:**  
   The `FUNCTIONS` dictionary in the script defines the functions to optimize. Each function entry includes:
   - `func`: The objective function.
   - `lb` and `ub`: The lower and upper bounds for the search space.
   - `name`: A display name.
   - `num_runs`: Number of independent runs to perform.
   
   You can add more functions as needed.

3. **Run the Script:**  
   Execute the main script using:

   ```bash
   python main_script.py
   ```

4. **Results:**  
   - CSV files with run histories are saved under `outputs/<function_key>/historiales/`.
   - Summary CSV files (best, worst, average, and standard deviation) are saved under `outputs/<function_key>/resumenes/`.
   - Plots for fitness evolution and, for 2D functions, a 3D surface plot are saved in the corresponding `outputs/<function_key>/` folder.
   - The script also displays the plots after saving them.

## Genetic Algorithm Workflow

1. **Initialization:**  
   A population is generated uniformly within the specified bounds.

2. **Evaluation:**  
   The fitness of each individual is computed using the objective function (minimization is achieved by returning the negative of the evaluated value where needed).

3. **Selection:**  
   The tournament selection method chooses parents based on their fitness.

4. **Crossover (SBX):**  
   Parents undergo SBX crossover with boundary corrections to generate offspring.

5. **Mutation:**  
   Offspring are mutated using a polynomial mutation operator that respects the variable boundaries.

6. **Elitism:**  
   The best solution of each generation is retained to ensure convergence.

7. **Iteration:**  
   The process repeats for a set number of generations, recording the best fitness per generation.

8. **Post-processing:**  
   After all generations, the best, worst, average, and standard deviation of fitness are computed and stored. Visualization functions generate fitness evolution and 3D surface plots.

## Customization

- **Changing GA Parameters:**  
  Adjust parameters such as population size, number of generations, crossover probability, etc., in the configuration section.
  
- **Adding New Objective Functions:**  
  Extend the `FUNCTIONS` dictionary with your custom function, bounds, and other parameters.

## Output Files and Visualization

- **Historiales:**  
  Contains CSV files with the evolution of the best fitness and the best variables (`x1` and `x2`) per generation for each run.

- **Resúmenes:**  
  Contains CSV files summarizing the best, average, and worst solutions along with fitness values for each run.

- **Plots:**  
  - *Fitness Evolution:* Two subplots show the original and normalized fitness values over generations.
  - *3D Surface Plot:* Visualizes the objective function surface along with the best solutions found over different runs (only for 2D functions).

## License

*Include license information here if applicable.*

## Acknowledgments
> This project was inspired by the lecture on Selected Topics of Bioinspired Algorithms provided by Professor Daniel Molina Pérez at Escuela Superior de Cómputo - Instituto Politécnico Nacional. I also greatly benefited from the methodologies described in *Analysis and Enhancement of Simulated Binary Crossover* by J. Chacón and C. Segura, which helped shape the implementation of the Genetic Algorithm, including techniques like tournament selection, SBX crossover, and polynomial mutation.

