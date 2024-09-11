import core_parameters as params
from CompNeuroPy import load_variables
import numpy as np

fitness__array = []
for sim_id in range(params.INVESTIGATE_CORTEX_N_TOTAL):
    ### load results
    try:
        loaded_dict = load_variables(
            name_list=[f"deap_cma_result_{sim_id}"],
            path=params.INVESTIGATE_CORTEX_SAVE_FOLDER,
        )
    except FileNotFoundError:
        print(
            f"No optimization results found for sim_id {sim_id}, please run the optimization first"
        )
        continue
    ### get fitness
    deap_cma_result = loaded_dict[f"deap_cma_result_{sim_id}"]
    fitness = deap_cma_result["best_fitness"]
    fitness__array.append([sim_id, fitness])

### sort by fitness
fitness__array = np.array(fitness__array)
sorted_indices = np.argsort(fitness__array[:, 1])
fitness__array = fitness__array[sorted_indices]
### print best ten (max params.INVESTIGATE_CORTEX_N_TOTAL)
for i in range(min(10, len(fitness__array))):
    print(f"nr {round(fitness__array[i, 0])} with fitness {fitness__array[i, 1]}")

print(
    f"from {params.INVESTIGATE_CORTEX_SAVE_FOLDER}: nr {round(fitness__array[0, 0])} with fitness {fitness__array[0, 1]} is best"
)
