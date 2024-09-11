from CompNeuroPy import run_script_parallel, create_data_raw_folder
import core_parameters as params

if __name__ == "__main__":
    if params.INVESTIGATE_CORTEX_OPTIMIZE:
        ### create the data folder
        create_data_raw_folder(
            fodler_name=params.INVESTIGATE_CORTEX_SAVE_FOLDER,
            parameter_module=params,
        )

    run_script_parallel(
        script_path="investigate_cortex.py",
        n_jobs=params.N_JOBS,
        args_list=[[f"{i}"] for i in range(params.INVESTIGATE_CORTEX_N_TOTAL)],
    )

"""
from investigate_cortex_data: nr 3 is best

### investigate_cortex_data_1:
from investigate_cortex_data_1: nr 206 with fitness 0.31678225963851814 is best
top ten:
nr 206 with fitness 0.31678225963851814
nr 68 with fitness 0.3244246734541159
nr 7 with fitness 0.3255557334104028
nr 37 with fitness 0.3259212561562988
nr 227 with fitness 0.3273978720555868
nr 17 with fitness 0.3290469714298922
nr 6 with fitness 0.33333956683671534
nr 54 with fitness 0.33335881621582947
nr 168 with fitness 0.3334280242416035
nr 331 with fitness 0.3649067980063581
worked not well because duration is at max loss everywhere

### investigate_cortex_data_2:
top ten:
nr 269 with fitness 0.03582383443574755
nr 88 with fitness 0.03621481312873276
nr 329 with fitness 0.03633396927118371
nr 237 with fitness 0.036408968606879366
nr 40 with fitness 0.036600840484207676
nr 287 with fitness 0.036682560968213984
nr 192 with fitness 0.03677837077287097
nr 333 with fitness 0.036894537701098296
nr 59 with fitness 0.03699582699315799
nr 42 with fitness 0.037064861407775464
all look good, took 59 due to small sigma in half-gaussian connections (less synapses)
"""
