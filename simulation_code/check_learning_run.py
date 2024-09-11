from CompNeuroPy import run_script_parallel, create_data_raw_folder
import core_parameters as params
import itertools

if __name__ == "__main__":
    if params.CHECK_LEARNING_SIMULATE:
        ### create the data folder
        create_data_raw_folder(
            folder_name=params.CHECK_LEARNING_SAVE_FOLDER,
            parameter_module=params,
        )

        ### depedning on sim_idx different paramter variants are simulated
        if params.CHECK_LEARNING_SIM_IDX == 0 or params.CHECK_LEARNING_SIM_IDX == "all":
            ### adjust paramters by myself in script check_learning.py
            idx_list = [(0, 0)]
            run_script_parallel(
                script_path="check_learning.py",
                n_jobs=params.N_JOBS,
                args_list=[
                    ["0", f"{alpha_idx}", f"{prob_idx}"]
                    for alpha_idx, prob_idx in idx_list
                ],
            )
        if params.CHECK_LEARNING_SIM_IDX == 1 or params.CHECK_LEARNING_SIM_IDX == "all":
            ### adjust parameters by sys.argv in script check_learning.py
            idx_list = list(
                itertools.product(
                    range(params.CHECK_LEARNING_ALPHA_STEPS),
                    range(params.CHECK_LEARNING_PROB_STEPS),
                )
            )
            run_script_parallel(
                script_path="check_learning.py",
                n_jobs=params.N_JOBS,
                args_list=[
                    ["1", f"{alpha_idx}", f"{prob_idx}"]
                    for alpha_idx, prob_idx in idx_list
                ],
            )

    if params.CHECK_LEARNING_ANALYZE:
        ### SIM_IDX_ANALYZE defines which simulations are analyzed (see different
        ### parameter variants above)
        if (
            params.CHECK_LEARNING_SIM_IDX_ANALYZE == 0
            or params.CHECK_LEARNING_SIM_IDX_ANALYZE == "all"
        ):
            run_script_parallel(
                script_path="check_learning_analyze.py",
                n_jobs=params.N_JOBS,
                args_list=[["0"]],
            )
        if (
            params.CHECK_LEARNING_SIM_IDX_ANALYZE == 1
            or params.CHECK_LEARNING_SIM_IDX_ANALYZE == "all"
        ):
            run_script_parallel(
                script_path="check_learning_analyze.py",
                n_jobs=params.N_JOBS,
                args_list=[["1"]],
            )
