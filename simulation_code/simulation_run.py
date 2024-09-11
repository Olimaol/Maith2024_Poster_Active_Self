from CompNeuroPy import run_script_parallel, create_data_raw_folder
import core_parameters as params


if __name__ == "__main__":
    if params.SIMULATION_SIMULATE:
        ### create the data folder
        create_data_raw_folder(
            folder_name=params.SIMULATION_SAVE_FOLDER,
            parameter_module=params,
        )
        if params.SIMULATION_OPTIMIZE:
            ### run the simulation
            run_script_parallel(
                script_path="simulation.py",
                n_jobs=params.N_JOBS,
                args_list=[[f"{i}"] for i in range(params.SIMULATION_N_TOTAL)],
            )
        else:
            ### run the simulation
            run_script_parallel(
                script_path="simulation.py",
                n_jobs=params.N_JOBS,
                args_list=[["0"]],
            )
    if params.SIMULATION_ANALYZE:
        if params.SIMULATION_ID_ANALYZE == "all":
            ### run the analysis
            run_script_parallel(
                script_path="simulation_analyze.py",
                n_jobs=params.N_JOBS,
                args_list=[[f"{i}"] for i in range(params.SIMULATION_N_TOTAL)],
            )
        else:
            ### run the analysis
            run_script_parallel(
                script_path="simulation_analyze.py",
                n_jobs=params.N_JOBS,
                args_list=[[f"{params.SIMULATION_ID_ANALYZE}"]],
            )
