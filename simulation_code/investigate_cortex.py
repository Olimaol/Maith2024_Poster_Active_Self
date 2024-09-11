from core_model import cortex_model
from ANNarchy import simulate, get_population
from CompNeuroPy import (
    CompNeuroMonitors,
    PlotRecordings,
    cnp_clear,
    DeapCma,
    save_variables,
    load_variables,
)
import matplotlib.pyplot as plt
import numpy as np
import sys
import core_parameters as params
from core_helping_functions import r_squared

"""
script for simulating params.INVESTIGATE_CORTEX_T_TOTAL(1700) ms only with cortex

maximum state storing duration is 1500 ms in experiment, therefore find cortex
parameters that allow for storing a state for 1700 ms
"""


def error_function(target_value, actual_value):
    """
    The error is between 0 (minimum) and 1 (maximum). It's defined by a gaussian
    function with a standard deviation of target_value/10. The error is 0 when the
    actual value is equal to the target value.
    """
    return 1 - np.exp(-0.5 * ((actual_value - target_value) / (target_value / 10)) ** 2)


def run_cortex_get_error(
    cor__cor__sequence_prop_exc,
    cor__cor__sequence_prop_forward_inh,
    cor__cor__sequence_prop_backward_inh,
    cor__cor__self_exc,
    cor_cor_sigma_exc,
    cor_cor_sigma_forward_inh,
    cor_cor_sigma_backward_inh,
    cor_input_amplitude,
    do_plot=False,
):
    """
    clear ANNarchy, create the cortex model with given parameters, simulate and get the
    error for the given parameters
    """
    cnp_clear()
    ### set the parameters for inner cortex projections
    cortex_model.model_kwargs = {
        "weights": {
            "cor__cor__sequence_prop_exc": cor__cor__sequence_prop_exc,
            "cor__cor__sequence_prop_forward_inh": cor__cor__sequence_prop_forward_inh,
            "cor__cor__sequence_prop_backward_inh": cor__cor__sequence_prop_backward_inh,
            "cor__cor__self_exc": cor__cor__self_exc,
            "cor__cor__sequence_end": 0,
        },
        "n_state": 1,
        "n_cor_sequence": 50,
        "cor_sequence_delays": 20,
        "cor_cor_sigma_exc": cor_cor_sigma_exc,
        "cor_cor_sigma_forward_inh": cor_cor_sigma_forward_inh,
        "cor_cor_sigma_backward_inh": cor_cor_sigma_backward_inh,
    }
    cortex_model.create(
        warn=False,
        compile_folder_name=f"{cortex_model.compile_folder_name}_{sys.argv[1]}",
    )

    ### create monitors
    mon_dict = {"cor": ["r"]}
    monitors = CompNeuroMonitors(mon_dict=mon_dict)

    ### simulate initial 500 ms before the params.INVESTIGATE_CORTEX_T_TOTAL ms
    simulate(500)

    ### reset time with monitors
    monitors.reset(populations=False, projections=False, synapses=False)

    ### record the activity of the cortex
    monitors.start()

    ### simulate the params.INVESTIGATE_CORTEX_T_TOTAL ms with cortex input at the beginning
    get_population("cor")[0, 0].I_ext = cor_input_amplitude
    simulate(params.SIMULATION_T_INPUT_COR)
    get_population("cor")[0, 0].I_ext = 0
    simulate(params.INVESTIGATE_CORTEX_T_TOTAL - params.SIMULATION_T_INPUT_COR)

    ### get the recordings
    recordings = monitors.get_recordings()
    recording_times = monitors.get_recording_times()

    ### plot the recordings
    if do_plot:
        PlotRecordings(
            figname=f"investigate_cortex_r_matrix_{sys.argv[1]}.png",
            recordings=recordings,
            recording_times=recording_times,
            shape=(1, 1),
            plan={
                "position": [1],
                "variable": ["r"],
                "compartment": ["cor"],
                "format": ["matrix"],
            },
        )

    ### define how the cortex should be and calculate error
    ### first a linear function with y=neuron id and x=time of maximum activity
    y = list(range(cortex_model.model_kwargs["n_cor_sequence"]))
    x = recordings[0]["cor;r"].argmax(axis=0)
    ### define target linear function
    target_duration = params.SIMULATION_T_INPUT_COR
    target_slope = 50 / (params.INVESTIGATE_CORTEX_T_TOTAL - (target_duration / 2))
    target_intercept = -target_slope * (target_duration / 2)
    ### calculate error for linear fit, how good does the target linear function fit
    ### the data
    error_linear_fit = 1 - r_squared(
        slope=target_slope, intercept=target_intercept, x=x, y=y
    )
    ### plot the linear regression
    if do_plot:
        plt.figure()
        plt.subplot(211)
        plt.plot(x, y, "o", label="data")
        plt.plot(
            np.linspace(0, params.INVESTIGATE_CORTEX_T_TOTAL, 1000),
            target_slope * np.linspace(0, params.INVESTIGATE_CORTEX_T_TOTAL, 1000)
            + target_intercept,
            label="target",
        )
        plt.title(f"y_target = {target_slope} * x + {target_intercept}")
        plt.xlabel("time")
        plt.ylabel("neuron id")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.xlim(0, params.INVESTIGATE_CORTEX_T_TOTAL)
        plt.ylim(50, 0)
        plt.subplot(212)
        plt.plot(error_function(5, np.linspace(0, 10, 1000)))
        plt.savefig(f"investigate_cortex_linear_regression_{sys.argv[1]}.png")

    ### second the average duration should be the same as the input duration
    r_arr = recordings[0]["cor;r"]
    ### for each neuron (second idx) get how long the activity is above 0.05
    ### if there are multiple separated regions of activity, only the last one is considered
    ### and the loss is increased by 1 (add_loss=1), the target is only one region of activity
    duration = np.zeros(50)
    add_loss = 0
    for i in range(50):
        idx = np.where(r_arr[:, i] > 0.05)[0]
        idx_split = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        if len(idx_split) > 1:
            add_loss = 1
        duration[i] = len(idx_split[-1]) if len(idx_split) > 0 else 0
    ### calculate the target duration for each neuron with the target linear function)
    x_plus = (np.arange(50) + target_slope * target_duration) * (1 / target_slope)
    x_minus = np.arange(50) * (1 / target_slope)
    target_duration = np.clip(x_plus, 0, params.INVESTIGATE_CORTEX_T_TOTAL) - np.clip(
        x_minus, 0, params.INVESTIGATE_CORTEX_T_TOTAL
    )
    ### calculate the error for the duration
    error_duration = np.mean(error_function(target_duration, duration)) + add_loss

    ### third the peak acitivty should be the input amplitude for each neuron
    peak_activity = np.max(r_arr, axis=0)
    target_peak_activity = cor_input_amplitude
    error_peak_activity = np.mean(error_function(target_peak_activity, peak_activity))

    if not params.INVESTIGATE_CORTEX_OPTIMIZE:
        print(
            f"error_linear_fit: {error_linear_fit}, error_duration: {error_duration}, error_peak_activity: {error_peak_activity}"
        )
        print(
            f"error_total: {np.mean(np.exp([error_linear_fit, error_duration, error_peak_activity])-1)}"
        )

    return np.mean(np.exp([error_linear_fit, error_duration, error_peak_activity]) - 1)


def evaluate_function(population):
    """
    evaluation funciton for DeapCma
    """
    loss_list = []
    ### the population is a list of individuals which are lists of parameters
    for individual in population:
        loss_of_individual = run_cortex_get_error(
            cor__cor__sequence_prop_exc=individual[0],
            cor__cor__sequence_prop_forward_inh=individual[1],
            cor__cor__sequence_prop_backward_inh=individual[2],
            cor__cor__self_exc=individual[3],
            cor_cor_sigma_exc=max([individual[4], 0.1]),
            cor_cor_sigma_forward_inh=max([individual[5], 0.1]),
            cor_cor_sigma_backward_inh=max([individual[6], 0.1]),
            cor_input_amplitude=1,
        )
        loss_list.append((loss_of_individual,))
    return loss_list


def perform_deap_optimization():

    ### define lower bounds, upper bounds and initial guess for the parameters
    lb = np.array([0, 0, 0, 0, 0.5, 0.5, 0.5])
    ub = np.array([2, 2, 2, 2, 4, 4, 4])
    p0 = np.array(
        [
            1.7820838305856164,
            1.3309095246114215,
            3.8605488632608784,
            3.5975913243754496,
            0.1,
            1.9556911628242564,
            3.874300469457968,
        ]
    )

    ### create an "minimal" instance of the DeapCma class
    deap_cma = DeapCma(
        lower=lb,
        upper=ub,
        evaluate_function=evaluate_function,
        max_evals=params.INVESTIGATE_CORTEX_MAX_EVALS,
        p0=p0,
        plot_file=f"{params.INVESTIGATE_CORTEX_SAVE_FOLDER}/logbook_{sys.argv[1]}.png",
    )

    ### run the optimization
    deap_cma_result = deap_cma.run()

    ### save the result
    save_variables(
        variable_list=[deap_cma_result],
        name_list=[f"deap_cma_result_{sys.argv[1]}"],
        path=params.INVESTIGATE_CORTEX_SAVE_FOLDER,
    )


if __name__ == "__main__":
    if params.INVESTIGATE_CORTEX_OPTIMIZE:
        perform_deap_optimization()
    else:
        try:
            loaded_dict = load_variables(
                name_list=[f"deap_cma_result_{sys.argv[1]}"],
                path=params.INVESTIGATE_CORTEX_SAVE_FOLDER,
            )
        except FileNotFoundError:
            print("No optimization results found, please run the optimization first")
            quit()

        ### print fitness and params
        deap_cma_result = loaded_dict[f"deap_cma_result_{sys.argv[1]}"]
        print(f"deap_cma_result_{sys.argv[1]}: {deap_cma_result['best_fitness']}")
        for i in range(7):
            print(f"param{i}: {deap_cma_result['param'+str(i)]}")

        ### run the model with the best parameters
        run_cortex_get_error(
            cor__cor__sequence_prop_exc=deap_cma_result["param0"],
            cor__cor__sequence_prop_forward_inh=deap_cma_result["param1"],
            cor__cor__sequence_prop_backward_inh=deap_cma_result["param2"],
            cor__cor__self_exc=deap_cma_result["param3"],
            cor_cor_sigma_exc=max([deap_cma_result["param4"], 0.1]),
            cor_cor_sigma_forward_inh=max([deap_cma_result["param5"], 0.1]),
            cor_cor_sigma_backward_inh=max([deap_cma_result["param6"], 0.1]),
            cor_input_amplitude=1,
            do_plot=True,
        )
