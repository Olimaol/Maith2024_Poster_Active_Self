import core_parameters as params
from CompNeuroPy import PlotRecordings, load_variables, create_dir, print_df
from CompNeuroPy.monitors import RecordingTimes
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.signal import savgol_filter

"""
SIM_IDX_ANALYZE value defines which simulations are analyzed, see check_learning_run.py
"""


def plot_single_trial(
    recordings, recording_times: RecordingTimes, population_name_list, chunk=0
):
    ### plot activities/rates of all populations
    PlotRecordings(
        figname=f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/single_trial_r_line.png",
        recordings=recordings,
        recording_times=recording_times,
        shape=(2, 3),
        plan={
            "position": [idx + 1 for idx in range(len(population_name_list))],
            "variable": ["r" for _ in population_name_list],
            "compartment": [pop for pop in population_name_list],
            "format": ["line" for _ in population_name_list],
        },
        chunk=chunk,
    )
    PlotRecordings(
        figname=f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/single_trial_r_matrix.png",
        recordings=recordings,
        recording_times=recording_times,
        shape=(2, 3),
        plan={
            "position": [idx + 1 for idx in range(len(population_name_list))],
            "variable": ["r" for _ in population_name_list],
            "compartment": [pop for pop in population_name_list],
            "format": ["matrix" for _ in population_name_list],
        },
        chunk=chunk,
    )
    ### plot variables relevant for cor-stn learning
    PlotRecordings(
        figname=f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/single_trial_learning_cor_stn.png",
        recordings=recordings,
        recording_times=recording_times,
        shape=(5, 1),
        plan={
            "position": [1, 2, 3, 4, 5],
            "variable": [
                "r",
                "learn_mod",
                "r_trace_increase_event",
                "mod_increase",
                "w",
            ],
            "compartment": ["cor", "stn", "cor", "stn", "cor__stn"],
            "format": ["matrix"] * 5,
        },
        chunk=chunk,
    )
    ### plot variables relevant for stn-snr learning
    PlotRecordings(
        figname=f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/single_trial_learning_stn_snr.png",
        recordings=recordings,
        recording_times=recording_times,
        shape=(6, 1),
        plan={
            "position": [1, 2, 3, 4, 5, 6],
            "variable": [
                "r",
                "learn_mod",
                "r_trace_increase_event",
                "r_trace_increase_event_low",
                "mod_increase",
                "w",
            ],
            "compartment": ["stn", "snr", "stn", "stn", "snr", "stn__snr"],
            "format": ["matrix"] * 6,
        },
        chunk=chunk,
    )
    ### plot all input variables and the rate of stn, snr and pf
    PlotRecordings(
        figname=f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/single_trial_inputs_stn_snr_pf.png",
        recordings=recordings,
        recording_times=recording_times,
        shape=(5, 3),
        plan={
            "position": list(range(1, 16)),
            "variable": ["r" for _ in range(3)]
            + ["I_signal" for _ in range(3)]
            + ["I_noise" for _ in range(3)]
            + ["I_homeostasis" for _ in range(3)]
            + ["I" for _ in range(3)],
            "compartment": ["stn", "snr", "pf"] * 5,
            "format": ["line"] * 15,
        },
        chunk=chunk,
    )

    ### figure with stn, snr and pf rates for both outcome channels
    fig = plt.figure(figsize=(6.4 * 2, 4.8))
    plt.subplot(121)
    _, r_stn_arr = recording_times.combine_periods(
        recordings=recordings, recording_data_str="stn;r", chunk=chunk
    )
    _, r_snr_arr = recording_times.combine_periods(
        recordings=recordings, recording_data_str="snr;r", chunk=chunk
    )
    _, r_sc_arr = recording_times.combine_periods(
        recordings=recordings, recording_data_str="sc;r", chunk=chunk
    )
    t_arr, r_pf_arr = recording_times.combine_periods(
        recordings=recordings, recording_data_str="pf;r", chunk=chunk
    )
    plt.plot(t_arr, r_stn_arr[:, 0], label="stn")
    plt.plot(t_arr, r_snr_arr[:, 0], label="snr")
    plt.plot(t_arr, r_pf_arr[:, 0], label="pf")
    plt.plot(t_arr, r_sc_arr[:, 0], label="sc")
    plt.ylabel("channel 0")
    plt.xlabel("time")
    plt.xlim(1000 - 100, 1000 + 200)
    plt.legend()
    plt.subplot(122)
    plt.plot(t_arr, r_stn_arr[:, 1], label="stn")
    plt.plot(t_arr, r_snr_arr[:, 1], label="snr")
    plt.plot(t_arr, r_pf_arr[:, 1], label="pf")
    plt.plot(t_arr, r_sc_arr[:, 1], label="sc")
    plt.ylabel("channel 1")
    plt.xlabel("time")
    plt.xlim(1000 - 100, 1000 + 200)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/single_trial_rates_stn_snr_pf.png"
    )
    plt.close(fig)


def plot_weight_change_stn_snr(weight_stn_snr_arr):

    plt.figure()
    for idx in range(weight_stn_snr_arr.shape[1]):
        plt.plot(weight_stn_snr_arr[:, idx, 0], label=f"{idx}")
    plt.legend()
    plt.xlabel("trial")
    plt.ylabel("weight")
    plt.grid(True)
    plt.savefig(f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/weight_change_stn_snr.png")


def plot_weight_change_cor_stn(weight_cor_stn_arr):
    ### create matrix plot of weights
    fig = plt.figure()
    for idx in range(weight_cor_stn_arr.shape[1]):
        plt.subplot(weight_cor_stn_arr.shape[1], 1, idx + 1)
        ### heatmap of weights with vertical axis for trials
        plt.imshow(
            weight_cor_stn_arr[:, idx, :].T,
            aspect="auto",
            vmin=np.min(weight_cor_stn_arr),
            vmax=np.max(weight_cor_stn_arr),
        )
        plt.ylabel(f"outcome side {['left', 'right'][idx]}")
        plt.colorbar()
    plt.xlabel("trial")
    plt.tight_layout()
    plt.savefig(f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/weight_change_cor_stn.png")
    plt.close(fig)

    ### create line plot of weights
    fig = plt.figure()
    for idx in range(weight_cor_stn_arr.shape[1]):
        plt.subplot(weight_cor_stn_arr.shape[1], 1, idx + 1)
        plt.plot(weight_cor_stn_arr[:, idx, :], color="black")
        plt.ylabel(f"outcome side {['left', 'right'][idx]}")
        plt.grid(True)
    plt.xlabel("trial")
    plt.savefig(
        f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/weight_change_cor_stn_line.png"
    )
    plt.close(fig)


def get_pf_responses_data(
    outcome_delay_arr,
    outcome_side_arr,
    cor_input_arr,
    recordings,
    recording_times: RecordingTimes,
    pre_onset,
    post_onset,
    get_max=False,
):
    df_dict = {
        "trial": [],
        "outcome_side": [],
        "input_outcome_delay": [],
        "outcome_response_left": [],
        "outcome_response_right": [],
        "cor_input": [],
    }
    ### get pf responses for all trials
    for trial_idx in range(params.CHECK_LEARNING_N_TRIALS):
        df_dict["trial"].append(trial_idx)
        df_dict["outcome_side"].append(outcome_side_arr[trial_idx])
        df_dict["input_outcome_delay"].append(outcome_delay_arr[trial_idx])
        df_dict["cor_input"].append(cor_input_arr[trial_idx])
        ### get the time points of the outcome
        outcome_time = 1000 + outcome_delay_arr[trial_idx]

        ### get the activity of the pf around the outcome
        # pf_neuron: 0-l, 1-r
        time_arr, data_arr = recording_times.combine_periods(
            recordings=recordings, recording_data_str="pf;r", chunk=trial_idx
        )
        ### get data around outcome onset
        peri_outcome_time_mask = (time_arr >= outcome_time - pre_onset) & (
            time_arr < outcome_time + post_onset
        )
        ### get data
        if get_max:
            df_dict["outcome_response_left"].append(
                np.max(data_arr[peri_outcome_time_mask, 0], keepdims=True)
            )
            df_dict["outcome_response_right"].append(
                np.max(data_arr[peri_outcome_time_mask, 1], keepdims=True)
            )
        else:
            df_dict["outcome_response_left"].append(data_arr[peri_outcome_time_mask, 0])
            df_dict["outcome_response_right"].append(
                data_arr[peri_outcome_time_mask, 1]
            )

    ### convert to dataframe
    df = pd.DataFrame(df_dict)

    return df


def plot_pf_responses_matrix_plot(pf_response_df, pre_onset, post_onset):

    plt.figure(figsize=(6.4 * 4, 4.8 * 2))

    for delay_idx, input_outcome_delay in enumerate(
        params.CHECK_LEARNING_T_OFFSET_INPUT_OUTCOME
    ):
        plt.subplot(1, 6, 1 + delay_idx * 2)
        plt.title(f"outcome left {input_outcome_delay}")
        outcome_left_response = np.stack(
            pf_response_df["outcome_response_left"][
                (pf_response_df["outcome_side"] == 0)
                & (pf_response_df["input_outcome_delay"] == input_outcome_delay)
                & (pf_response_df["cor_input"] == True)
            ]
        )
        plot_max_line_in_matrix(outcome_left_response, -pre_onset, post_onset)

        plt.subplot(1, 6, 2 + delay_idx * 2)
        plt.title(f"outcome right {input_outcome_delay}")
        outcome_right_response = np.stack(
            pf_response_df["outcome_response_right"][
                (pf_response_df["outcome_side"] == 1)
                & (pf_response_df["input_outcome_delay"] == input_outcome_delay)
                & (pf_response_df["cor_input"] == True)
            ]
        )
        plot_max_line_in_matrix(outcome_right_response, -pre_onset, post_onset)

    plt.tight_layout()
    plt.savefig(f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/pf_response_matrix.png")


def plot_max_line_in_matrix(matrix, start, end):

    plt.imshow(
        matrix, aspect="auto", vmin=0, vmax=1, extent=[start, end, matrix.shape[0], 0]
    )
    ax: plt.Axes = plt.twiny()
    ax.plot(
        savgol_filter(np.max(matrix, axis=1), 5, 1),
        np.arange(matrix.shape[0]),
        color="red",
        alpha=0.5,
    )
    ax.set_xlim(0, 1)
    ax.tick_params(axis="x", colors="red")


if __name__ == "__main__":

    create_dir(f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots")

    if int(sys.argv[1]) == 0:
        ### analyze a single simulation with self-adjsuted parameters
        save_str = "0_0_0"

        ### load the recordings
        loaded_vars = load_variables(
            name_list=[
                f"recordings_{save_str}",
                f"recording_times_{save_str}",
                f"weight_cor_stn_arr_{save_str}",
                f"weight_stn_snr_arr_{save_str}",
                f"outcome_side_arr_{save_str}",
                f"outcome_delay_arr_{save_str}",
                f"population_name_list_{save_str}",
                f"cor_input_arr_{save_str}",
            ],
            path=params.CHECK_LEARNING_SAVE_FOLDER,
        )
        recordings = loaded_vars[f"recordings_{save_str}"]
        recording_times: RecordingTimes = loaded_vars[f"recording_times_{save_str}"]
        weight_cor_stn_arr = loaded_vars[f"weight_cor_stn_arr_{save_str}"]
        weight_stn_snr_arr = loaded_vars[f"weight_stn_snr_arr_{save_str}"]
        outcome_side_arr = loaded_vars[f"outcome_side_arr_{save_str}"]
        outcome_delay_arr = loaded_vars[f"outcome_delay_arr_{save_str}"]
        population_name_list = loaded_vars[f"population_name_list_{save_str}"]
        cor_input_arr = loaded_vars[f"cor_input_arr_{save_str}"]

        if params.CHECK_LEARNING_N_TRIALS == 1:

            if params.CHECK_LEARNING_SINGLE_TRIAL_PLOT:
                plot_single_trial(
                    recordings, recording_times, population_name_list, chunk=0
                )

        if params.CHECK_LEARNING_N_TRIALS > 1:

            if params.CHECK_LEARNING_CUE_TARGET_STATS_PLOT:
                ### check outcome statistics

                ### after input, when was outcome and on which side
                plot = sns.jointplot(
                    x=outcome_side_arr,
                    y=outcome_delay_arr,
                    kind="scatter",
                    marginal_kws=dict(bins=20, fill=True),
                )
                plot.set_axis_labels("outcome side", "outcome time")
                plot.figure.suptitle("Outcome after input")
                plt.tight_layout()
                plot.figure.savefig(
                    f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/outcome_time_side.png"
                )
                plt.close(plot.figure)

            if params.CHECK_LEARNING_SINGLE_TRIAL_PLOT:
                plot_single_trial(
                    recordings,
                    recording_times,
                    population_name_list,
                    chunk=params.CHECK_LEARNING_N_TRIALS - 1,
                )

            if params.CHECK_LEARNING_WEIGHT_CHANGE_PLOT:
                plot_weight_change_stn_snr(weight_stn_snr_arr)
                plot_weight_change_cor_stn(weight_cor_stn_arr)

            if params.CHECK_LEARNING_PF_RESPONSE_PLOT:
                pre_onset = 10
                post_onset = 60
                pf_response_df = get_pf_responses_data(
                    outcome_delay_arr,
                    outcome_side_arr,
                    cor_input_arr,
                    recordings,
                    recording_times,
                    pre_onset,
                    post_onset,
                )

                ### plot the data as a matrix, x-axis: time, y-axis: trial
                plot_pf_responses_matrix_plot(pf_response_df, pre_onset, post_onset)

    elif int(sys.argv[1]) == 1:
        ### analyze multiple simulations with different combinations of alpha and prob
        idx_list = list(
            itertools.product(
                range(params.CHECK_LEARNING_ALPHA_STEPS),
                range(params.CHECK_LEARNING_PROB_STEPS),
            )
        )

        cor_stn_alpha_arr = np.linspace(0.0, 1.0, params.CHECK_LEARNING_ALPHA_STEPS)
        prob_arr = np.linspace(0.5, 1.0, params.CHECK_LEARNING_PROB_STEPS)

        cor_stn_alpha_list = []
        prob_list = []
        slope_list = []

        for alpha_idx, prob_idx in idx_list:
            save_str = f"1_{alpha_idx}_{prob_idx}"

            cor_stn_alpha = cor_stn_alpha_arr[alpha_idx]
            prob = prob_arr[prob_idx]

            ### load the recordings
            loaded_vars = load_variables(
                name_list=[
                    f"weight_cor_stn_arr_{save_str}",
                ],
                path=params.CHECK_LEARNING_SAVE_FOLDER,
            )
            ### indizes are (trial, outcome side, cortex neuron)
            weight_cor_stn_arr = loaded_vars[f"weight_cor_stn_arr_{save_str}"]

            slope_stn_list = [None, None]
            for stn_neuron in [0, 1]:
                ### get index of cortex neuron with highest average weight to 1st stn neuron
                w = weight_cor_stn_arr[:, stn_neuron, :].mean(axis=0)
                high_idx_p = np.argmax(w)
                ### get the weights over all trials of this cortex neuron
                w_p = weight_cor_stn_arr[:, stn_neuron, high_idx_p]
                ### get the slope of a linear regression with intercept 0
                reg = LinearRegression(fit_intercept=False).fit(
                    np.arange(params.CHECK_LEARNING_N_TRIALS).reshape(-1, 1), w_p
                )
                slope_stn_list[stn_neuron] = reg.coef_[0]

            ### append cor_stn_alpha, prob and the slopes to lists
            ### for p (first cortex population)
            cor_stn_alpha_list.append(cor_stn_alpha)
            prob_list.append(prob)
            slope_list.append(slope_stn_list[0])
            ### for 1-p (second cortex population)
            cor_stn_alpha_list.append(cor_stn_alpha)
            prob_list.append(1 - prob)
            slope_list.append(slope_stn_list[1])

        df = pd.DataFrame(
            {
                "cor_stn_alpha": cor_stn_alpha_list,
                "prob": prob_list,
                "slope": slope_list,
            }
        )
        # sort by cor_stn_alpha and prob
        df = df.sort_values(by=["cor_stn_alpha", "prob"])
        print_df(df)

        ### plot the slope over alpha and prob as a heatmap
        fig = plt.figure()
        plt.imshow(
            np.array(df["slope"]).reshape(
                params.CHECK_LEARNING_ALPHA_STEPS, params.CHECK_LEARNING_PROB_STEPS * 2
            ),
            aspect="auto",
            extent=[0, 1, 1, 0],
        )
        plt.colorbar()
        plt.xlabel("prob")
        plt.ylabel("cor_stn_alpha")
        plt.title("slope")
        plt.savefig(f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/slope_alpha_prob.png")
        plt.close(fig)

        ### plot alpha over slope and prob as scatter plot with color-code
        fig = plt.figure()
        plt.scatter(
            prob_list,
            slope_list,
            c=cor_stn_alpha_list,
            cmap="viridis",
        )
        plt.colorbar()
        plt.xlabel("prob")
        plt.ylabel("slope")
        plt.title("cor_stn_alpha")
        plt.savefig(f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/slope_prob_alpha.png")
        plt.close(fig)

        ### find for each prob the root of the slope for increasing alpha
        root_list = []
        prob_arr_full = df["prob"].unique()
        prob_arr_full = np.sort(prob_arr_full)
        for prob_idx, prob in enumerate(prob_arr_full):
            ### separate dataframe for each prob
            df_prob_all = df[df["prob"] == prob]

            ### find root
            from scipy.optimize import root_scalar

            df_prob = df_prob_all[
                df_prob_all["slope"] > 0.05 * np.max(df_prob_all["slope"])
            ]

            reg = LinearRegression().fit(
                df_prob["cor_stn_alpha"].values.reshape(-1, 1),
                df_prob["slope"].values,
            )
            f = lambda x: reg.intercept_ + reg.coef_[0] * x

            try:
                root_result = root_scalar(f, bracket=[0, 1], method="brentq")
                root = root_result.root
            except:
                root = np.nan
            root_list.append(root)

            fig = plt.figure()
            plt.plot(
                df_prob_all["cor_stn_alpha"],
                df_prob_all["slope"],
                color="black",
                label="data",
            )
            plt.plot(
                np.arange(0, 1.01, 0.01),
                f(np.arange(0, 1.01, 0.01)),
                color="yellow",
                ls="--",
                label="regression",
            )
            if root is not None:
                plt.axvline(root, color="red", linestyle="--", label="root")
            plt.legend()
            plt.xlabel("cor_stn_alpha")
            plt.ylabel("slope")
            plt.title(f"prob = {prob}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/root_alpha_{prob_idx}.png"
            )
            plt.close(fig)

        ### plot root over prob
        root_list = np.array(root_list, dtype=float)
        reg = LinearRegression().fit(
            prob_arr_full[np.logical_not(np.isnan(root_list))].reshape(-1, 1),
            root_list[np.logical_not(np.isnan(root_list))],
        )
        f = lambda x: reg.intercept_ + reg.coef_[0] * x
        fig = plt.figure()
        plt.plot(prob_arr_full, root_list, "k.", label="data")
        plt.plot(
            np.arange(0.0, 1.01, 0.01),
            f(np.arange(0.0, 1.01, 0.01)),
            color="black",
            label="regression",
        )
        plt.legend()
        plt.xlabel("prob")
        plt.ylabel("alpha root")
        plt.title(f"root(prob) = {reg.intercept_:.2f} + {reg.coef_[0]:.2f} * prob")
        plt.grid(True)
        plt.text(0.2, 0.5, "LTD", color="red")
        plt.text(0.7, 0.2, "LTP", color="green")
        plt.tight_layout()
        plt.savefig(f"{params.CHECK_LEARNING_SAVE_FOLDER}/plots/root_prob.png")
        plt.close(fig)
