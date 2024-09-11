import core_parameters as params
from CompNeuroPy import PlotRecordings, load_variables, create_dir
from CompNeuroPy.monitors import RecordingTimes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import pandas as pd
from scipy.signal import savgol_filter
import sys
import os


def plot_pf_response_line(
    time_arr, response_arr, early: bool, label: str, color: str, linestyle: str
):
    if early:
        start_idx = 0
        end_idx = min([response_arr.shape[0] // 2, 100])
    else:
        start_idx = max([response_arr.shape[0] // 2, response_arr.shape[0] - 100])
        end_idx = response_arr.shape[0]
    plt.plot(
        time_arr,
        response_arr[start_idx:end_idx].mean(axis=0),
        label=label,
        color=color,
        linestyle=linestyle,
    )
    plt.fill_between(
        time_arr,
        response_arr[start_idx:end_idx].mean(axis=0)
        - response_arr[start_idx:end_idx].std(axis=0),
        response_arr[start_idx:end_idx].mean(axis=0)
        + response_arr[start_idx:end_idx].std(axis=0),
        alpha=0.5,
        color=color,
        linestyle=linestyle,
        # without lines on the edge
        linewidth=0,
    )


def adjust_text_of_current_axis(fontsize):
    ca = plt.gca()
    ### set the title font to Arial, bold
    ca.title.set_fontname("Arial")
    ca.title.set_fontweight("bold")
    ca.title.set_fontsize(fontsize)
    ### set the labels font to Arial, bold
    ca.xaxis.label.set_fontname("Arial")
    ca.yaxis.label.set_fontname("Arial")
    ca.xaxis.label.set_fontweight("bold")
    ca.yaxis.label.set_fontweight("bold")
    ca.xaxis.label.set_fontsize(fontsize)
    ca.yaxis.label.set_fontsize(fontsize)
    ### set the label ticks font to Arial
    for tick in ca.get_xticklabels() + ca.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontsize(fontsize)
    ### recreate the legend with font Arial
    if ca.get_legend():
        ca.get_legend().remove()
        ca.legend(
            prop={
                "family": "Arial",
                "size": fontsize,
            }
        )


def plot_pf_responses_line_plot(pf_response_df, pre_onset, post_onset):
    time_arr = np.linspace(
        -pre_onset, post_onset, len(pf_response_df["cue_response_left"][0])
    )

    fig = plt.figure(figsize=params.SIMULATION_PF_RESPONSE_PLOT_PROPERTIES["figsize"])

    cue_preferred_response = get_cue_response_preffered(pf_response_df, True)
    cue_notpreferred_response = get_cue_response_preffered(pf_response_df, False)
    target_congruent_response = get_target_response_congruent(
        pf_response_df,
        True,
        400,
    )
    target_incongruent_response = get_target_response_congruent(
        pf_response_df,
        False,
        400,
    )
    for early_late_idx, early in enumerate([True, False]):

        plt.subplot(2, 2, 1 + 2 * early_late_idx)
        plot_pf_response_line(
            time_arr,
            cue_preferred_response,
            early,
            "preferred",
            color="royalblue",
            linestyle="solid",
        )
        plot_pf_response_line(
            time_arr,
            cue_notpreferred_response,
            early,
            "not preferred",
            color="darkorange",
            linestyle="solid",
        )
        plt.grid(params.SIMULATION_PF_RESPONSE_PLOT_PROPERTIES["grid"])
        plt.ylim(0.15, 0.95)
        if early_late_idx == 0:
            plt.title("CM/Pf cue response")
            plt.ylabel("early\nfiring rate")
            plt.gca().set_xticklabels([])
        else:
            plt.xlabel("time from cue onset [ms]")
            plt.ylabel("late\nfiring rate")
            plt.legend()
        adjust_text_of_current_axis(
            fontsize=params.SIMULATION_PF_RESPONSE_PLOT_PROPERTIES["fontsize"]
        )

        plt.subplot(2, 2, 2 + 2 * early_late_idx)
        plot_pf_response_line(
            time_arr,
            target_congruent_response,
            early,
            "valid",
            color="royalblue",
            linestyle=(5, (10, 3)),
        )
        plot_pf_response_line(
            time_arr,
            target_incongruent_response,
            early,
            "invalid",
            color="cornflowerblue",
            linestyle="dotted",
        )
        plt.grid(params.SIMULATION_PF_RESPONSE_PLOT_PROPERTIES["grid"])
        plt.ylim(0.15, 0.95)
        if early_late_idx == 0:
            plt.title("CM/Pf target response (preferred, 400 ms)")
            plt.gca().set_xticklabels([])
        else:
            plt.xlabel("time from target onset [ms]")
            plt.legend()
        plt.gca().set_yticklabels([])
        adjust_text_of_current_axis(
            fontsize=params.SIMULATION_PF_RESPONSE_PLOT_PROPERTIES["fontsize"]
        )

    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    plt.savefig(
        f"{params.SIMULATION_SAVE_FOLDER}/plots/pf_response.png",
        dpi=300,
    )
    plt.close(fig)


def get_cue_response_preffered(pf_response_df: pd.DataFrame, preferred: bool):
    pf_response_cue_preferred = pd.DataFrame(
        {
            "cue_response": pd.concat(
                [
                    pf_response_df["cue_response_left"][
                        pf_response_df["cue_side"] == (0 if preferred else 1)
                    ],
                    pf_response_df["cue_response_right"][
                        pf_response_df["cue_side"] == (1 if preferred else 0)
                    ],
                ],
                axis=0,
                ignore_index=True,
            ),
            "trial": pd.concat(
                [
                    pf_response_df["trial"][pf_response_df["cue_side"] == 0],
                    pf_response_df["trial"][pf_response_df["cue_side"] == 1],
                ],
                axis=0,
                ignore_index=True,
            ),
        }
    ).sort_values(by="trial")

    # convert the sequence of arrays to a single array (first dimension is trial)
    pf_response_cue_preferred = np.stack(pf_response_cue_preferred["cue_response"])
    return pf_response_cue_preferred


def get_target_response_congruent(
    pf_response_df: pd.DataFrame, congruent: bool, cue_target_delay: int | None = None
):
    pf_response_target_congruent = pd.DataFrame(
        {
            "target_response": pd.concat(
                [
                    pf_response_df["target_response_left"][
                        (pf_response_df["target_side"] == 0)
                        & (pf_response_df["cue_side"] == (0 if congruent else 1))
                        & (
                            (pf_response_df["cue_target_delay"] == cue_target_delay)
                            if cue_target_delay is not None
                            else True
                        )
                    ],
                    pf_response_df["target_response_right"][
                        (pf_response_df["target_side"] == 1)
                        & (pf_response_df["cue_side"] == (1 if congruent else 0))
                        & (
                            (pf_response_df["cue_target_delay"] == cue_target_delay)
                            if cue_target_delay is not None
                            else True
                        )
                    ],
                ],
                axis=0,
                ignore_index=True,
            ),
            "trial": pd.concat(
                [
                    pf_response_df["trial"][
                        (pf_response_df["target_side"] == 0)
                        & (pf_response_df["cue_side"] == (0 if congruent else 1))
                        & (
                            (pf_response_df["cue_target_delay"] == cue_target_delay)
                            if cue_target_delay is not None
                            else True
                        )
                    ],
                    pf_response_df["trial"][
                        (pf_response_df["target_side"] == 1)
                        & (pf_response_df["cue_side"] == (1 if congruent else 0))
                        & (
                            (pf_response_df["cue_target_delay"] == cue_target_delay)
                            if cue_target_delay is not None
                            else True
                        )
                    ],
                ],
                axis=0,
                ignore_index=True,
            ),
        }
    ).sort_values(by="trial")

    ### raise error if there are no rows
    if pf_response_target_congruent.shape[0] == 0:
        raise ValueError(
            f"No trials found for the given conditions (congruent: {congruent}, cue_target_delay: {cue_target_delay})"
        )

    # convert the sequence of arrays to a single array (first dimension is trial)
    pf_response_target_congruent = np.stack(
        pf_response_target_congruent["target_response"]
    )
    return pf_response_target_congruent


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


def plot_pf_responses_matrix_plot(pf_response_df, pre_onset, post_onset):

    fig = plt.figure(figsize=(6.4 * 4, 4.8 * 2))
    plt.subplot(1, 9, 1)
    plt.title("hold light")
    hold_response = np.stack(pf_response_df["hold_response"])
    plot_max_line_in_matrix(hold_response, 0, post_onset)

    plt.subplot(1, 9, 2)
    plt.title("cue preferred")
    cue_preferred_response = get_cue_response_preffered(pf_response_df, True)
    plot_max_line_in_matrix(cue_preferred_response, -pre_onset, post_onset)

    plt.subplot(1, 9, 3)
    plt.title("cue non-preferred")
    cue_non_preferred_response = get_cue_response_preffered(pf_response_df, False)
    plot_max_line_in_matrix(cue_non_preferred_response, -pre_onset, post_onset)

    for delay_idx, cue_target_delay in enumerate(params.SIMULATION_T_OFFSET_CUE_TARGET):
        plt.subplot(1, 9, 4 + delay_idx * 2)
        plt.title(f"target congruent {cue_target_delay}")
        target_congruent_response = get_target_response_congruent(
            pf_response_df, True, cue_target_delay
        )
        plot_max_line_in_matrix(target_congruent_response, -pre_onset, post_onset)

        plt.subplot(1, 9, 5 + delay_idx * 2)
        plt.title(f"target incongruent {cue_target_delay}")
        target_incongruent_response = get_target_response_congruent(
            pf_response_df, False, cue_target_delay
        )
        plot_max_line_in_matrix(target_incongruent_response, -pre_onset, post_onset)

    plt.tight_layout()
    plt.savefig(
        f"{params.SIMULATION_SAVE_FOLDER}/plots/pf_response_matrix_{sys.argv[1]}.png"
    )
    plt.close(fig)


def get_pf_responses_data(
    cue_time_arr,
    target_time_arr,
    cue_side_arr,
    target_side_arr,
    recordings,
    recording_times: RecordingTimes,
    pre_onset,
    post_onset,
    get_max=False,
):
    df_dict = {
        "trial": [],
        "cue_side": [],
        "target_side": [],
        "cue_target_delay": [],
        "hold_response": [],
        "cue_response_left": [],
        "cue_response_right": [],
        "target_response_left": [],
        "target_response_right": [],
    }
    ### get pf responses for all trials
    for trial_idx in range(params.SIMULATION_N_TRIALS):
        df_dict["trial"].append(int(trial_idx))
        df_dict["cue_side"].append(int(cue_side_arr[trial_idx]))
        df_dict["target_side"].append(int(target_side_arr[trial_idx]))
        ### get the time points of the cue and target
        cue_time = cue_time_arr[trial_idx]
        target_time = target_time_arr[trial_idx]
        df_dict["cue_target_delay"].append(int(round(target_time - cue_time)))

        ### get the activity of the pf after around the hold light
        time_arr, data_arr = recording_times.combine_periods(
            recordings=recordings, recording_data_str="pf;r", chunk=trial_idx
        )
        peri_hold_light_time_mask = (time_arr >= 0) & (time_arr < post_onset)
        if get_max:
            df_dict["hold_response"].append(
                np.max(data_arr[peri_hold_light_time_mask, 0], keepdims=True)
            )
        else:
            df_dict["hold_response"].append(data_arr[peri_hold_light_time_mask, 0])

        ### get the activity of the pf around the cue
        # pf_neuron: 1-l, 2-r
        ### get data around cue onset
        peri_cue_time_maks = (time_arr >= cue_time - pre_onset) & (
            time_arr < cue_time + post_onset
        )
        ### get data
        if get_max:
            df_dict["cue_response_left"].append(
                np.max(data_arr[peri_cue_time_maks, 1], keepdims=True)
            )
            df_dict["cue_response_right"].append(
                np.max(data_arr[peri_cue_time_maks, 2], keepdims=True)
            )
        else:
            df_dict["cue_response_left"].append(data_arr[peri_cue_time_maks, 1])
            df_dict["cue_response_right"].append(data_arr[peri_cue_time_maks, 2])

        ### get the activity of the pf around the target
        # pf_neuron: 1-l, 2-r
        ### get data around target onset
        peri_target_time_mask = (time_arr >= target_time - pre_onset) & (
            time_arr < target_time + post_onset
        )
        ### get data
        if get_max:
            df_dict["target_response_left"].append(
                np.max(data_arr[peri_target_time_mask, 1], keepdims=True)
            )
            df_dict["target_response_right"].append(
                np.max(data_arr[peri_target_time_mask, 2], keepdims=True)
            )
        else:
            df_dict["target_response_left"].append(data_arr[peri_target_time_mask, 1])
            df_dict["target_response_right"].append(data_arr[peri_target_time_mask, 2])

    ### convert to dataframe
    df = pd.DataFrame(df_dict)

    return df


def plot_cortex_and_responses_matrix(
    r_arr, label, time_mask, x_label, x_ticks, top_label, mid_label, bottom_label
):
    plt.imshow(
        r_arr[time_mask].T,
        aspect="auto",
        vmin=0,
        vmax=1,
        extent=[-100, 200, r_arr.shape[1], 0],
    )
    plt.axhline(y=r_arr.shape[1] / 3 - (0.5 if "Cortex" in label else 0), color="black")
    plt.axhline(
        y=2 * r_arr.shape[1] / 3 - (0.5 if "Cortex" in label else 0), color="black"
    )
    plt.xlabel(x_label)
    plt.ylabel(
        label,
        rotation=0,
        labelpad=params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["y_label_pad"],
        va="center",
    )
    plt.text(
        x=params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["text_position"],
        y=r_arr.shape[1] / 6,
        s=top_label,
        va="center",
        ha="left",
        fontdict={
            "family": "Arial",
            "size": params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["fontsize"],
        },
    )
    plt.text(
        x=params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["text_position"],
        y=3 * r_arr.shape[1] / 6,
        s=mid_label,
        va="center",
        ha="left",
        fontdict={
            "family": "Arial",
            "size": params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["fontsize"],
        },
    )
    plt.text(
        x=params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["text_position"],
        y=5 * r_arr.shape[1] / 6,
        s=bottom_label,
        va="center",
        ha="left",
        fontdict={
            "family": "Arial",
            "size": params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["fontsize"],
        },
    )
    plt.yticks([])
    if not (x_ticks):
        plt.xticks([])
    plt.grid(params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["grid"])


def plot_single_trial(
    recordings, recording_times, population_name_list, cue_time_arr, chunk=0
):
    ### plot activities/rates of all populations
    PlotRecordings(
        figname=f"{params.SIMULATION_SAVE_FOLDER}/plots/single_trial_r_line_{sys.argv[1]}.png",
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
        figname=f"{params.SIMULATION_SAVE_FOLDER}/plots/single_trial_r_matrix_{sys.argv[1]}.png",
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

    ### get firing rate arrays and corresponding time array

    _, r_cor_arr = recording_times.combine_periods(
        recordings=recordings, recording_data_str="cor;r", chunk=chunk
    )
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

    ### figure with sc, stn, snr and pf rates as lines
    fig = plt.figure(figsize=(6.4 * r_stn_arr.shape[1], 4.8))
    for channel_idx in range(r_stn_arr.shape[1]):
        plt.subplot(1, r_stn_arr.shape[1], channel_idx + 1)
        plt.plot(t_arr, r_stn_arr[:, channel_idx], label=f"stn {channel_idx}")
        plt.plot(t_arr, r_snr_arr[:, channel_idx], label=f"snr {channel_idx}")
        plt.plot(t_arr, r_pf_arr[:, channel_idx], label=f"pf {channel_idx}")
        plt.plot(t_arr, r_sc_arr[:, channel_idx], label=f"sc {channel_idx}")
        plt.ylabel(f"channel {channel_idx}")
        plt.xlabel("time")
        plt.xlim(cue_time_arr[chunk] - 100, cue_time_arr[chunk] + 200)
        plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{params.SIMULATION_SAVE_FOLDER}/plots/single_trial_rates_stn_snr_pf.png"
    )
    plt.close(fig)

    ### figure with cor, sc, stn, snr and pf rates as matrix
    time_mask = (t_arr >= cue_time_arr[chunk] - 100) & (
        t_arr < cue_time_arr[chunk] + 200
    )
    fig = plt.figure(
        figsize=params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["figsize"]
    )
    plot_data_list = [
        ["State\n(Cortex)", r_cor_arr],
        ["STN", r_stn_arr],
        ["SNr", r_snr_arr],
        ["CM/Pf", r_pf_arr],
        ["Outcome\n(SC)", r_sc_arr],
    ]
    rowspan_first = params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["rowspan_first"]
    rowspan_others = params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES[
        "rowspan_others"
    ]
    n_rows = rowspan_first + (len(plot_data_list) - 1) * rowspan_others
    n_columns = 1
    for plot_idx, plot_data in enumerate(plot_data_list):
        label, r_arr = plot_data

        row_nbr = plot_idx + (rowspan_first - 1 if plot_idx > 0 else 0)

        plt.subplot2grid(
            shape=(n_rows, n_columns),
            loc=(row_nbr, 0),
            rowspan=rowspan_first if plot_idx == 0 else rowspan_others,
        )
        if plot_idx == 0:
            top_label = "'hold light\nflashed'"
            mid_label = "'cue was\nleft'"
            bottom_label = "'cue was\nright'"
        elif plot_idx == len(plot_data_list) - 1:
            top_label = "'hold light'"
            mid_label = "'left light'"
            bottom_label = "'right light'"
        else:
            top_label = ""
            mid_label = ""
            bottom_label = ""
        plot_cortex_and_responses_matrix(
            r_arr,
            label,
            time_mask,
            x_label=(
                "time from cue onset [ms]"
                if plot_idx == len(plot_data_list) - 1
                else ""
            ),
            x_ticks=plot_idx == len(plot_data_list) - 1,
            top_label=top_label,
            mid_label=mid_label,
            bottom_label=bottom_label,
        )
        adjust_text_of_current_axis(
            fontsize=params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["fontsize"]
        )

    plt.tight_layout(
        pad=0,
        h_pad=params.SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES["h_pad"],
        w_pad=1.08,
    )
    plt.savefig(
        f"{params.SIMULATION_SAVE_FOLDER}/plots/single_trial_rates_cor_and_responses_matrix.png",
        dpi=300,
    )
    plt.close(fig)


def plot_weight_change_stn_snr(weight_stn_snr_arr):
    print(weight_stn_snr_arr.shape)
    print(weight_stn_snr_arr[0])
    print(weight_stn_snr_arr[-1])
    fig = plt.figure()
    for idx in range(weight_stn_snr_arr.shape[1]):
        plt.plot(weight_stn_snr_arr[:, idx, 0], label=f"{idx}")
    plt.legend()
    plt.xlabel("trial")
    plt.ylabel("weight")
    plt.grid(True)
    plt.savefig(
        f"{params.SIMULATION_SAVE_FOLDER}/plots/weight_change_stn_snr_{sys.argv[1]}.png"
    )
    plt.close(fig)


def plot_weight_change_cor_stn(weight_cor_stn_arr):
    ### create matrix plot of weights
    fig = plt.figure(figsize=(6.4 * params.N_STATE, 4.8 * weight_cor_stn_arr.shape[1]))
    # subplots: rows = outcome, columns = state
    for state_idx in range(params.N_STATE):
        for outcome_idx in range(weight_cor_stn_arr.shape[1]):
            plt.subplot(
                weight_cor_stn_arr.shape[1],
                params.N_STATE,
                state_idx + 1 + outcome_idx * params.N_STATE,
            )
            if outcome_idx == 0:
                plt.title(
                    f"state {['hold', 'cue was left', 'cue was right'][state_idx]}"
                )
            if state_idx == 0:
                plt.ylabel(
                    f"outcome {['hold light','left ligth', 'right light'][outcome_idx]}"
                )
            ### heatmap of weights with vertical axis for trials
            state_start_idx = state_idx * params.N_COR_SEQUENCE
            state_end_idx = (state_idx + 1) * params.N_COR_SEQUENCE
            plt.imshow(
                weight_cor_stn_arr[:, outcome_idx, state_start_idx:state_end_idx].T,
                aspect="auto",
                vmin=np.min(weight_cor_stn_arr),
                vmax=np.max(weight_cor_stn_arr),
            )
            plt.colorbar()
    plt.xlabel("trial")
    plt.tight_layout()
    plt.savefig(f"{params.SIMULATION_SAVE_FOLDER}/plots/weight_change_cor_stn.png")
    plt.close(fig)

    ### create line plot of weights
    fig = plt.figure(figsize=(6.4 * params.N_STATE, 4.8 * weight_cor_stn_arr.shape[1]))
    # subplots: rows = outcome, columns = state
    for state_idx in range(params.N_STATE):
        for outcome_idx in range(weight_cor_stn_arr.shape[1]):
            plt.subplot(
                weight_cor_stn_arr.shape[1],
                params.N_STATE,
                state_idx + 1 + outcome_idx * params.N_STATE,
            )
            if outcome_idx == 0:
                plt.title(f"state {state_idx}")
            if state_idx == 0:
                plt.ylabel(
                    f"outcome {['hold light','left ligth', 'right light'][outcome_idx]}"
                )
            state_start_idx = state_idx * params.N_COR_SEQUENCE
            state_end_idx = (state_idx + 1) * params.N_COR_SEQUENCE
            plt.plot(
                weight_cor_stn_arr[:, outcome_idx, state_start_idx:state_end_idx],
                color="black",
            )
            plt.grid(True)
    plt.xlabel("trial")
    plt.tight_layout()
    plt.savefig(f"{params.SIMULATION_SAVE_FOLDER}/plots/weight_change_cor_stn_line.png")
    plt.close(fig)


def plot_cue_target_stats(cue_side_arr, cue_time_arr, target_side_arr, target_time_arr):
    ### after hold, when was cue and on which side
    plot = sns.jointplot(
        x=cue_side_arr,
        y=cue_time_arr,
        kind="scatter",
        marginal_kws=dict(bins=20, fill=True),
    )
    plot.set_axis_labels("cue side", "cue time")
    plot.figure.suptitle("Cue after hold light")
    plt.tight_layout()
    plot.figure.savefig(
        f"{params.SIMULATION_SAVE_FOLDER}/plots/cue_time_side_{sys.argv[1]}.png"
    )
    plt.close(plot.figure)

    ### after cue, when was target and on which side
    ### cue was left
    df = pd.DataFrame(
        {
            "cue_side": cue_side_arr,
            "target_side": target_side_arr,
            "target_time": target_time_arr - cue_time_arr,
        }
    )
    plot = sns.jointplot(
        x="target_side",
        y="target_time",
        data=df[df["cue_side"] == 0],
        kind="scatter",
        marginal_kws=dict(bins=20, fill=True),
    )
    plot.set_axis_labels("target side", "target time")
    plot.figure.suptitle("Target after cue was left")
    plt.tight_layout()
    plot.figure.savefig(
        f"{params.SIMULATION_SAVE_FOLDER}/plots/target_time_side_left_{sys.argv[1]}.png"
    )
    plt.close(plot.figure)
    ### cue was right
    plot = sns.jointplot(
        x="target_side",
        y="target_time",
        data=df[df["cue_side"] == 1],
        kind="scatter",
        marginal_kws=dict(bins=20, fill=True),
    )
    plot.set_axis_labels("target side", "target time")
    plot.figure.suptitle("Target after cue was right")
    plt.tight_layout()
    plot.figure.savefig(
        f"{params.SIMULATION_SAVE_FOLDER}/plots/target_time_side_right_{sys.argv[1]}.png"
    )
    plt.close(plot.figure)


if __name__ == "__main__":
    ### Add font Arial to matplotlib
    # Specify the directory where Arial is located
    arial_font_path = "/usr/share/fonts/truetype/msttcorefonts/"
    # Add all fonts from the directory to matplotlib's font manager
    if os.path.exists(arial_font_path):
        for font_file in os.listdir(arial_font_path):
            if font_file.endswith(".ttf"):
                font_manager.fontManager.addfont(
                    os.path.join(arial_font_path, font_file)
                )

    ### load the recordings
    loaded_vars = load_variables(
        name_list=[
            f"recordings_{sys.argv[1]}",
            f"recording_times_{sys.argv[1]}",
            f"weight_cor_stn_arr_{sys.argv[1]}",
            f"weight_stn_snr_arr_{sys.argv[1]}",
            f"cue_side_arr_{sys.argv[1]}",
            f"target_side_arr_{sys.argv[1]}",
            f"cue_time_arr_{sys.argv[1]}",
            f"target_time_arr_{sys.argv[1]}",
            f"population_name_list_{sys.argv[1]}",
        ],
        path=params.SIMULATION_SAVE_FOLDER,
    )
    recordings = loaded_vars[f"recordings_{sys.argv[1]}"]
    recording_times: RecordingTimes = loaded_vars[f"recording_times_{sys.argv[1]}"]
    weight_cor_stn_arr = loaded_vars[f"weight_cor_stn_arr_{sys.argv[1]}"]
    weight_stn_snr_arr = loaded_vars[f"weight_stn_snr_arr_{sys.argv[1]}"]
    cue_side_arr = loaded_vars[f"cue_side_arr_{sys.argv[1]}"]
    target_side_arr = loaded_vars[f"target_side_arr_{sys.argv[1]}"]
    cue_time_arr = loaded_vars[f"cue_time_arr_{sys.argv[1]}"]
    target_time_arr = loaded_vars[f"target_time_arr_{sys.argv[1]}"]
    population_name_list = loaded_vars[f"population_name_list_{sys.argv[1]}"]

    create_dir(f"{params.SIMULATION_SAVE_FOLDER}/plots")

    if params.SIMULATION_N_TRIALS == 1:

        if params.SIMULATION_SINGLE_TRIAL_PLOT:
            plot_single_trial(
                recordings, recording_times, population_name_list, cue_time_arr, chunk=0
            )

    if params.SIMULATION_N_TRIALS > 1:

        if params.SIMULATION_CUE_TARGET_STATS_PLOT:
            ### check general cue, target statistics
            plot_cue_target_stats(
                cue_side_arr, cue_time_arr, target_side_arr, target_time_arr
            )

        if params.SIMULATION_SINGLE_TRIAL_PLOT:
            plot_single_trial(
                recordings,
                recording_times,
                population_name_list,
                cue_time_arr,
                chunk=params.SIMULATION_N_TRIALS - 1,
            )

        if params.SIMULATION_WEIGHT_CHANGE_PLOT:
            plot_weight_change_stn_snr(weight_stn_snr_arr)
            plot_weight_change_cor_stn(weight_cor_stn_arr)

        if params.SIMULATION_PF_RESPONSE_PLOT:
            pre_onset = 20
            post_onset = 100
            pf_response_df = get_pf_responses_data(
                cue_time_arr=cue_time_arr,
                target_time_arr=target_time_arr,
                cue_side_arr=cue_side_arr,
                target_side_arr=target_side_arr,
                recordings=recordings,
                recording_times=recording_times,
                pre_onset=pre_onset,
                post_onset=post_onset,
            )

            ### plot the data
            plot_pf_responses_line_plot(pf_response_df, pre_onset, post_onset)

            ### plot the data as a matrix, x-axis: time, y-axis: trial
            plot_pf_responses_matrix_plot(pf_response_df, pre_onset, post_onset)
