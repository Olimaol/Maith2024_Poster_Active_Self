from ANNarchy import (
    simulate,
    setup,
    get_projection,
    set_seed,
)
import numpy as np
from CompNeuroPy import (
    CompNeuroMonitors,
    save_variables,
    CompNeuroExp,
    DeapCma,
)
import core_parameters as params
from core_model import prediction_model
from core_experiment import TrialMinamimotoAndKimura
from tqdm import tqdm
from simulation_analyze import get_pf_responses_data, get_target_response_congruent
from sklearn.linear_model import LinearRegression
import sys

sim_id = int(sys.argv[1])


class MyExp(CompNeuroExp):
    def __init__(
        self,
        monitors: CompNeuroMonitors,
        trial_sim: TrialMinamimotoAndKimura,
        show_progress=True,
    ):
        super().__init__(monitors=monitors)
        self.trial_sim = trial_sim
        self.progress_bar = (
            tqdm(range(params.SIMULATION_N_TRIALS))
            if show_progress
            else range(params.SIMULATION_N_TRIALS)
        )

    def run(self):
        ### start each run identical
        set_seed(params.SIMULATION_SEED * sim_id)
        params.SIMULATION_MYRNG.reset()
        self.reset(
            populations=True,
            projections=True,
            synapses=True,
            model_state=True,
            parameters=False,
        )
        set_seed(params.SIMULATION_SEED * sim_id)
        self.trial_sim.sc_all = False

        if params.SIMULATION_N_TRIALS == 1:
            self.monitors.start()
        else:
            ### only start recording of pf for multiple trials (others start later)
            self.monitors.start(compartment_list=["pf"])

        ### record cor__stn weights, cue and target sides and times for all trials
        weight_cor_stn_arr = np.empty(
            (
                params.SIMULATION_N_TRIALS,
                params.N_OUT,
                params.N_STATE * params.N_COR_SEQUENCE,
            )
        )
        weight_stn_snr_arr = np.empty((params.SIMULATION_N_TRIALS, params.N_OUT, 1))
        cue_side_arr = np.empty(params.SIMULATION_N_TRIALS)
        target_side_arr = np.empty(params.SIMULATION_N_TRIALS)
        cue_time_arr = np.empty(params.SIMULATION_N_TRIALS)
        target_time_arr = np.empty(params.SIMULATION_N_TRIALS)

        ### loop over trials
        for trial_idx in self.progress_bar:
            ### before last trial start all recordings
            if (
                params.SIMULATION_N_TRIALS > 1
                and trial_idx == params.SIMULATION_N_TRIALS - 1
            ):
                self.monitors.start()
            ### in last trial activate all sc for target and use fastest target
            if trial_idx == params.SIMULATION_N_TRIALS - 1:
                self.trial_sim.sc_all = True
                # fast target resets automatically to False after run
                self.trial_sim.fast_target = True
            ### reset time / create new chunk
            self.reset(
                populations=False,
                projections=False,
                synapses=False,
                model_state=False,
                parameters=False,
            )
            ### run trial simulation
            trial_sim.run()
            ### get sides and times of cue and target
            cue_side_arr[trial_idx] = trial_sim.cue_side
            target_side_arr[trial_idx] = trial_sim.target_side
            cue_time_arr[trial_idx] = trial_sim.cue_time
            target_time_arr[trial_idx] = trial_sim.target_time
            ### get always the exact same trial simulation (does not affect model, only the random simulation)
            # params.SIMULATION_MYRNG.reset()
            ### store weights
            weight_cor_stn_arr[trial_idx] = get_projection("cor__stn").w
            weight_stn_snr_arr[trial_idx] = get_projection("stn__snr").w

        self.data["weight_cor_stn_arr"] = weight_cor_stn_arr
        self.data["weight_stn_snr_arr"] = weight_stn_snr_arr
        self.data["cue_side_arr"] = cue_side_arr
        self.data["target_side_arr"] = target_side_arr
        self.data["cue_time_arr"] = cue_time_arr
        self.data["target_time_arr"] = target_time_arr

        return self.results()


def get_loss_function(
    exp: MyExp, cor_stn_tau, cor_stn_alpha, stn_snr_tau, stn_snr_alpha
):

    ### set the paramters of the model
    get_projection("cor__stn").tau = cor_stn_tau
    get_projection("cor__stn").alpha = cor_stn_alpha
    get_projection("stn__snr").tau = stn_snr_tau
    get_projection("stn__snr").alpha = stn_snr_alpha

    ### run the experiment / multiple trials
    results = exp.run()

    ### get the pf responses of the trials (only max response)
    pre_onset = 10
    post_onset = 60
    pf_response_df = get_pf_responses_data(
        results.data["cue_time_arr"],
        results.data["target_time_arr"],
        results.data["cue_side_arr"],
        results.data["target_side_arr"],
        results.recordings,
        results.recording_times,
        pre_onset,
        post_onset,
        get_max=True,
    )

    ### get the pf responses for cue-target delay of 100 and separate for congruent and
    ### incongruent cues
    target_congruent_response = get_target_response_congruent(
        pf_response_df,
        True,
        100,
    )[:, 0]
    target_incongruent_response = get_target_response_congruent(
        pf_response_df,
        False,
        100,
    )[:, 0]

    ### linear regression for both
    reg_congruent = LinearRegression().fit(
        X=np.arange(len(target_congruent_response)).reshape(-1, 1),
        y=target_congruent_response,
    )
    reg_incongruent = LinearRegression().fit(
        X=np.arange(len(target_incongruent_response)).reshape(-1, 1),
        y=target_incongruent_response,
    )

    loss = reg_congruent.coef_[0] - np.clip(reg_incongruent.coef_[0], None, 0)

    return loss


def evaluate_function(population):
    loss_list = []
    ### the population is a list of individuals which are lists of parameters
    for individual in population:
        loss_of_individual = get_loss_function(
            my_exp,
            cor_stn_tau=individual[0],
            cor_stn_alpha=individual[1],
            stn_snr_tau=individual[2],
            stn_snr_alpha=individual[1] * individual[3],
        )
        loss_list.append((loss_of_individual,))
    return loss_list


if __name__ == "__main__":

    ### set seed for ANNarchy
    setup(seed=params.SIMULATION_SEED * sim_id)

    ### create the model and compile it
    prediction_model.create(
        warn=False,
        compile_folder_name=prediction_model.compile_folder_name + f"_{sim_id}",
    )

    ### create monitors
    if params.SIMULATION_N_TRIALS == 1:
        recording_period = ""
    else:
        recording_period = ";5"
    mon_dict = {
        f"{pop_name}{recording_period}": ["r"]
        for pop_name in prediction_model.populations
    }
    monitors = CompNeuroMonitors(mon_dict=mon_dict)

    ### create the simulation object for simulating a trial
    trial_sim = TrialMinamimotoAndKimura(
        rng=params.SIMULATION_MYRNG.rng,
        p_congruent=0.8,
        sc_input=True,
        verbose=False,
        hold_response=params.SIMULATION_HOLD_RESPONSE,
        hold_state=params.SIMULATION_HOLD_STATE,
        cue_response=params.SIMULATION_CUE_RESPONSE,
        cue_state=params.SIMULATION_CUE_STATE,
        only_left_cue=params.SIMULATION_ONLY_LEFT_CUE,
    )

    ### create the experiment object for simulating multiple trials
    my_exp = MyExp(
        monitors=monitors,
        trial_sim=trial_sim,
        show_progress=not params.SIMULATION_OPTIMIZE,
    )

    ### ramp up simulation to get to a stable state
    simulate(params.SIMULATION_T_RAMP_UP)
    my_exp.store_model_state(
        prediction_model.populations + prediction_model.projections
    )

    if params.SIMULATION_OPTIMIZE:
        ### define lower bounds of paramters to optimize
        lb = np.array([100, 0, 100, 0])

        ### define upper bounds of paramters to optimize
        ub = np.array([10000, 0.1, 5000, 0.5])

        ### create an "minimal" instance of the DeapCma class
        deap_cma = DeapCma(
            lower=lb,
            upper=ub,
            evaluate_function=evaluate_function,
            param_names=[
                "cor_stn_tau",
                "cor_stn_alpha",
                "stn_snr_tau",
                "stn_snr_alpha_factor",
            ],
            hard_bounds=True,
            p0=np.array([6500, 0.04, 200, 0.17]),
        )

        ### run the optimization
        deap_cma_result = deap_cma.run(max_evals=params.SIMULATION_MAX_EVALS)

        ### store the optimization results
        save_variables(
            variable_list=[deap_cma_result],
            name_list=[f"deap_cma_result_{sim_id}"],
            path=params.SIMULATION_SAVE_FOLDER,
        )

        ### set the optimized parameters for the following simulation
        get_projection("cor__stn").tau = deap_cma_result["cor_stn_tau"]
        get_projection("cor__stn").alpha = deap_cma_result["cor_stn_alpha"]
        get_projection("stn__snr").tau = deap_cma_result["stn_snr_tau"]
        get_projection("stn__snr").alpha = (
            deap_cma_result["cor_stn_alpha"] * deap_cma_result["stn_snr_alpha_factor"]
        )

    ### run experiment
    results = my_exp.run()

    ### store data
    save_variables(
        variable_list=[
            results.recordings,
            results.recording_times,
            results.data["weight_cor_stn_arr"],
            results.data["weight_stn_snr_arr"],
            results.data["cue_side_arr"],
            results.data["target_side_arr"],
            results.data["cue_time_arr"],
            results.data["target_time_arr"],
            prediction_model.populations,
        ],
        name_list=[
            f"recordings_{sim_id}",
            f"recording_times_{sim_id}",
            f"weight_cor_stn_arr_{sim_id}",
            f"weight_stn_snr_arr_{sim_id}",
            f"cue_side_arr_{sim_id}",
            f"target_side_arr_{sim_id}",
            f"cue_time_arr_{sim_id}",
            f"target_time_arr_{sim_id}",
            f"population_name_list_{sim_id}",
        ],
        path=params.SIMULATION_SAVE_FOLDER,
    )
