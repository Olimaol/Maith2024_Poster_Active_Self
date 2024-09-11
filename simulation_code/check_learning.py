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
)
import core_parameters as params
from core_model import prediction_model
from core_experiment import TrialCheckLearning
from tqdm import tqdm
import sys

"""
sys.argv values:
    0: check_learning.py
    1: simulation index
    2: alpha index, between 0 and 19
    3: probability index, between 0 and 19
"""
sim_idx = int(sys.argv[1])

### simd _idx == 0 --> adjust paramters by myself
if sim_idx == 0:
    cor_stn_alpha = 0.03
    prob = 0.8
    alpha_idx = 0
    prob_idx = 0
    do_monitor = True
elif sim_idx == 1:
    ### simd _idx == 1 --> adjust parameters alpha and prob by sys.argv
    alpha_idx = int(sys.argv[2])
    prob_idx = int(sys.argv[3])
    cor_stn_alpha = np.linspace(0.0, 1.0, params.CHECK_LEARNING_ALPHA_STEPS)[alpha_idx]
    prob = np.linspace(0.5, 1.0, params.CHECK_LEARNING_PROB_STEPS)[prob_idx]
    do_monitor = False

### set string to save files
save_str = f"{sim_idx}_{alpha_idx}_{prob_idx}"


def prepare_model():
    ### set seed for ANNarchy
    setup(seed=params.CHECK_LEARNING_SEED, precision="double")

    ### create the model and compile it
    prediction_model.model_kwargs = {
        "n_state": 1,
        "n_out": 2,
        "cor_stn_tau": 7500,
        "cor_stn_alpha": cor_stn_alpha,
        "cor_stn_w_init": 0.0,
        "stn_snr_tau": 250,
        "stn_snr_alpha": cor_stn_alpha * 0.17,
        "stn_snr_w_init": 0.0,
    }
    prediction_model.create(
        warn=False,
        compile_folder_name=f"check_learning_{save_str}",
    )

    ### create monitors
    if params.CHECK_LEARNING_N_TRIALS == 1:
        recording_period = ""
    else:
        recording_period = ";5"
    mon_dict = {
        f"{pop_name}{recording_period}": ["r"]
        for pop_name in prediction_model.populations
    }
    mon_dict[f"cor{recording_period}"].extend(["r_trace_increase_event"])
    mon_dict[f"stn{recording_period}"].extend(
        [
            "learn_mod",
            "mod_increase",
            "I_signal",
            "I_noise",
            "I_homeostasis",
            "I",
            "r_trace_increase_event",
            "r_trace_increase_event_low",
        ]
    )
    mon_dict[f"snr{recording_period}"].extend(
        [
            "learn_mod",
            "mod_increase",
            "I_signal",
            "I_noise",
            "I_homeostasis",
            "I",
        ]
    )
    mon_dict[f"pf{recording_period}"].extend(
        ["I_signal", "I_noise", "I_homeostasis", "I"]
    )
    mon_dict["cor__stn;10"] = ["w"]
    mon_dict["stn__snr;10"] = ["w"]
    monitors = CompNeuroMonitors(mon_dict=mon_dict)

    return monitors


class MyExp(CompNeuroExp):
    def __init__(self, monitors: CompNeuroMonitors, trial_sim: TrialCheckLearning):
        super().__init__(monitors=monitors)
        self.trial_sim = trial_sim

    def run(self):
        ### start each run identical
        set_seed(params.CHECK_LEARNING_SEED)
        params.CHECK_LEARNING_MYRNG.reset()
        self.reset(populations=True, projections=True, synapses=True, model_state=True)
        set_seed(params.CHECK_LEARNING_SEED)
        self.trial_sim.sc_input = True
        self.trial_sim.sc_all = False
        self.trial_sim.p_input = params.CHECK_LEARNING_P_INPUT

        if params.CHECK_LEARNING_N_TRIALS == 1:
            ### start all recordings for single trial
            if do_monitor:
                self.monitors.start()
        else:
            ### only start recording of pf for multiple trials (others start later)
            if do_monitor:
                self.monitors.start(compartment_list=["pf"])

        ### record cor__stn weights, cue and target sides and times for all trials
        weight_cor_stn_arr = np.empty(
            (params.CHECK_LEARNING_N_TRIALS, 2, 1 * params.N_COR_SEQUENCE)
        )
        weight_stn_snr_arr = np.empty((params.CHECK_LEARNING_N_TRIALS, 2, 1))
        outcome_side_arr = np.empty(params.CHECK_LEARNING_N_TRIALS)
        outcome_delay_arr = np.empty(params.CHECK_LEARNING_N_TRIALS)
        cor_input_arr = np.empty(params.CHECK_LEARNING_N_TRIALS)

        ### loop over trials
        for trial_idx in tqdm(range(params.CHECK_LEARNING_N_TRIALS)):
            ### before last trial start all recordings and deactivate sc input to check
            ### if cor-stn learning had an effect
            if (
                params.CHECK_LEARNING_N_TRIALS > 1
                and trial_idx == params.CHECK_LEARNING_N_TRIALS - 1
            ):
                if do_monitor:
                    self.monitors.start()
                if not params.CHECK_LEARNING_SC_LAST_TRIAL:
                    self.trial_sim.sc_input = False
            ### deactivate sc input after half of trials
            if (
                trial_idx == params.CHECK_LEARNING_N_TRIALS // 2
                and not params.CHECK_LEARNING_SC_SECOND_HALF
            ):
                self.trial_sim.sc_input = False
            ### activate all sc for last trial and definitely activate input and use
            ### fastest target
            if trial_idx == params.CHECK_LEARNING_N_TRIALS - 1:
                self.trial_sim.sc_all = True
                self.trial_sim.p_input = 1.0
                self.trial_sim.fast_target = True
            ### deactivate sc input
            if not params.CHECK_LEARNING_SC:
                self.trial_sim.sc_input = False
            ### reset time
            self.reset(
                populations=False, projections=False, synapses=False, model_state=True
            )
            ### run trial simulation
            self.trial_sim.run()
            ### get side of outcome
            outcome_side_arr[trial_idx] = self.trial_sim.outcome_side
            outcome_delay_arr[trial_idx] = self.trial_sim.delay_input_outcome
            ### get whether cor input was active
            cor_input_arr[trial_idx] = self.trial_sim.cor_input
            ### get always the exact same trial simulation (does not affect model, only the random simulation)
            # params.CHECK_LEARNING_MYRNG.reset()
            ### store weights
            weight_cor_stn_arr[trial_idx] = get_projection("cor__stn").w
            weight_stn_snr_arr[trial_idx] = get_projection("stn__snr").w

        self.data["weight_cor_stn_arr"] = weight_cor_stn_arr
        self.data["weight_stn_snr_arr"] = weight_stn_snr_arr
        self.data["outcome_side_arr"] = outcome_side_arr
        self.data["outcome_delay_arr"] = outcome_delay_arr
        self.data["cor_input_arr"] = cor_input_arr

        return self.results()


if __name__ == "__main__":
    ### prepare model
    monitors = prepare_model()

    ### create the simulation object for simulating a trial
    input_outcome_offset = params.CHECK_LEARNING_T_OFFSET_INPUT_OUTCOME
    trial_sim = TrialCheckLearning(
        rng=params.CHECK_LEARNING_MYRNG.rng,
        p=prob,
        input_outcome_offset=input_outcome_offset,
        verbose=False,
        p_input=params.CHECK_LEARNING_P_INPUT,
    )

    ### create the experiment object for simulating multiple trials
    my_exp = MyExp(monitors=monitors, trial_sim=trial_sim)

    ### run the simulation

    ### ramp up simulation to get to a stable state
    simulate(params.CHECK_LEARNING_T_RAMP_UP)
    ### store state for experiment
    my_exp.store_model_state(
        prediction_model.populations + prediction_model.projections
    )
    ### run experiment
    results = my_exp.run()

    ### store data
    if sim_idx == 0:
        save_variables(
            variable_list=[
                results.recordings,
                results.recording_times,
                results.data["weight_cor_stn_arr"],
                results.data["weight_stn_snr_arr"],
                results.data["outcome_side_arr"],
                results.data["outcome_delay_arr"],
                results.data["cor_input_arr"],
                prediction_model.populations,
            ],
            name_list=[
                f"recordings_{save_str}",
                f"recording_times_{save_str}",
                f"weight_cor_stn_arr_{save_str}",
                f"weight_stn_snr_arr_{save_str}",
                f"outcome_side_arr_{save_str}",
                f"outcome_delay_arr_{save_str}",
                f"cor_input_arr_{save_str}",
                f"population_name_list_{save_str}",
            ],
            path=params.CHECK_LEARNING_SAVE_FOLDER,
        )
    elif sim_idx == 1:
        save_variables(
            variable_list=[
                results.data["weight_cor_stn_arr"],
            ],
            name_list=[
                f"weight_cor_stn_arr_{save_str}",
            ],
            path=params.CHECK_LEARNING_SAVE_FOLDER,
        )
