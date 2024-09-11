from CompNeuroPy import RNG
import numpy as np
from ANNarchy import Population, Neuron, Projection, Synapse, Constant
import math
import matplotlib.pyplot as plt


### neuron types i.e. population types
class PopulationSigmoid(Population):
    """
    Type of Population with neuron model using sigmoid activation function.
    Parameters I_0, I_12, and tau must be set!
    """

    neuron_sigmoid = Neuron(
        parameters="""
        ### input and rate
        slope = 0.0 : population
        I_0 = 0.0 : population
        tau = 10.0 : population
        I_ext = 0.0
        ### noise
        noise = 0.0 : population
        tau_power = 10.0 : population
        snr_target = 10.0 : population
        ### homeostasis
        homeostasis = 0.0 : population
        r_homeostasis = 0.0 : population
        tau_homeostasis = 1.0 : population
        ### r trace
        tau_r_trace = 1.0 : population
        """,
        equations="""
        ### plasticity modulating input from pf (only relevant for stn and snr)
        learn_mod = sum(pf_input)
        mod_increase = sign(learn_mod - pf_r_homeostasis)

        ### input, noise, and membrane potential
        I_signal = pos(sum(exc) - sum(inh)) + I_ext - I_homeostasis
        I_noise = noise*Normal(0, 1)
        ### scale noise to reach target snr, scale factor is:
        ### scaling_factor = sqrt((power_I_signal/power_I_noise)/snr_target)
        ### since power of N(0,1) is 1, we can scale the noise by:
        ### scaling_factor = sqrt(power_I_signal/snr_target)
        I = I_signal + I_noise * sqrt(power_I_signal/snr_target)
        tau*dmp/dt = 2*(1/(1/exp(slope*(I - I_0)) + 1) - 0.5) - mp

        ### rate
        r = pos(mp)

        ### power of signal (relevant for noise)
        tau_power * dpower_I_signal/dt = I_signal**2 - power_I_signal

        ### homeostasis
        tau_homeostasis * dI_homeostasis/dt = homeostasis*(mp - r_homeostasis) : min = 0

        ### r trace and its increase
        tau_r_trace * dr_trace/dt = r - r_trace
        r_trace_increase_event = r_trace > 0.8
        r_trace_increase_event_low = r_trace > 0.4
        max_reached = 1.0 - r
        """,
    )

    def __init__(
        self,
        geometry,
        name=None,
        stop_condition=None,
        storage_order="post_to_pre",
        copied=False,
    ):
        ### annotate types for automatically added attributes

        ### parameters
        ### input and rate
        self.slope: float
        self.I_0: float
        self.tau: float
        self.I_ext: np.ndarray
        ### noise
        self.noise: float
        self.tau_power: float
        self.snr_target: float
        ### homeostasis
        self.homeostasis: float
        self.r_homeostasis: float
        self.tau_homeostasis: float
        ### r trace
        self.tau_r_trace: float

        ### variables
        ### plasticity modulating input from pf (only relevant for stn and snr)
        self.learn_mod: np.ndarray
        self.mod_increase: np.ndarray
        ### input, noise, and membrane potential
        self.I_signal: np.ndarray
        self.I_noise: np.ndarray
        self.I: np.ndarray
        self.mp: np.ndarray
        ### rate and its mean and variance
        self.r: np.ndarray
        ### power of signal (relevant for noise)
        self.power_I_signal: np.ndarray
        ### homeostasis
        self.I_homeostasis: np.ndarray
        ### r trace and its increase
        self.r_trace: np.ndarray
        self.r_trace_increase_event: np.ndarray
        self.r_trace_increase_event_low: np.ndarray
        self.max_reached: np.ndarray

        super().__init__(
            geometry, self.neuron_sigmoid, name, stop_condition, storage_order, copied
        )

        ### initialize slope by setting I_12 (I_0 already set in super().__init__)
        self.I_12 = 0.5

    def __setattr__(self, name, value):
        """
        Set attribute and calculate slope if I_0 or I_12 is set.
        Cannot use @property for I_0 and I_12 because of the way Population is
        implemented (I_0 is set in the __init__ method of Population).
        """
        if name == "I_0" or name == "I_12":
            ### set attr
            super().__setattr__(name, value)
            ### calculate slope and set it too
            super().__setattr__(
                "slope", self._calculate_slope(I_0=self.I_0, I_12=self.I_12)
            )
        else:
            super().__setattr__(name, value)

    def set_params(
        self,
        I_0: float | None = None,
        I_12: float | None = None,
        tau: float | None = None,
        I_ext: float | None = None,
        noise: float | None = None,
        tau_power: float | None = None,
        snr_target: float | None = None,
        homeostasis: float | None = None,
        r_homeostasis: float | None = None,
        tau_homeostasis: float | None = None,
        tau_r_trace: float | None = None,
    ):
        """
        Set the parameters of the population.

        Args:
            I_0 (float, optional):
                The input for which the rate of the neuron is 0. Default is 0.0.
            I_12 (float, optional):
                The input for which the rate of the neuron is 0.5. Default is 0.5.
            tau (float, optional):
                The time constant of the neuron. Default is 10.0.
            I_ext (float, optional):
                The external input to the neuron. Default is 0.0.
            noise (float, optional):
                Set it to 0 to deactivate noise or to 1 to let the SNR reach the target.
                Default is 0.0.
            tau_power (float, optional):
                Time constant of the power of the signal. Default is 10.0.
            snr_target (float, optional):
                Target SNR of the neuron inputs. Default is 10.0.
            homeostasis (float, optional):
                Set it to 0 to deactivate homeostasis or to 1 to activate it. Default is
                0.0.
            r_homeostasis (float, optional):
                Target rate (i.e. membrane potential) for homeostasis. Default is 0.0.
            tau_homeostasis (float, optional):
                Time constant of the homeostasis input. Default is 1.0.
            tau_r_trace (float, optional):
                Time constant of the trace of the rate. Default is 1.
        """
        ### set I_0
        if I_0 is not None:
            self.I_0 = I_0
        ### set I_12
        if I_12 is not None:
            self.I_12 = I_12
        ### set tau
        if tau is not None:
            self.tau = tau
        ### set I_ext
        if I_ext is not None:
            self.I_ext = I_ext
        ### set noise
        if noise is not None:
            self.noise = noise
        ### set tau_power
        if tau_power is not None:
            self.tau_power = tau_power
        ### set snr_target
        if snr_target is not None:
            self.snr_target = snr_target
        ### set homeostasis
        if homeostasis is not None:
            self.homeostasis = homeostasis
        ### set r_homeostasis
        if r_homeostasis is not None:
            self.r_homeostasis = r_homeostasis
        ### set tau_homeostasis
        if tau_homeostasis is not None:
            self.tau_homeostasis = tau_homeostasis
        ### set tau_r_trace
        if tau_r_trace is not None:
            self.tau_r_trace = tau_r_trace

    @staticmethod
    def _calculate_slope(I_0: float, I_12: float) -> float:
        """
        Calculate the slope of the sigmoid function based on I_0 and I_12.
        """
        ### make sure I_12 is larger than I_0
        I_12 = max(I_12, I_0 + 1e-6)
        return math.log10(3) / (
            math.log10(math.exp(1)) * I_12 - math.log10(math.exp(1)) * I_0
        )

    @staticmethod
    def r_of_I(I: float | np.ndarray, I_0: float, I_12: float) -> float | np.ndarray:
        """
        Calculate the rate of the neuron based on the input I.

        Args:
            I (float or np.ndarray):
                The input to the neuron.
            I_0 (float):
                The input for which the rate of the neuron is 0.
            I_12 (float):
                The input for which the rate of the neuron is 0.5.

        Returns:
            float or np.ndarray:
                The rate of the neuron.
        """
        slope = PopulationSigmoid._calculate_slope(I_0, I_12)
        return 2 * (1 / (1 / np.exp(slope * (I - I_0)) + 1) - 0.5)

    def plot_response_curve(self):
        """
        Plot the response curve of the sigmoid neuron.
        """
        print(f"Plotting response curve of {self.name}...")
        start = self.I_0 - (self.I_12 - self.I_0)
        end = self.I_12 + (self.I_12 - self.I_0)
        i_arr = np.linspace(start, end, 100)
        r_arr = self.r_of_I(i_arr, self.I_0, self.I_12)

        plt.close("all")
        plt.figure(figsize=(6.4 * 2, 4.8 * 2))
        plt.plot(i_arr, r_arr)
        plt.axvline(0, color="black", linestyle="--")
        plt.xlim(start, end)
        plt.ylim(0, 1)
        plt.xticks(np.arange(start, end + 0.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid(True)
        plt.xlabel("I")
        plt.ylabel("r")
        plt.tight_layout()
        plt.title(f"Response Curve {self.name}")
        plt.savefig(f"response_curve_{self.name}.png", dpi=300)


class PopulationSigmoidCortex(Population):
    """
    Type of Population with neuron model using sigmoid activation function.
    Parameters I_0, I_12, and tau must be set! Parameter slope is calculated based on
    I_0 and I_12.
    """

    neuron_sigmoid = Neuron(
        parameters="""
        ### input and rate
        slope = 0.0 : population
        I_0 = 0.0 : population
        tau = 10.0 : population
        I_ext = 0.0
        ### r trace
        tau_r_trace = 1.0 : population
        """,
        equations="""
        ### input
        I_signal = sum(exc) - sum(inh) + I_ext

        ### rate
        tau*dr/dt = pos(2*(1/(1/exp(slope*(I_signal - I_0)) + 1) - 0.5)) - r

        ### r trace and its increase
        tau_r_trace * dr_trace/dt = r - r_trace
        r_trace_increase_event = r_trace > 0.8
        """,
    )

    def __init__(
        self,
        geometry,
        name=None,
        stop_condition=None,
        storage_order="post_to_pre",
        copied=False,
    ):
        ### annotate types for automatically added attributes

        ### parameters
        ### input and rate
        self.slope: float
        self.I_0: float
        self.tau: float
        self.I_ext: np.ndarray
        ### r trace
        self.tau_r_trace: float

        ### variables
        ### input
        self.I_signal: np.ndarray
        ### rate
        self.r: np.ndarray
        ### r trace and its increase
        self.r_trace: np.ndarray
        self.r_trace_increase: np.ndarray

        super().__init__(
            geometry, self.neuron_sigmoid, name, stop_condition, storage_order, copied
        )

        ### initialize slope by setting I_12 (I_0 already set in super().__init__)
        self.I_12 = 0.5

    def __setattr__(self, name, value):
        """
        Set attribute and calculate slope if I_0 or I_12 is set.
        Cannot use @property for I_0 and I_12 because of the way Population is
        implemented (I_0 is set in the __init__ method of Population).
        """
        if name == "I_0" or name == "I_12":
            ### set attr
            super().__setattr__(name, value)
            ### calculate slope and set it too
            super().__setattr__("slope", self._calculate_slope())
        else:
            super().__setattr__(name, value)

    def set_params(
        self,
        I_0: float | None = None,
        I_12: float | None = None,
        tau: float | None = None,
        I_ext: float | None = None,
        tau_r_trace: float | None = None,
    ):
        """
        Set the parameters of the population.

        Args:
            I_0 (float, optional):
                The input for which the rate of the neuron is 0. Default is 0.0.
            I_12 (float, optional):
                The input for which the rate of the neuron is 0.5. Default is 0.5.
            tau (float, optional):
                The time constant of the neuron. Default is 10.0.
            I_ext (float, optional):
                The external input to the neuron. Default is 0.0.
            tau_r_trace (float, optional):
                Time constant of the trace of the rate. Default is 1.
        """
        ### set I_0
        if I_0 is not None:
            self.I_0 = I_0
        ### set I_12
        if I_12 is not None:
            self.I_12 = I_12
        ### set tau
        if tau is not None:
            self.tau = tau
        ### set I_ext
        if I_ext is not None:
            self.I_ext = I_ext
        ### set tau_r_trace
        if tau_r_trace is not None:
            self.tau_r_trace = tau_r_trace

    def _calculate_slope(self):
        """
        Calculate the slope of the sigmoid function based on I_0 and I_12.
        """
        I_0 = self.I_0
        ### make sure I_12 is larger than I_0
        I_12 = max(self.I_12, I_0 + 1e-6)
        return math.log10(3) / (
            math.log10(math.exp(1)) * I_12 - math.log10(math.exp(1)) * I_0
        )

    def plot_response_curve(self):
        """
        Plot the response curve of the sigmoid neuron.
        """
        print(f"Plotting response curve of {self.name}...")
        start = self.I_0 - (self.I_12 - self.I_0)
        end = self.I_12 + (self.I_12 - self.I_0)
        i_arr = np.linspace(start, end, 100)
        r_arr = np.clip(
            2 * (1 / (1 / np.exp(self.slope * (i_arr - self.I_0)) + 1) - 0.5), 0, None
        )

        plt.close("all")
        plt.figure(figsize=(6.4 * 2, 4.8 * 2))
        plt.plot(i_arr, r_arr)
        plt.axvline(0, color="black", linestyle="--")
        plt.xlim(start, end)
        plt.ylim(0, 1)
        plt.xticks(np.arange(start, end + 0.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid(True)
        plt.xlabel("I")
        plt.ylabel("r")
        plt.tight_layout()
        plt.title(f"Response Curve {self.name}")
        plt.savefig(f"response_curve_{self.name}.png", dpi=300)


class PopulationInput(Population):
    """
    Type of Population with neuron model using input I_ext as rate.
    """

    neuron_input = Neuron(
        parameters="""
        I_ext = 0
        """,
        equations="""
        I = I_ext
        r = I
        """,
    )

    def __init__(
        self,
        geometry,
        name=None,
        stop_condition=None,
        storage_order="post_to_pre",
        copied=False,
    ):
        ### annotate types for automatically added attributes
        ### parameters
        self.I_ext: np.ndarray
        ### variables
        self.I: np.ndarray
        self.r: np.ndarray

        super().__init__(
            geometry, self.neuron_input, name, stop_condition, storage_order, copied
        )

    def set_params(self, I_ext: float | np.ndarray | None = None):
        """
        Set the parameters of the population.

        Args:
            I_ext (float or np.ndarray, optional):
                The external input i.e. the rate of the neuron. Default is 0.
        """
        ### set I_ext
        if I_ext is not None:
            self.I_ext = I_ext


### synapse types i.e. projection types
class ProjectionLearning(Projection):
    """
    Type of Projection with learning synapse. Learning rule is still to be implemented.
    """

    learning_synapse = Synapse(
        parameters="""
        tau = 1.0 : projection
        alpha = 1.0 : projection
        """,
        equations="""
        tau * dw/dt = pre.r_trace_increase_event*post.max_reached*(post.mod_increase - alpha)     : min = 0
        """,
    )

    def __init__(
        self,
        pre,
        post,
        target,
        name=None,
        disable_omp=True,
        copied=False,
    ):
        ### annotate types for automatically added attributes
        ### parameters
        self.tau: float
        self.alpha: float
        ### variables
        self.w: np.ndarray

        super().__init__(
            pre=pre,
            post=post,
            target=target,
            synapse=self.learning_synapse,
            name=name,
            disable_omp=disable_omp,
            copied=copied,
        )

    def set_params(
        self,
        tau: float | None = None,
        alpha: float | None = None,
    ):
        """
        Set the parameters of the projection.

        Args:
            tau (float, optional):
                The time constant of the synapse (i.e. how fast the weight changes).
                Default is 10.0.
            alpha (float, optional):
                The factor to weight the learning rule. Weights increase if pre activity
                is increased and post modulation is increased. Without increased post
                modulation, the weights decrease. Alpha weights the decrease. Alpha = 1
                means that the post modulation increase would need to be present all the
                time to archeive dw/dt=0. If alpha is zero, the weights would only
                increase. In between, the weights would increase if the post modulation
                increase is present "often enough". Default is 1.0.
        """
        ### set tau
        if tau is not None:
            self.tau = tau
        ### set alpha
        if alpha is not None:
            self.alpha = alpha


### synapse types i.e. projection types
class ProjectionLearningSTNSNr(Projection):
    """
    Type of Projection with learning synapse. Learning rule is still to be implemented.
    """

    learning_synapse = Synapse(
        parameters="""
        tau = 1.0 : projection
        alpha = 1.0 : projection
        """,
        equations="""
        ### if cor-stn decreases, pre_event does not happen anymore, thus this cannot unlearn, therfore for unlearn use lower threshold
        tau * dw/dt = pre.r_trace_increase_event*post.max_reached*post.mod_increase - pre.r_trace_increase_event_low*post.max_reached*alpha  : min = 0
        """,
    )

    def __init__(
        self,
        pre,
        post,
        target,
        name=None,
        disable_omp=True,
        copied=False,
    ):
        ### annotate types for automatically added attributes
        ### parameters
        self.tau: float
        self.alpha: float
        ### variables
        self.w: np.ndarray

        super().__init__(
            pre=pre,
            post=post,
            target=target,
            synapse=self.learning_synapse,
            name=name,
            disable_omp=disable_omp,
            copied=copied,
        )

    def set_params(
        self,
        tau: float | None = None,
        alpha: float | None = None,
    ):
        """
        Set the parameters of the projection.

        Args:
            tau (float, optional):
                The time constant of the synapse (i.e. how fast the weight changes).
                Default is 10.0.
            alpha (float, optional):
                The factor to weight the learning rule. Weights increase if pre activity
                is increased and post modulation is increased. Without increased post
                modulation, the weights decrease. Alpha weights the decrease. Alpha = 1
                means that the post modulation increase would need to be present all the
                time to archeive dw/dt=0. If alpha is zero, the weights would only
                increase. In between, the weights would increase if the post modulation
                increase is present "often enough". Default is 1.0.
        """
        ### set tau
        if tau is not None:
            self.tau = tau
        ### set alpha
        if alpha is not None:
            self.alpha = alpha


################################### MODEL PARAMETERS ###################################
### parameters
N_STATE = 3
N_OUT = 3
N_COR_SEQUENCE = 50
PLOT_RESPONSE_CURVES = False

### populations
PF_BASE = 0.2
PF_I_0 = 0.0
PF_I_12 = 0.5
STN_BASE = 0.2
STN_I_0 = 0.0
STN_I_12 = 0.5
SNR_BASE = 0.35
SNR_I_0 = 0.0
SNR_I_12 = 0.5
PF_R_HOMEOSTASIS = Constant(
    "pf_r_homeostasis", PopulationSigmoid.r_of_I(PF_BASE, PF_I_0, PF_I_12)
)
STN_R_HOMEOSTASIS = PopulationSigmoid.r_of_I(STN_BASE, STN_I_0, STN_I_12)
SNR_R_HOMEOSTASIS = PopulationSigmoid.r_of_I(SNR_BASE, SNR_I_0, SNR_I_12)

POP_PARAMS = {
    "cor": [
        None,  # I_0
        None,  # I_12
        None,  # tau
        None,  # I_ext
        20.0,  # tau_r_trace
    ],
    "cor_aux": [
        None,  # I_0
        0.3,  # I_12
        None,  # tau
        None,  # I_ext
        None,  # tau_r_trace
    ],
    "pf": [
        PF_I_0,  # I_0
        PF_I_12,  # I_12
        None,  # tau
        PF_BASE,  # I_ext
        1,  # noise
        None,  # tau_power
        None,  # snr_target
        None,  # homeostasis
        PF_R_HOMEOSTASIS,  # r_homeostasis
        None,  # tau_homeostasis
        None,  # tau_r_trace
    ],
    "snr": [
        SNR_I_0,  # I_0
        SNR_I_12,  # I_12
        None,  # tau
        SNR_BASE,  # I_ext
        1,  # noise
        None,  # tau_power
        None,  # snr_target
        1,  # homeostasis
        SNR_R_HOMEOSTASIS,  # r_homeostasis
        40000,  # tau_homeostasis
        None,  # tau_r_trace
    ],
    "stn": [
        STN_I_0,  # I_0
        STN_I_12,  # I_12
        None,  # tau
        STN_BASE,  # I_ext
        1,  # noise
        None,  # tau_power
        None,  # snr_target
        None,  # homeostasis
        STN_R_HOMEOSTASIS,  # r_homeostasis
        None,  # tau_homeostasis
        20,  # tau_r_trace
    ],
}

### initial weights of projections


def get_input_weights_pf(
    I_pf_pre_learn=1.5,
    I_pf_post_learn=PF_BASE,
    r_snr_pre_learn=SNR_R_HOMEOSTASIS,
    r_snr_post_learn=0.80,
):
    """
    Calculate the input weights of the Pf from SC and SNr.

    Args:
        I_pf_pre_learn (float):
            Input to the Pf population before learning (should be SC input + baseline Pf
            - baseline input SNr).
        I_pf_post_learn (float):
            Input to the Pf population after learning (should be SC input + baseline Pf
            - increased input SNr).
        r_snr_pre_learn (float):
            Rate of the SNr population before learning. (should be baseline SNr rate)
        r_snr_post_learn (float):
            Rate of the SNr population after learning. (should be increased SNr rate)

    Returns:
        dict: Dictionary with the input weights of the Pf population from SC and SNr.
    """

    w_snr__pf__inh = (I_pf_pre_learn - I_pf_post_learn) / (
        r_snr_post_learn - r_snr_pre_learn
    )
    w_sc__pf__exc = I_pf_pre_learn - PF_BASE + r_snr_pre_learn * w_snr__pf__inh

    return {
        "snr__pf__inh": w_snr__pf__inh,
        "sc__pf__exc": w_sc__pf__exc,
    }


input_weights_pf = get_input_weights_pf()


WEIGHTS = {
    ### within cortex
    "cor__cor__self_exc": 5.371289580626674,
    "cor__cor__sequence_prop_exc": 4.027674110366309,
    "cor__cor__sequence_prop_forward_inh": 3.320253584069181,
    "cor__cor__sequence_prop_backward_inh": 5.714659679147405,
    "cor__cor__sequence_end": 20.0,
    "cor_aux__cor": 20.0,
    ### plastic cor-stn
    "cor__stn": 0.0,
    ### plastic stn-snr
    "stn__snr": 0.0,
    ### default connections
    "sc__pf__exc": input_weights_pf["sc__pf__exc"],
    # "pf__stn__exc": 1.0,
    "pf__stn__pf_input": 1.0,
    # "pf__snr__exc": 0.1,
    "pf__snr__pf_input": 1.0,
    "snr__pf__inh": input_weights_pf["snr__pf__inh"],
}
### cor sequence propagation
COR_INPUT_AMPLITUDE = 1.0
COR_COR_SIGMA_EXC = 0.7314968637408101
COR_COR_SIGMA_FORWARD_INH = 1.7518487158431617
COR_COR_SIGMA_BACKWARD_INH = 3.115515977168838
COR_SEQUENCE_DELAYS = 20
COR_SEQUENCE_CIRCULAR = True
FORWARD_INHIBITION = True
BACKWARD_INHIBITION = True
### learning rule parameters (other important paramters are r_std_thresh and
### mod_std_thresh of pre and post synaptic populations)
COR_STN_TAU = 7500
COR_STN_ALPHA = 0.03
STN_SNR_TAU = 250
STN_SNR_ALPHA = COR_STN_ALPHA * 0.17

################################### SCRIPT PARAMETERS ##################################
### number workers for parallel scripts
N_JOBS = 2

################################ SIMULATION PARAMETERS #################################
### run_simulation script
# timings
SIMULATION_N_TRIALS = 3000
SIMULATION_T_RAMP_UP = 2000
# cue light 0.5-1.5s after hold light
SIMULATION_T_OFFSET_HOLD_CUE = [500, 1500]
# target light 0.1, 0.4 or 0.7s after cue light
SIMULATION_T_OFFSET_CUE_TARGET = [100, 400, 700]
# inter trial interval 3-5s
SIMULATION_T_OFFSET_TARGET_END = [3000, 5000]
# stimulus onset - cortex (state) activation delay
SIMULATION_T_COR_STATE_DELAY = 50
# how long are input populations activated
SIMULATION_T_INPUT_COR = 200
SIMULATION_T_INPUT_SC = 50
# how the model should encode the task
SIMULATION_HOLD_RESPONSE = True
SIMULATION_HOLD_STATE = True
SIMULATION_CUE_RESPONSE = True
SIMULATION_CUE_STATE = True
SIMULATION_ONLY_LEFT_CUE = False
# rng
SIMULATION_SEED = 420
SIMULATION_MYRNG = RNG(SIMULATION_SEED)
# general
SIMULATION_SAVE_FOLDER = "simulation_data"
SIMULATION_SIMULATE = True
SIMULATION_OPTIMIZE = False
SIMULATION_N_TOTAL = 23
SIMULATION_MAX_EVALS = 1000
SIMULATION_ANALYZE = True
# int or "all" in which case all simulations (N_TOTAL) are analyzed
SIMULATION_ID_ANALYZE = 0
SIMULATION_PF_RESPONSE_PLOT = True
SIMULATION_SINGLE_TRIAL_PLOT = True
SIMULATION_WEIGHT_CHANGE_PLOT = False
SIMULATION_CUE_TARGET_STATS_PLOT = False
# plots properties
SIMULATION_PF_RESPONSE_PLOT_PROPERTIES = {
    "fontsize": 12,
    "figsize": (20 / 2.54, 15 / 2.54),
    "grid": False,
}
SIMULATION_COR_AND_RESPONSES_PLOT_PROPERTIES = {
    "fontsize": 12,
    "figsize": (12 / 2.54, 20 / 2.54),
    "grid": False,
    "y_label_pad": 30,
    "text_position": 205,
    "h_pad": 0.3,
    "rowspan_first": 3,
    "rowspan_others": 1,
}
### run_investigate_cortex_parallel script
INVESTIGATE_CORTEX_SAVE_FOLDER = "investigate_cortex_data_2"
INVESTIGATE_CORTEX_T_TOTAL = 1700
INVESTIGATE_CORTEX_MAX_EVALS = 450
INVESTIGATE_CORTEX_OPTIMIZE = False
INVESTIGATE_CORTEX_N_TOTAL = 350
### check_learning script
CHECK_LEARNING_SAVE_FOLDER = "check_learning_data"
CHECK_LEARNING_SIMULATE = True
CHECK_LEARNING_ANALYZE = True
CHECK_LEARNING_SIM_IDX = 0
CHECK_LEARNING_SIM_IDX_ANALYZE = 0
CHECK_LEARNING_ALPHA_STEPS = 20
CHECK_LEARNING_PROB_STEPS = 20
CHECK_LEARNING_SC = True
CHECK_LEARNING_SC_SECOND_HALF = True
CHECK_LEARNING_SC_LAST_TRIAL = True
CHECK_LEARNING_RANDOM_DELAY = True
CHECK_LEARNING_N_TRIALS = 3000
CHECK_LEARNING_SEED = 1
CHECK_LEARNING_MYRNG = RNG(CHECK_LEARNING_SEED)
CHECK_LEARNING_T_RAMP_UP = 2000
CHECK_LEARNING_SINGLE_TRIAL_PLOT = True
CHECK_LEARNING_WEIGHT_CHANGE_PLOT = True
CHECK_LEARNING_CUE_TARGET_STATS_PLOT = True
CHECK_LEARNING_PF_RESPONSE_PLOT = True
CHECK_LEARNING_T_COR_STATE_DELAY = 50
CHECK_LEARNING_T_INPUT_COR = 200
CHECK_LEARNING_T_INPUT_SC = 50
CHECK_LEARNING_T_OFFSET_INPUT_END = 3500
CHECK_LEARNING_T_OFFSET_INPUT_OUTCOME = [100, 400, 700]
CHECK_LEARNING_P_INPUT = 0.5
