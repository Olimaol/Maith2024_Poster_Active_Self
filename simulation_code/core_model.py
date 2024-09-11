from CompNeuroPy import (
    CompNeuroModel,
)
from core_parameters import (
    PopulationInput,
    PopulationSigmoid,
    PopulationSigmoidCortex,
    N_STATE,
    N_OUT,
    N_COR_SEQUENCE,
    POP_PARAMS,
    PLOT_RESPONSE_CURVES,
    COR_STN_TAU,
    COR_STN_ALPHA,
    STN_SNR_TAU,
    STN_SNR_ALPHA,
    WEIGHTS,
)
from core_helping_functions import (
    connect_cor_within,
    connect_cor__stn,
    connect_default,
    connect_stn__snr,
)

### TODO remove cor_aux, states are restted by simply setting rates to 0 before setting input current


def create_model(
    n_state=N_STATE,
    n_out=N_OUT,
    cor_stn_tau=COR_STN_TAU,
    cor_stn_alpha=COR_STN_ALPHA,
    cor_stn_w_init=WEIGHTS["cor__stn"],
    stn_snr_tau=STN_SNR_TAU,
    stn_snr_alpha=STN_SNR_ALPHA,
    stn_snr_w_init=WEIGHTS["stn__snr"],
    weights=WEIGHTS,
):
    """
    Create the model. Create all populations and projections and set their parameters.
    Cortex should store states as sequences like in [Parker et al. (2022)](https://doi.org/10.1016/j.celrep.2022.110756)

    !!!warning:
        This model creation function overrides some global parameters.

    Args:
        n_state (int):
            Number of states in the cortex.
        n_out (int):
            Number of outcomes in the model.
        cor_stn_tau (float):
            Time constant for the connection from the cortex to the STN.
        cor_stn_alpha (float):
            Alpha parameter for the connection from the cortex to the STN.
        stn_snr_tau (float):
            Time constant for the connection from the STN to the SNr.
        stn_snr_alpha (float):
            Alpha parameter for the connection from the STN to the SNr.
        weights (dict):
            Dictionary with the weights for the fixed default connections. Keys are the
            connection names in format pre__post__target.
            By default the global weights are used. If weights are given only these will
            differ from the global weights.
    """
    ### create populations
    ### input populations
    cor = PopulationSigmoidCortex(geometry=(n_state, N_COR_SEQUENCE), name="cor")
    cor_aux = PopulationSigmoidCortex(geometry=n_state, name="cor_aux")
    sc = PopulationInput(geometry=n_out, name="sc")
    ### other populations
    pf = PopulationSigmoid(geometry=n_out, name="pf")
    snr = PopulationSigmoid(geometry=n_out, name="snr")
    stn = PopulationSigmoid(geometry=n_out, name="stn")

    ### set (initial) parameters
    cor.set_params(*POP_PARAMS["cor"])
    cor_aux.set_params(*POP_PARAMS["cor_aux"])
    pf.set_params(*POP_PARAMS["pf"])
    snr.set_params(*POP_PARAMS["snr"])
    stn.set_params(*POP_PARAMS["stn"])

    ### plot response curves with the set parameters
    if PLOT_RESPONSE_CURVES:
        cor.plot_response_curve()
        cor_aux.plot_response_curve()
        pf.plot_response_curve()
        snr.plot_response_curve()
        stn.plot_response_curve()

    ### connect populations
    # cor within
    connect_cor_within(cor, cor_aux, **{"n_state": n_state})
    # plastic
    connect_cor__stn(
        cor,
        stn,
        tau=cor_stn_tau,
        alpha=cor_stn_alpha,
        w_init=cor_stn_w_init,
    )
    connect_stn__snr(
        stn,
        snr,
        tau=stn_snr_tau,
        alpha=stn_snr_alpha,
        w_init=stn_snr_w_init,
    )
    # fixed defaults
    connect_default(sc, pf, **weights)
    connect_default(pf, stn, target="pf_input", **weights)
    connect_default(pf, snr, target="pf_input", **weights)
    connect_default(snr, pf, target="inh", **weights)


def create_model_only_cortex(
    weights,
    n_state,
    n_cor_sequence,
    cor_sequence_delays,
    cor_cor_sigma_exc,
    cor_cor_sigma_forward_inh,
    cor_cor_sigma_backward_inh,
):
    """
    Create only the Cortex of the model with its inner connections.
    Cortex should store states as sequences like in [Parker et al. (2022)](https://doi.org/10.1016/j.celrep.2022.110756)

    !!!warning:
        This model creation function overrides some global parameters like N_STATE,
        N_COR_SEQUENCE and the parameters related to the cortex connections!

    Args:
        weights (dict):
            Dictionary with the weights for the connections. With the keys:
            - cor__cor__self_exc
            - cor__cor__sequence_prop_exc
            - cor__cor__sequence_prop_forward_inh
            - cor__cor__sequence_prop_backward_inh
        n_state (int):
            Number of states in the cortex.
        n_cor_sequence (int):
            Number of cor neurons encoding a state as a sequence.
        cor_sequence_delays (float):
            Delay in ms of the sequence propagation in the cortex.
        cor_cor_sigma_exc (float):
            Sigma of the gaussian for the excitatory connections within the cortex
            sequences.
        cor_cor_sigma_forward_inh (float):
            Sigma of the gaussian for the forward inhibitory connections within the
            cortex sequences.
        cor_cor_sigma_backward_inh (float):
            Sigma of the gaussian for the backward inhibitory connections within the
            cortex sequences.
    """
    ### create populations
    cor = PopulationSigmoid(geometry=(n_state, n_cor_sequence), name="cor")

    ### set (initial) parameters
    cor.set_params(*POP_PARAMS["cor"])

    ### connect populations
    connect_cor_within(
        cor,
        **{
            "weights": weights,
            "n_state": n_state,
            "n_cor_sequence": n_cor_sequence,
            "cor_sequence_delays": cor_sequence_delays,
            "cor_cor_sigma_exc": cor_cor_sigma_exc,
            "cor_cor_sigma_forward_inh": cor_cor_sigma_forward_inh,
            "cor_cor_sigma_backward_inh": cor_cor_sigma_backward_inh,
        }
    )


prediction_model = CompNeuroModel(
    model_creation_function=create_model,
    name="BG prediction model",
    description="Model of STN-SNr-PF network for learning to predict an outcome encoded in the SC population.",
    do_create=False,
    do_compile=False,
    compile_folder_name="bg_prediction_model",
)

cortex_model = CompNeuroModel(
    model_creation_function=create_model_only_cortex,
    name="Cortex model",
    description="Model of the cortex network for storing states.",
    do_create=False,
    do_compile=False,
    compile_folder_name="cortex_model",
)
