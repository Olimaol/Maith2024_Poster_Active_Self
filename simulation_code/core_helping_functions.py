from ANNarchy import Projection, Population
import numpy as np
from core_parameters import (
    ProjectionLearning,
    ProjectionLearningSTNSNr,
    N_COR_SEQUENCE,
    COR_COR_SIGMA_EXC,
    COR_COR_SIGMA_FORWARD_INH,
    COR_COR_SIGMA_BACKWARD_INH,
    WEIGHTS,
    N_STATE,
    COR_SEQUENCE_DELAYS,
    COR_SEQUENCE_CIRCULAR,
    BACKWARD_INHIBITION,
    FORWARD_INHIBITION,
    COR_STN_TAU,
    COR_STN_ALPHA,
    STN_SNR_TAU,
    STN_SNR_ALPHA,
)


### helping functions
def _insert_array(a, b, position_idx, circular):
    # a has to be smaller or equal to b
    if len(a) > len(b):
        raise ValueError("a has to be smaller or equal to b")
    # a has to be odd
    if len(a) % 2 == 0:
        raise ValueError("a has to be odd")

    # Calculate the mid-point to center the values of a around position_idx
    mid_point = len(a) // 2

    # Calculate the start and end indices for inserting a
    start_idx = position_idx - mid_point
    end_idx = position_idx + mid_point + 1

    # If start_idx is negative, wrap around to the end of b
    if start_idx < 0 and circular:
        start_idx += len(b)
    # Or cut off the beginning of a
    elif start_idx < 0:
        a = a[-start_idx:]
        start_idx = 0

    # If end_idx exceeds the length of b, wrap around to the beginning
    if end_idx > len(b) and circular:
        end_idx -= len(b)
    # Or cut off the end of a
    elif end_idx > len(b):
        a = a[: -(end_idx - len(b))]
        end_idx = len(b)

    # Insert values of a into b
    if start_idx < end_idx:
        b[start_idx:end_idx] = a
    else:
        # If a wraps around, insert into two parts
        b[start_idx:] = a[: len(b) - start_idx]
        b[:end_idx] = a[len(b) - start_idx :]

    return b


def _half_gaussion_con(n_pop, sigma, weight, reverse=False, circular=False):
    """
    Create a connection matrix for two equally sized populations using a "half" gaussian
    kernel.

    Returns:
        np.ndarray:
            The connection matrix.
    """
    ### create a "half" gaussian kernel with an area of weight
    n_kernel = np.clip(
        2 * int(2 * sigma) + 1, 3, n_pop if n_pop % 2 == 1 else n_pop - 1
    )
    x = np.arange(n_kernel) - n_kernel // 2
    gauss_kernel = np.array(np.exp(-(x**2) / (2 * sigma**2)), dtype=object)
    gauss_kernel[: n_kernel // 2] = (
        weight * gauss_kernel[: n_kernel // 2] / np.sum(gauss_kernel[: n_kernel // 2])
    )
    gauss_kernel[n_kernel // 2 :].fill(None)
    if reverse:
        gauss_kernel = gauss_kernel[::-1]

    ### create matrix (post, pre)
    con = np.empty((n_pop, n_pop), dtype=object)
    con.fill(None)
    ### loop over post neurons i.e. rows of con
    for idx_post in range(n_pop):
        ### add the gaussian kernel so that it is centered on the current post neuron
        ### (idx_post is the center of the kernel), if the kernel goes over the edge of
        ### the matrix, it continues on the other side
        _insert_array(gauss_kernel, con[idx_post, :], idx_post, circular)
    return con


def connect_cor_within(cor: Population, cor_aux: None | Population = None, **kwargs):
    """
    Create the connetions within the cor population.
    (1) Self excitation.
    (2) Sequence propagation (directional gaussian kernel) within each state
    subpopulation.
    (3) Optional inhibition of "backward" (directional gauss, other direction) neurons
    within each state subpopulation.
    (4) Optional inhibition of "forward" (directional gauss, same direction as exc)
    neurons within each state subpopulation.
    (5) Inhibition from cor_aux of a state to all other state subpopulations. (for
    switching between states, i.e. suppress other states when a state is set in cor).

    Args:
        cor (Population):
            The cor population. Needs to be 2D. First dimension is the state, second
            dimension are the neurons within the state (where the sequence is propagated).
        cor_aux (Population, optional):
            The cor_aux population. Size should be the same as the first dimension of
            cor (i.e. the number of states). Default is None.

    Returns:
        cor__cor__self_exc (Projection):
            The self excitation projection.
        cor__cor__sequence_exc (list[Projection]):
            The sequence propagation projections for each state subpopulation.
        cor__cor__sequence_forward_inh (list[Projection]):
            The forward inhibition projections for each state subpopulation.
        cor__cor__sequence_backward_inh (list[Projection]):
            The backward inhibition projections for each state subpopulation.
        cor_aux__cor (list[list[Projection]]):
            The inhibition projections from cor_aux to all other state subpopulations.
            First index is the state of the presynaptic cor_aux population, second index
            is the state of the postsynaptic cor subpopulation. The list is empty if
            cor_aux is None.
    """
    ### try to get the constants from the kwargs, if not set, use the global constants
    weights = kwargs.get("weights", WEIGHTS)
    n_state = kwargs.get("n_state", N_STATE)
    n_cor_sequence = kwargs.get("n_cor_sequence", N_COR_SEQUENCE)
    cor_sequence_delays = kwargs.get("cor_sequence_delays", COR_SEQUENCE_DELAYS)
    cor_sequence_circular = kwargs.get("cor_sequence_circular", COR_SEQUENCE_CIRCULAR)
    backward_inhibition = kwargs.get("backward_inhibition", BACKWARD_INHIBITION)
    forward_inhibition = kwargs.get("forward_inhibition", FORWARD_INHIBITION)
    cor_cor_sigma_exc = kwargs.get("cor_cor_sigma_exc", COR_COR_SIGMA_EXC)
    cor_cor_sigma_forward_inh = kwargs.get(
        "cor_cor_sigma_forward_inh", COR_COR_SIGMA_FORWARD_INH
    )
    cor_cor_sigma_backward_inh = kwargs.get(
        "cor_cor_sigma_backward_inh", COR_COR_SIGMA_BACKWARD_INH
    )

    ### self excitation, except the last neuron in each state (which stops the sequence)
    cor__cor__self_exc = Projection(
        pre=cor[:, : n_cor_sequence - 1],
        post=cor[:, : n_cor_sequence - 1],
        target="exc",
        name="cor__cor__self_exc",
    )
    cor__cor__self_exc.connect_one_to_one(weights=weights["cor__cor__self_exc"])

    ### sequence propagation within states
    cor__cor__sequence_exc = []
    cor__cor__sequence_forward_inh = []
    cor__cor__sequence_backward_inh = []
    cor_aux__cor = []
    cor__cor__sequence_end = []
    for state_idx in range(n_state):
        ### connection within each state subpopulation
        ### exc propagation
        cor__cor__sequence_exc.append(
            Projection(
                pre=cor[state_idx, :],
                post=cor[state_idx, :],
                target="exc",
                name=f"cor__cor__sequence_exc_{state_idx}",
            )
        )
        cor__cor__sequence_exc[state_idx].connect_from_matrix(
            _half_gaussion_con(
                n_pop=n_cor_sequence,
                sigma=cor_cor_sigma_exc,
                weight=weights["cor__cor__sequence_prop_exc"],
                reverse=False,
                circular=cor_sequence_circular,
            ),
            delays=cor_sequence_delays,
        )

        ### connection within each state subpopulation
        ### inhibition propagation
        if forward_inhibition:
            cor__cor__sequence_forward_inh.append(
                Projection(
                    pre=cor[state_idx, :],
                    post=cor[state_idx, :],
                    target="inh",
                    name=f"cor__cor__sequence_forward_inh_{state_idx}",
                )
            )
            cor__cor__sequence_forward_inh[state_idx].connect_from_matrix(
                _half_gaussion_con(
                    n_pop=n_cor_sequence,
                    sigma=cor_cor_sigma_forward_inh,
                    weight=weights["cor__cor__sequence_prop_forward_inh"],
                    reverse=False,
                    circular=cor_sequence_circular,
                ),
                delays=cor_sequence_delays,
            )
        ### connection within each state subpopulation
        ### inhibition of "backward" neurons
        if backward_inhibition:
            cor__cor__sequence_backward_inh.append(
                Projection(
                    pre=cor[state_idx, :],
                    post=cor[state_idx, :],
                    target="inh",
                    name=f"cor__cor__sequence_backward_inh_{state_idx}",
                )
            )
            cor__cor__sequence_backward_inh[state_idx].connect_from_matrix(
                _half_gaussion_con(
                    n_pop=n_cor_sequence,
                    sigma=cor_cor_sigma_backward_inh,
                    weight=weights["cor__cor__sequence_prop_backward_inh"],
                    reverse=True,
                    circular=cor_sequence_circular,
                ),
                delays=cor_sequence_delays,
            )

        ### connection from cor_aux to its state
        if cor_aux is not None:
            cor_aux__cor.append(
                Projection(
                    pre=cor_aux[state_idx],
                    post=cor[state_idx, :],
                    target="inh",
                    name=f"cor_aux__cor_{state_idx}",
                )
            )
            cor_aux__cor[state_idx].connect_all_to_all(weights=weights["cor_aux__cor"])

        ### connection within each state
        ### last neuron of the state inhibits all other neurons of the state (ends the
        ### sequence)
        cor__cor__sequence_end.append(
            Projection(
                pre=cor[state_idx, n_cor_sequence - 1 : n_cor_sequence],
                post=cor[state_idx, :],
                target="inh",
                name=f"cor__cor__sequence_end_{state_idx}",
            )
        )
        cor__cor__sequence_end[state_idx].connect_all_to_all(
            weights=weights["cor__cor__sequence_end"]
        )

    return (
        cor__cor__self_exc,
        cor__cor__sequence_exc,
        cor__cor__sequence_forward_inh,
        cor__cor__sequence_backward_inh,
        cor_aux__cor,
    )


def connect_cor__stn(
    cor: Population,
    stn: Population,
    tau=COR_STN_TAU,
    alpha=COR_STN_ALPHA,
    w_init=WEIGHTS["cor__stn"],
):

    cor__stn = ProjectionLearning(pre=cor, post=stn, target="exc", name="cor__stn")
    cor__stn.connect_all_to_all(weights=w_init)
    cor__stn.set_params(
        tau=tau,
        alpha=alpha,
    )

    return cor__stn


def connect_stn__snr(
    stn: Population,
    snr: Population,
    tau=STN_SNR_TAU,
    alpha=STN_SNR_ALPHA,
    w_init=WEIGHTS["stn__snr"],
):
    ### TODO if this doesn't change, combine it with connect_cor__stn function
    stn__snr = ProjectionLearningSTNSNr(
        pre=stn, post=snr, target="exc", name="stn__snr"
    )
    stn__snr.connect_one_to_one(weights=w_init)
    stn__snr.set_params(
        tau=tau,
        alpha=alpha,
    )

    return stn__snr


def connect_default(
    pre: Population, post: Population, target: str = "exc", mode="one_to_one", **weights
):
    """
    Connects two populations with a one-to-one or all_to_all connection using weights
    from global constant WEIGHTS.

    Args:
        pre (Population):
            Pre-synaptic population
        post (Population):
            Post-synaptic population
        target (str):
            Target of the connection. Default is "exc".
        mode (str):
            Mode of the connection. Default is "one_to_one".

    Returns:
        projection (Projection):
            The created Projection.
    """
    proj_name = f"{pre.name}__{post.name}__{target}"
    projection = Projection(pre=pre, post=post, target=target, name=proj_name)
    if mode == "one_to_one":
        projection.connect_one_to_one(
            weights=weights.get(proj_name, WEIGHTS[proj_name])
        )
    elif mode == "all_to_all":
        projection.connect_all_to_all(
            weights=weights.get(proj_name, WEIGHTS[proj_name])
        )
    return projection


def r_squared(slope, intercept, x, y):
    """
    Calculate the R^2 value of a linear regression (given by slope and intercept) for
    the data points (x, y).

    Args:
        slope (float):
            The slope of the linear regression.
        intercept (float):
            The intercept of the linear regression.
        x (np.ndarray):
            The x values of the data points.
        y (np.ndarray):
            The y values of the data points.
    """
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot
