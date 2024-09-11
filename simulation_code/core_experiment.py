from ANNarchy import get_population
from CompNeuroPy import SimulationEvents
from core_parameters import (
    N_STATE,
    SIMULATION_T_OFFSET_HOLD_CUE,
    SIMULATION_T_OFFSET_CUE_TARGET,
    SIMULATION_T_OFFSET_TARGET_END,
    SIMULATION_T_INPUT_COR,
    SIMULATION_T_INPUT_SC,
    SIMULATION_T_COR_STATE_DELAY,
    COR_INPUT_AMPLITUDE,
    CHECK_LEARNING_T_COR_STATE_DELAY,
    CHECK_LEARNING_T_INPUT_COR,
    CHECK_LEARNING_T_INPUT_SC,
    CHECK_LEARNING_T_OFFSET_INPUT_END,
)
from numpy.random import Generator


### define simulation events
class TrialMinamimotoAndKimura(SimulationEvents):
    """
    Class to simulate a trial of the task from [Minamimoto and Kimura (2002)](https://doi.org/10.1152/jn.2002.87.6.3090).

    Attributes:
        cue_side (int):
            Side of the cue light (0: left, 1: right) of the last run trial.
        target_side (int):
            Side of the target light (0: left, 1: right) of the last run trial.
        congruent (int):
            Cue-target congruency (0: congruent, 1: incongruent) of the last run trial.
        cue_time (int):
            Time of the cue light onset (relative to the hold light onset) of the last
            run trial.
        target_time (int):
            Time of the target light onset (relative to the hold light onset) of the
            last run trial.
        fast_target (bool):
            Whether the target light onset delay is minimal (only set for a single run).
            Defaults to False.
    """

    def __init__(
        self,
        rng: Generator,
        n_state: int = N_STATE,
        p_congruent: float = 0.8,
        sc_input: bool = True,
        hold_state: bool = True,
        hold_response: bool = True,
        cue_state: bool = True,
        cue_response: bool = True,
        only_left_cue: bool = False,
        sc_all: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            rng (np.random.Generator):
                Random number generator.
            n_state (int):
                Number of states in the cortex. Defaults to N_STATE.
            p_congruent (float):
                Probability for a congruent trial. Defaults to 0.8.
            sc_input (bool):
                Whether to activate the SC input during the trial. Defaults to True.
            hold_state (bool):
                Whether to activate the hold state in the cortex. Defaults to True.
            hold_response (bool):
                Whether to activate the hold response in the SC. Defaults to True.
            cue_state (bool):
                Whether to activate the cue state in the cortex. Defaults to True.
            cue_response (bool):
                Whether to activate the cue response in the SC. Defaults to True.
            only_left_cue (bool):
                Whether only the left cue causes responses in the cortex and SC. Defaults
                to False.
            sc_all (bool):
                Whether to activate all SC neurons during target. Defaults to False.
            verbose (bool):
                Whether to print additional information. Defaults to False.
        """
        ### raise value error if n_state is not 3 or 5
        if n_state not in [3, 5]:
            raise ValueError("n_state must be 3 or 5")
        ### set attributes for the simulation
        self._cue_side = None
        self._congruent = None
        self._delay_hold_cue = None
        self._delay_cue_target = None
        self._delay_target_end = None
        self.rng = rng
        self.n_state = n_state
        self.p_congruent = p_congruent
        self.p_incongruent = 1 - p_congruent
        self.sc_input = sc_input
        self.hold_state = hold_state
        self.hold_response = hold_response
        self.cue_state = cue_state
        self.cue_response = cue_response
        self.only_left_cue = only_left_cue
        self.fast_target = False
        self.sc_all = sc_all
        super().__init__(verbose=verbose)

        ### add all events
        self._add_all_events()

    def run(self):
        """run function from SimulationEvents"""
        super().run()

    ### create only readable properties
    @property
    def cue_side(self):
        if self._cue_side is None:
            return self._cue_side
        return int(self._cue_side)

    @property
    def target_side(self):
        if self._congruent is None or self._cue_side is None:
            return None
        return self.cue_side if self.congruent == 0 else 1 - self.cue_side

    @property
    def congruent(self):
        if self._congruent is None:
            return self._congruent
        return int(self._congruent)

    @property
    def cue_time(self):
        return self._delay_hold_cue

    @property
    def target_time(self):
        return self.cue_time + self._delay_cue_target

    ### functions for random delays
    def _get_delay_hold_cue(self):
        self._delay_hold_cue = self.rng.integers(
            low=SIMULATION_T_OFFSET_HOLD_CUE[0],
            high=SIMULATION_T_OFFSET_HOLD_CUE[1] + 1,
        )
        return self._delay_hold_cue

    def _get_delay_cue_target(self):
        if self.fast_target:
            self._delay_cue_target = min(SIMULATION_T_OFFSET_CUE_TARGET)
            self.fast_target = False
        else:
            self._delay_cue_target: int = self.rng.choice(
                SIMULATION_T_OFFSET_CUE_TARGET
            )
        return self._delay_cue_target

    def _get_delay_target_end(self):
        self._delay_target_end = self.rng.integers(
            low=SIMULATION_T_OFFSET_TARGET_END[0],
            high=SIMULATION_T_OFFSET_TARGET_END[1] + 1,
        )
        return self._delay_target_end

    ### add all events
    def _add_all_events(self):
        ### hold light
        self.add_event(
            name="hold_light",
            trigger={
                "activate_cor_hold": SIMULATION_T_COR_STATE_DELAY,
                "activate_sc_hold": 0,
                "cue_light": self._get_delay_hold_cue,
            },
        )

        ### activate cor hold
        self.add_event(
            name="activate_cor_hold",
            trigger={
                "cor_hold_on": 0,
                "cor_hold_off": SIMULATION_T_INPUT_COR,
            },
        )

        self.add_event(
            name="cor_hold_on",
            effect=self.cor_hold_on,
        )

        self.add_event(
            name="cor_hold_off",
            effect=self.cor_hold_off,
        )

        ### activate sc hold
        self.add_event(
            name="activate_sc_hold",
            trigger={
                "sc_hold_on": 0,
                "sc_hold_off": SIMULATION_T_INPUT_SC,
            },
        )

        self.add_event(
            name="sc_hold_on",
            effect=self.sc_hold_on,
        )

        self.add_event(
            name="sc_hold_off",
            effect=self.sc_hold_off,
        )

        ### cue light
        self.add_event(
            name="cue_light",
            effect=self.set_cue,
            trigger={
                "activate_cor_cue": SIMULATION_T_COR_STATE_DELAY,
                "activate_sc_cue": 0,
                "target_light": self._get_delay_cue_target,
            },
        )

        ### activate cor cue
        self.add_event(
            name="activate_cor_cue",
            trigger={
                "cor_cue_on": 0,
                "cor_cue_off": SIMULATION_T_INPUT_COR,
            },
        )

        self.add_event(
            name="cor_cue_on",
            effect=self.cor_cue_on,
        )

        self.add_event(
            name="cor_cue_off",
            effect=self.cor_cue_off,
        )

        ### activate sc cue
        self.add_event(
            name="activate_sc_cue",
            trigger={
                "sc_cue_on": 0,
                "sc_cue_off": SIMULATION_T_INPUT_SC,
            },
        )

        self.add_event(
            name="sc_cue_on",
            effect=self.sc_cue_on,
        )

        self.add_event(
            name="sc_cue_off",
            effect=self.sc_cue_off,
        )

        ### target light (only activates cortical states if self.n_state == 5)
        self.add_event(
            name="target_light",
            effect=self.set_target,
            trigger=(
                {
                    "activate_cor_target": SIMULATION_T_COR_STATE_DELAY,
                    "activate_sc_target": 0,
                    "end": self._get_delay_target_end,
                }
                if self.n_state == 5
                else {
                    "activate_sc_target": 0,
                    "end": self._get_delay_target_end,
                }
            ),
        )

        ### activate cor target
        if self.n_state == 5:
            self.add_event(
                name="activate_cor_target",
                trigger={
                    "cor_target_on": 0,
                    "cor_target_off": SIMULATION_T_INPUT_COR,
                },
            )

            self.add_event(
                name="cor_target_on",
                effect=self.cor_target_on,
            )

            self.add_event(
                name="cor_target_off",
                effect=self.cor_target_off,
            )

        ### activate sc target
        self.add_event(
            name="activate_sc_target",
            trigger={
                "sc_target_on": 0,
                "sc_target_off": SIMULATION_T_INPUT_SC,
            },
        )

        self.add_event(
            name="sc_target_on",
            effect=self.sc_target_on,
        )

        self.add_event(
            name="sc_target_off",
            effect=self.sc_target_off,
        )

    """
    in the following event effects functions

    cor indices (states):
    0: wait for cue (hold)
    1: cue was left
    2: cue was right
    3: target was congruent
    4: target was incongruent

    3 and 4 only if self.n_state == 5, elif self.n_state == 3 only 0, 1, 2

    sc indices (outcomes):
    0: hold light
    1: left side light
    2: right side light
    """

    def cor_hold_on(self):
        if self.hold_state:
            ### reset the state in the cortex
            get_population("cor")[0, :].r = 0
            ### activate the state in the cortex
            get_population("cor")[0, 0].I_ext = COR_INPUT_AMPLITUDE

    def cor_hold_off(self):
        get_population("cor")[0, 0].I_ext = 0

    def sc_hold_on(self):
        ### activate SC and suppress the corresponsing state (before it gets active
        ### again, just in case it is activated it will be reset)
        if self.sc_input and self.hold_response:
            get_population("sc")[0].I_ext = 1
        # get_population("cor_aux")[0].I_ext = 1

    def sc_hold_off(self):
        get_population("sc")[0].I_ext = 0
        # get_population("cor_aux")[0].I_ext = 0

    def set_cue(self):
        ### 0: left, 1: right
        self._cue_side = self.rng.choice([0, 1], p=[0.5, 0.5])

    def cor_cue_on(self):
        if self.cue_state and (self.cue_side == 0 or not self.only_left_cue):
            ### reset the state in the cortex
            get_population("cor")[self.cue_side + 1, :].r = 0
            ### activate the state in the cortex
            get_population("cor")[self.cue_side + 1, 0].I_ext = COR_INPUT_AMPLITUDE

    def cor_cue_off(self):
        get_population("cor")[self.cue_side + 1, 0].I_ext = 0

    def sc_cue_on(self):
        if (
            self.sc_input
            and self.cue_response
            and (self.cue_side == 0 or not self.only_left_cue)
        ):
            get_population("sc")[self.cue_side + 1].I_ext = 1
        # get_population("cor_aux")[self.cue_side + 1].I_ext = 1

    def sc_cue_off(self):
        get_population("sc")[self.cue_side + 1].I_ext = 0
        # get_population("cor_aux")[self.cue_side + 1].I_ext = 0

    def set_target(self):
        ### 0: congruent, 1: incongruent
        self._congruent = self.rng.choice([0, 1], p=[0.8, 0.2])
        ### self.target_side is set as a property depending on self._congruent

    def cor_target_on(self):
        ### reset the state in the cortex
        get_population("cor")[self.congruent + 3, :].r = 0
        ### activate the state in the cortex
        get_population("cor")[self.congruent + 3, 0].I_ext = COR_INPUT_AMPLITUDE

    def cor_target_off(self):
        get_population("cor")[self.congruent + 3, 0].I_ext = 0

    def sc_target_on(self):
        if self.sc_input:
            if self.sc_all:
                get_population("sc").I_ext = 1
            else:
                get_population("sc")[self.target_side + 1].I_ext = 1
        # if self.n_state == 5:
        #     get_population("cor_aux")[self.congruent + 3].I_ext = 1

    def sc_target_off(self):
        if self.sc_all:
            get_population("sc").I_ext = 0
        else:
            get_population("sc")[self.target_side + 1].I_ext = 0
        # if self.n_state == 5:
        #     get_population("cor_aux")[self.congruent + 3].I_ext = 0


class TrialCheckLearning(SimulationEvents):
    """
    Class to simulate a trial in which a single corte state gets active always at the
    same time and after a fixed delay a random sc neuron gets active (output).

    Attributes:
        outcome_side (int):
            Side of the outcome (0: left, 1: right) of the last run trial.
        delay_input_outcome (int):
            Delay between the input and the outcome of the last run trial.
        cor_input (bool):
            Whether the input state in the cortex was activated in the last run trial.
        fast_target (bool):
            Whether the input-outcome delay is minimal (only set for a single run).
            Defaults to False.
    """

    def __init__(
        self,
        rng: Generator,
        p: float = 0.8,
        p_input: float = 1.0,
        input_outcome_offset: int | list = 600,
        sc_all: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            rng (np.random.Generator):
                Random number generator.
            p (float):
                Probability for outcome side left. Defaults to 0.8.
            p_input (float):
                Probability for input. Defaults to 1.0.
            input_outcome_offset (int or list):
                Offset between the input and the outcome. If list is given, delay is
                randomly chosen from list. Defaults to 0.
            sc_all (bool):
                Whether to activate all sc neurons. Defaults to False.
            verbose (bool):
                Whether to print additional information. Defaults to False.
        """
        ### set attributes for the simulation
        self._outcome_side = None
        self._delay_input_outcome = None
        self._cor_input = None
        self.input_outcome_offset = input_outcome_offset
        self.rng = rng
        self.p = p
        self.p_input = p_input
        self.sc_all = sc_all
        self.sc_input = True
        self.fast_target = False
        super().__init__(verbose=verbose)

        ### add all events
        self._add_all_events()

    ### create only readable properties
    @property
    def outcome_side(self):
        return self._outcome_side

    @property
    def delay_input_outcome(self):
        return self._delay_input_outcome

    @property
    def cor_input(self):
        return self._cor_input

    ### functions for random delays
    def _get_delay_input_outcome(self):
        if isinstance(self.input_outcome_offset, list):
            if self.fast_target:
                self._delay_input_outcome = min(self.input_outcome_offset)
                self.fast_target = False
            else:
                self._delay_input_outcome = self.rng.choice(self.input_outcome_offset)
        else:
            self._delay_input_outcome = self.input_outcome_offset
        return self._delay_input_outcome

    ### add all events
    def _add_all_events(self):
        ### input light (without otucome in sc)
        self.add_event(
            name="input",
            onset=1000,
            effect=self.set_input,
            trigger={
                "activate_cor": CHECK_LEARNING_T_COR_STATE_DELAY,
                "activate_sc_input": 0,
                "outcome": self._get_delay_input_outcome,
                "end": CHECK_LEARNING_T_OFFSET_INPUT_END,
            },
        )

        ### activate cor for input
        self.add_event(
            name="activate_cor",
            trigger={
                "cor_on": 0,
                "cor_off": CHECK_LEARNING_T_INPUT_COR,
            },
        )

        self.add_event(
            name="cor_on",
            effect=self.cor_on,
        )

        self.add_event(
            name="cor_off",
            effect=self.cor_off,
        )

        ### input response in sc, input is like the cue for left
        self.add_event(
            name="activate_sc_input",
            trigger={
                "sc_input_on": 0,
                "sc_input_off": CHECK_LEARNING_T_INPUT_SC,
            },
        )

        self.add_event(
            name="sc_input_on",
            effect=self.sc_input_on,
        )

        self.add_event(
            name="sc_input_off",
            effect=self.sc_input_off,
        )

        ### outcome light
        self.add_event(
            name="outcome",
            effect=self.set_outcome_side,
            trigger={
                "activate_sc_outcome": 0,
            },
        )

        ### activate sc for outcome
        self.add_event(
            name="activate_sc_outcome",
            trigger={
                "sc_on": 0,
                "sc_off": CHECK_LEARNING_T_INPUT_SC,
            },
        )

        self.add_event(
            name="sc_on",
            effect=self.sc_on,
        )

        self.add_event(
            name="sc_off",
            effect=self.sc_off,
        )

    """
    in the following event effects functions

    cor indices (states):
    0: input

    sc indices (outcomes):
    0: left outcome side
    1: right outcome side
    """

    def set_input(self):
        ### set whether the input state in the cortex is activated
        ### if yes --> cue for left, if no --> cue for right
        self._cor_input = self.rng.choice(
            [True, False], p=[self.p_input, 1 - self.p_input]
        )

    def cor_on(self):
        if self.cor_input:
            ### reset the state in the cortex
            get_population("cor")[0, :].r = 0
            ### activate the state in the cortex
            get_population("cor")[0, 0].I_ext = COR_INPUT_AMPLITUDE

    def cor_off(self):
        get_population("cor")[0, 0].I_ext = 0

    def set_outcome_side(self):
        ### 0: left, 1: right
        if self.cor_input:
            # cue for left -> left has probability p
            self._outcome_side = self.rng.choice([0, 1], p=[self.p, 1 - self.p])
        else:
            # cue for right -> right has probability p
            self._outcome_side = self.rng.choice([1, 0], p=[self.p, 1 - self.p])

    def sc_on(self):
        if self.sc_input:
            get_population("sc")[self.outcome_side].I_ext = 1
            if self.sc_all:
                get_population("sc").I_ext = 1

    def sc_off(self):
        get_population("sc").I_ext = 0

    def sc_input_on(self):
        if self.cor_input:
            # cue for left -> left response
            get_population("sc")[0].I_ext = 1
        else:
            # cue for right -> right response
            get_population("sc")[1].I_ext = 1

    def sc_input_off(self):
        get_population("sc")[0].I_ext = 0
