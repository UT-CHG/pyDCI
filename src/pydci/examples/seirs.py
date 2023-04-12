"""
Lotka-Volterra (Predator-Prey) System
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from pydci.Model import DynamicModel


SEIRS_PARAM_MINS = [0, 0, 0, 0]

# For periodic behavior
R_0 = 3.0

# Sample every week
SEIRS_SAMPLE_TS = 7

# Populations are all from 0-1 -> Fraction of population
SEIRS_NOISE = 0.005

# Parameters from bjornstad2020seirs paper
SEIRS_P1 = [
    3.0/14., # beta transmission rate
    1./7.,   # sigma incubation rate i.e. rate at which exposed hosts become infected - 1 week
    1./14.,  # gamma  mean recovery rate - 2 weeks
    1./365.,  # xi - loss off imunity rate - 1 year
]


# (1) Policy Lockdown: => Slower Transmission Rate(beta) - Time 25
SEIRS_P2 = [
    0.3,  # beta
    0.25,  # sigma
    0.1,  # gamma
    0.1,  # xi 
]
# (2) Virus Mutation: => Faster Incubation Rate (sigma) - Time 75
SEIRS_P3 = [
    0.3,  # beta
    0.3,  # sigma
    0.15,  # gamma
    0.1,  # xi 
]

SEIRS_X0 = [
        0.99,    # susceptible
        0.001,  # exposed
        0,  # infected 
        0,  # recovered
]


def seir_system(
    states: np.ndarray, time: np.ndarray, *parameters: tuple[float, float]
) -> np.ndarray:
    """
    Parameters
    ----------
    states : np.ndarray
        An array of the current states of the system
    time : np.ndarray
        simulation time array
    *parameters : tuple[float, float]
        The parameters of the model: beta, gamma
    Returns
    -------
    np.ndarray
        The derivatives of the states with respect to time
    """

    beta, sigma, gamma, xi = parameters

    xdot = np.array(
        [
            -beta * states[2] * states[0] + xi * states[3],
            -sigma * states[1] + beta * states[2] * states[0],
            -gamma * states[2] + sigma * states[1],
            gamma * states[2] - xi * states[3],
        ]
    )

    return xdot


class SEIRSModel(DynamicModel):
    def __init__(
        self,
        x0=SEIRS_X0,
        lam_true=SEIRS_P1,
        solve_ts=0.1,
        sample_ts=SEIRS_SAMPLE_TS,
        measurement_noise=SEIRS_NOISE,
        **kwargs
    ):
        super().__init__(
            x0,
            lam_true,
            solve_ts=solve_ts,
            sample_ts=sample_ts,
            measurement_noise=measurement_noise,
            param_mins=SEIRS_PARAM_MINS,
            **kwargs
        )

    def forward_model(self, x0, times, parameter_samples) -> None:
        """
        Integrates SEIRS Model equations using scipy odeint
        """
        return odeint(seir_system, x0, times, args=parameter_samples)

    def plot_states(self):
        """
        Plot states over time
        """
        fig, ax = plt.subplots(4, 1, figsize=(18, 16))
        for i, ax in enumerate(ax.flat):
            self.plot_state(state_idx=i, ax=ax)



