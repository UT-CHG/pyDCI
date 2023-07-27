"""
Lotka-Volterra (Predator-Prey) System
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from pydci.Model import DynamicModel

SEIRS_PARAM_MINS = [0, 0, 0, 0]

# For periodic behavior
R_0 = 3.0

# Daily infection counts
# Simluate batches of weekly data.
SEIRS_SAMPLE_TS = 1

# Populations are all from 0-1 -> Fraction of population
SEIRS_NOISE = 0.005

# Parameters from bjornstad2020seirs paper
SEIRS_P1 = [
    R_0 / 14.0,  # beta transmission rate - R_0 / gamma -> R_0 > 0 for periodic behavior
    1.0 / 7.0,  # sigma incubation rate i.e. rate at which exposed hosts become infected - 1 week
    1.0 / 14.0,  # gamma  mean recovery rate - 2 weeks
    1.0 / 365.0,  # xi - loss off imunity rate - 1 year
]

# (1) Policy Lockdown 1 month in: => Slower Transmission Rate(beta) - Time 75
SEIRS_P2 = [
    0.5 * R_0 / 14.0,  # transmission rate halved
    1.0 / 7.0,  # sigm
    1.0 / 14.0,
    1.0 / 365.0,
]
# (2) Virus Mutation 1 year in: => Faster Incubation Rate (sigma) - Time 150
SEIRS_P3 = [
    0.5 * R_0 / 14.0,  # transmission rate halved
    1.0 / 3.5,   # Incubation rate halved -> Exposed hosts become infected quicker
    1.0 / 14.0,
    1.0 / 365.0,
]

SEIRS_X0 = [
    0.99,  # susceptible
    0.010,  # exposed
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
        state_idxs = [2], # Only observe infected state
        **kwargs
    ):
        super().__init__(
            x0,
            lam_true,
            solve_ts=solve_ts,
            sample_ts=sample_ts,
            measurement_noise=measurement_noise,
            param_mins=SEIRS_PARAM_MINS,
            state_idxs=state_idxs,
            **kwargs
        )

    def forward_model(self, x0, times, parameter_samples) -> None:
        """
        Integrates SEIRS Model equations using scipy odeint
        """
        return odeint(seir_system, x0, times, args=parameter_samples)

    def plot_states(self, plot_samples: bool = False):
        """
        Plot states over time
        """
        fig, ax = plt.subplots(4, 1, figsize=(18, 16))
        state_labels = ["Susceptible", "Exposed", "Infected", "Recovered"]
        for i, ax in enumerate(ax.flat):
            self.plot_state(state_idx=i, ax=ax)
            ax.set_title(f'{state_labels[i]}')
            ax.set_ylabel("Fraction of Population")
            ax.set_xlabel("Time (Days)")


    def plot_infected(self, **kwargs):
        """
        Plot infected population over time 
        """
        ax = self.plot_state(state_idx=2, **kwargs)
        ax.set_ylabel('Fraction of Population')
        ax.set_xlabel('Days')
        ax.set_title('Infected Population')

        return ax