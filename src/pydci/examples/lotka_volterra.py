"""
Lotka-Volterra (Predator-Prey) System
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import odeint

from pydci.Model import DynamicModel

# Baseline
LV_P1 = [
    1.0,  # alpha - prey growth rate
    0.02,  # beta - prey death rate
    0.01,  # delta - predator growth rate
    1.0,  # gamma - predator death rate
]

# Increase in death rate of prey
LV_P2 = [
    1.1,  # alpha - prey growth rate
    0.7,  # beta - prey death rate
    0.1,  # delta - predator growth rate
    0.5,  # gamma - predator death rate
]

LV_PARAM_MINS = 4 * [0.0]


def lotka_volterra_system(
    states: list[float],
    time: np.array,
    *parameters: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Parameters
    ----------
    states : list[float]
        A list of the current states of the system.
        states[0] - prey, states[1] - predator

    time : np.ndarray
        simulation time array
    *parameters : tuple[float, float, float, float]
        The parameters of the model: alpha, beta, delta, gamma
        alpha: growth rate of prey population
        beta: death rate of prey population
        delta: growth rate of predator population
        gamma: death rate of predator population

    Returns
    -------
    np.ndarray
        The derivatives of the states with respect to time.
    """
    alpha, beta, delta, gamma = parameters

    xdot = np.array(
        [
            states[0] * (alpha - beta * states[1]),
            states[1] * (-gamma + states[0] * delta),
        ]
    )
    return xdot


class LotkaVolterraModel(DynamicModel):
    """
    Lotka-Volterra Predator Prey model
    """

    def __init__(
        self,
        x0=[100, 10],
        lam_true=LV_P1,
        solve_ts=0.01,
        sample_ts=1,
        measurement_noise=1,
        **kwargs,
    ):
        super().__init__(
            x0,
            lam_true,
            solve_ts=solve_ts,
            sample_ts=sample_ts,
            measurement_noise=measurement_noise,
            param_mins=LV_PARAM_MINS,
            **kwargs,
        )

    def forward_model(self, x0, times, parameter_samples) -> None:
        """
        Runs the RLC model for a specified number of drift windows.
        Uses the initial state, the drift windows, the times, the get_parameters
        and save_output methods of the class to integrate the system of ODEs and
        save the output for each window.

        Parameters
        ----------
        self : object
            The instance of the class
        """
        return odeint(lotka_volterra_system, x0, times, args=parameter_samples)

    def plot_states(self):
        """
        Plot states over time
        """
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        title = ["Predator", "Prey"]
        for i, ax in enumerate(ax.flat):
            self.plot_state(state_idx=i, ax=ax)
            ax.set_title(f"{i}: {title[i]} Temporal Evolution")

    def plot_true_phase_space(
        self,
        n_ints: int = 1,
        int_size: int = None,
        ax: plt.Axes = None,
    ):
        """
        Plot true phase space of prey to predator.

        Parameters
        ----------
        n_ints : int, optional
            Number of intervals to seperate data into, by default 1, or all the data.
        int_size : int
            Size of intervals, by default None, or all the data.
        ax : plt.Axes, optional
            Matplotlib axes to plot on, by default None, or create new axes.

        Returns
        -------
        plt.Axes
            Matplotlib axes with plot.
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        start = 0
        int_size = len(self.data[0]) if int_size is None else int_size
        cols = [c for c in self.data[0].columns if c.startswith("q_lam_true")]
        for n in np.arange(int_size, (1 + n_ints) * int_size, int_size):
            ax = sns.scatterplot(
                self.data[0].loc[list(range(start, n))][cols],
                x="q_lam_true_0",
                y="q_lam_true_1",
                ax=ax,
            )
            start = n

        return ax

    def plot_obs_phase_space(
        self,
        n_ints: int = 1,
        int_size: int = None,
        ax: plt.Axes = None,
    ):
        """
        Plot observed phase space of prey to predator, with error included.

        Parameters
        ----------
        n_ints : int, optional
            Number of intervals to seperate data into, by default 1, or all the data.
        int_size : int
            Size of intervals, by default None, or all the data.
        ax : plt.Axes, optional
            Matplotlib axes to plot on, by default None, or create new axes.

        Returns
        -------
        plt.Axes
            Matplotlib axes with plot.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        start = 0
        int_size = len(self.data[0]) if int_size is None else int_size
        obs_cols = [c for c in self.data[0].columns if c.startswith("q_lam_obs")]
        for n in np.arange(int_size, (1 + n_ints) * int_size, int_size):
            ax = sns.scatterplot(
                self.data[0].dropna().iloc[list(range(start, n))][obs_cols],
                x="q_lam_obs_0",
                y="q_lam_obs_1",
                marker="x",
                linewidth=2,
                ax=ax,
            )
            start = n

        return ax


# TODO: Workout seeds -> Put ones to save in dictionaries
# initial Measure Outcomes
# ------------------------
# (4431, 1394)      pretty good
# (629449, 281824)  good
# (590903, 655235)  pretty good
# (997469, 279770)  okay (bad E(r) to good estimate match)
# (581506, 895913)  best

# lv1_initial_seed = np.random.randint(0, 10e5)
# lv1_measurement_seed = np.random.randint(0, 10e5)
#
# lv1_initial_seed = 581506
# lv1_measurement_seed =  895913
