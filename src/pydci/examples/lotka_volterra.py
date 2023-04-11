"""
Lotka-Volterra (Predator-Prey) System
"""
import numpy as np
from scipy.integrate import odeint

from pydci.Model import DynamicModel

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
    alpha, beta, gamma, delta = parameters

    xdot = np.array(
        [
            states[0] * (alpha - beta * states[1]),
            states[1] * (-gamma + states[0] * delta),
        ]
    )
    return xdot

class LotkaVolteraModel(DynamicModel):

    # Baseline
    lv_p1 = [
        1.1,  # alpha - prey growth rate
        0.4,  # beta - prey death rate
        0.5,  # gamma - predator death rate
        0.1,  # delta - predator growth rate
    ]

    # Increase in death rate of prey
    lv_p2 = [
        1.1,  # alpha - prey growth rate
        0.7,  # beta - prey death rate
        0.5,  # gamma - predator death rate
        0.1,  # delta - predator growth rate
    ]

    lv_param_mins = 4 * [0.0]

    def __init__(self,
                 x0=[2, 4],
                 lam_true=lv_p1,
                 solve_ts=0.1,
                 sample_ts=1.0,
                 measurement_noise=0.25,
                 **kwargs
                ):
        super().__init__(x0, lam_true,
                         solve_ts=solve_ts,
                         sample_ts=sample_ts,
                         measurement_noise=measurement_noise,
                         param_mins=lv_param_mins,
                         **kwargs)

    def forward_model(x0, times, parameter_samples) -> None:
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
