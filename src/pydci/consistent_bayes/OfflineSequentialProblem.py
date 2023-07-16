"""
Consistent Bayesian Formulation for Data-Consistent Inversion

The classes in this module all derive off of the Consistent-Bayesian formulation
for solving Stochastic Inverse problems first proposed in [1]. The classes all
inherit from the base class, `DCIProblem` and all have the following structure
in general in terms of how they are to be used:

1. Initialization: Upon initailization the state of the system is set,
including parameter samples, their values evaluated through the forward model,
and data/assumptions on observed data. This is used to initailiaze a pandas
DataFrame `state`, that stores these values and is used for computing and
storing final solutions
2. Solving: solve() -> Main method called to solve the problem class. Specific
parameters controlling how the algorithm is solved can be set here. The results
of the solve are store in the `result` attribute of the class.
3. Visualizing: plot_L(), plot_D(), plot_dists() -> Plot resulting
distributions from solving the problem.

References
----------
[1] T. Butler, J. Jakeman, and T. Wildey, “Combining Push-Forward Measures
and Bayes’ Rule to Construct Consistent Solutions to Stochastic Inverse
Problems,” SIAM J. Sci. Comput., vol. 40, no. 2, pp. A984–A1011, Jan. 2018,
doi: 10.1137/16M1087229.

TODO List:

    - Mud point for inherited classes not being plotted correctly
    - Sequential (should be renamed split sequential?) - Plot distribution
    methods for plotting distributions per iteration, qoi combinations,
    or pca values
    Dynamic Problem -> Finish

"""
import itertools
import pdb
import random
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import alive_bar
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike
from rich.table import Table
from scipy.stats import rv_continuous  # type: ignore
from scipy.stats.distributions import norm
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from pydci.consistent_bayes.MUDProblem import MUDProblem
from pydci.log import disable_log, enable_log, log_table, logger
from pydci.utils import KDEError, fit_domain, get_df, put_df, set_shape, closest_factors

sns.color_palette("bright")
sns.set_style("darkgrid")

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"


class PCAMUDProblem(MUDProblem):
    """
    PCA MUD Problem

    Sets up a Maximal Updated Density (MUD) parameter estimation using the
    `q_pca` map to aggregate data as proposed in [1]. The `q_pca` map is a way
    of aggregating observed data with simulated data for parameter estimation
    problems using the Data Consistent Inversion (DCI) framework. By inverting
    on a map of aggregated data instead of the map that produced the data
    itself, the variance in the MUD parameter estimate can be reduced as more
    data is incorporated.

    This class extends the MUDProblem class by using the `q_pca()` function
    before solving the parameter estimation problem to aggregate data and invert
    on the data-constructed map instead.

    Attributes
    ----------
    pca_res : List[pd.DataFrame]

    Methods
    -------
    solve(pca_mask=None, max_nc=None, best_method="closest", exp_thresh=0.5)
        Solve the parameter estimation problem, with the parameters relevant
        to aggregating the data into the `q_pca()` map and determing how many
        principal components to use for optimal solution.

    TODO:
        - Make pca_maks a property

    """

    def __init__(
        self,
        samples,
        data,
        std_dev,
        pi_in=None,
    ):
        self.samples = samples
        self.data = data
        self.std_dev = std_dev
        self.dists = {
            "pi_in": pi_in,
            "pi_pr": None,
            "pi_obs": None,
            "pi_up": None,
            "pi_pf": None,
        }

    def solve(
        self,
        pca_components=[[0]],
        pca_mask: List[int] = None,
        pca_splits: List[int] = 1,
        exp_thresh: float = 0.5,
        state_extra: dict = None,
    ):
        """
        Solve the parameter estimation problem

        This extends the `MUDProblem` solution class by using the `q_pca()` map
        to aggregate data between the observed and predicted values and
        determine the best MUD estimate that fits the data.

        Parameters
        ----------
        """
        it_results = []
        weights = []
        failed = False
        if exp_thresh <= 0:
            msg = f"Expected ratio thresh must be a float > 0: {exp_thresh}"
            logger.error(msg)
            raise ValueError(msg)
        if isinstance(pca_splits, int) or pca_splits is None:
            # Make even number of splits of all qoi if mask is not specified
            pca_mask = np.arange(self.n_qoi) if pca_mask is None else pca_mask
            pca_splits = [
                range(x[0], x[-1] + 1) for x in np.array_split(pca_mask, pca_splits)

            ]
        elif isinstance(pca_splits, list):
            if pca_mask is not None:
                raise ValueError(
                    "Cannot specify both pca_mask and non-integer pca_splits"
                )
        iterations = [(i, j) for i in pca_splits for j in pca_components]
        prev_in = self.dists['pi_in']
        for i, (pca_mask, pca_cs) in enumerate(iterations):
            str_val = pca_mask if pca_mask is not None else "ALL"
            logger.info(f"Iteration {i}: Solving using ({str_val}, {pca_cs})")

            # TODO: Make sure this fixes all cases
            # ! Problem: Setting weights erases pi_in, and when
            # ! We are doing online iteration, this wipes our previously compute pi_up
            # ! So we suffer from sampling error twice on each iteration....
            # ! Fix for now is to change set weights to only whipe dists dictionary
            # ! on first iteration since first iteration passes in [] for weights
            if i != 0:
                self.set_weights(weights)
            try:
                self.solve(
                    pca_mask=pca_mask,
                    pca_components=pca_cs,
                )
            except ZeroDivisionError as z:
                if i == 0:
                    z.msg = "Pred assumption failed on first iteration."
                    raise z
                else:
                    logger.info(f"({i}): pred assumption failed - str({z})")
                    failed = True
            except KDEError as k:
                if i == 0:
                    k.msg = "Failed to estiamte KDEs on first iteration."
                    raise k
                else:
                    logger.info(f"({i}): KDE estimation failed - str({k})")
                    failed = True
            except LinAlgError as l:
                # * Thrown when pdf() method fails on pi_in from another iteration
                if i == 0:
                    l.msg = "Unknown linalg error on first iteration."
                    raise l
                else:
                    logger.info(f"({i}): PDF on constructed kde failed. " +
                                f"Highly correlated data, or curse of dim - str({l})")
                    failed = True
            else:
                e_r = self.result["e_r"].values[0]
                if (diff := np.abs(e_r - 1.0)) > exp_thresh or failed:
                    logger.info(f"|E(r) - 1| = {diff} > {exp_thresh} - Stopping")
                    failed = True

            if failed:
                logger.info(f"Resetting to last solution at {iterations[i-1]}")
                self.set_weights(weights[:-1])
                self.solve(
                    pca_mask=iterations[i - 1][0],
                    pca_components=iterations[i - 1][1],
                )
                break
            else:
                state_vals = {
                    "iteration": len(it_results),
                    "pca_components": str(pca_cs),
                    "pca_mask": str(pca_mask),
                }
                if state_extra is not None:
                    state_vals.update(state_extra)
                self.save_state(state_vals)
                it_results.append(self.result.copy())
                it_results[-1]["i"] = len(it_results) - 1
                if i != len(iterations) - 1:
                    logger.info("Updating weights")
                    weights.append(self.state["ratio"].values)

        self.it_results = pd.concat(it_results)
        self.result = self.it_results.iloc[[-1]]
