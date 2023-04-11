import pdb
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from numpy.typing import ArrayLike
from scipy.stats import distributions as dist  # type: ignore
from scipy.stats import rv_continuous  # type: ignore

from pydci.DCIProblem import DCIProblem
from pydci.log import disable_log, enable_log, logger
from pydci.pca import pca
from pydci.utils import fit_domain, get_df, put_df, set_shape


class MUDProblem(DCIProblem):
    """
    Maximal Updated Density Problem

    Maxmal Update Density Inverse problem class for parameter identification.
    This extends the DCIProblem class by computing the Maximal Updated Density,
    or MUD, point, as the parameter sample that maximizes the updated
    distribution in order to solve a parmater estimation problem, as first
    proposed in [1].

    The key distinction is the assumptions being made in the data between a
    parameter estimation problem and a general Data Consistent Inversion
    problem. In a MUDProblem, we assume that the source of the error in the
    parameter samples that we want to quantify is epistemic in nature data,
    and thus our data come from deterministic map, that when pushed forward
    through our QoI map is simply populated with error. The goal is then to
    determine the true value of the parameter that produced the observed data,
    instead of quantifying the probability distribution of the parameter itself,
    and our solution is a point and not a distribution.

    This class extends the DCIProblem class in the following ways:

        1. Initialization - Instead of receiving an observed distribution on
        data as the input, the observed data itself should be passed, along
        with the standard deviation associated with the i.i.d. Gaussian noise
        the data is assumed to be populate with. The observe distribution is
        set by default to a Gaussian Distribution with mean around the man of
        the data and standard deviation equal to the passed in `std_dev`.
        2. `solve()` - Solve method is extended by computing the mud point and
        storing it in the attribute `mud_point`. Note this is calculated as the
        sample that has the maximum `pi_up` value in the classes's `state`
        DataFrame, NOT the maximum value as determined by the kde computed on
        these values.
        3. Plotting - Plots add vertical lines for MUD points on the parameter
        distribution plots, and options for plotting the true value if known.

    Note: this class does no data-aggregation using data-constructed QoI maps
    as proposed in [1] for parameter estimation. See sub-classes `WMEMUDProblem`
    and `PCAMUDProblem` for classes that use data-constructed QoI maps for
    parameter estimation.

    Attributes
    ----------
    data : ArrayLike
        Observed data. Should be of same dimension as the number of observed
        states for each passsed in sample, `q_lam`.
    data : ArrayLike
        Observed data. Should be of same dimension as the number of observed
        states for each passsed in sample, `q_lam`.

    References
    ----------
    [1] M. Pilosov, C. del-Castillo-Negrete, T. Y. Yen, T. Butler, and C.
    Dawson, “Parameter estimation with maximal updated densities,” Computer
    Methods in Applied Mechanics and Engineering, vol. 407, p. 115906, Mar.
    2023, doi: 10.1016/j.cma.2023.115906.
    """

    def __init__(
        self,
        lam,
        q_lam,
        data,
        std_dev,
        init_dist: rv_continuous = None,
        weights: ArrayLike = None,
        normalize: bool = False,
        max_nc: int = None,
    ):
        # Assume gaussian error around mean of data with assumed noise
        self.data = set_shape(np.array(data), (-1, 1))
        obs_dist = dist.norm(loc=np.mean(data), scale=std_dev)
        super().__init__(
            lam,
            q_lam,
            obs_dist,
            init_dist=init_dist,
            weights=weights,
            normalize=normalize,
        )
        self.data = data
        self.std_dev = std_dev
        self.mud_point = None

    def solve(self):
        """
        Solve MUD Parameter Estimation Problem

        Extends the parent method by computing the MUD point, the solution
        to the parameter estimation problem, as the samples that maximizes the
        `pi_up` column in the state DataFrame. This MUD Point is stored in the
        results DataFrame that is returned.
        """
        super().solve()
        m = np.argmax(self.state["pi_up"])
        mud_point = get_df(self.state.loc[[m]], "lam", size=self.n_params)
        self.result = put_df(self.result, "lam_MUD", mud_point, size=self.n_params)
        self.mud_point = mud_point[0]

    def plot_param_state(
        self,
        true_vals=None,
        ax=None,
        param_idx=0,
        ratio_col="ratio",
        plot_mud=True,
        plot_initial=True,
        plot_legend=True,
        figsize=(8, 8),
    ):
        """
        Plotting functions for MUDProblem Class
        """
        ax, labels = super().plot_param_state(
            ax=ax,
            param_idx=param_idx,
            ratio_col=ratio_col,
            plot_legend=False,
            plot_initial=plot_initial,
            figsize=figsize,
        )

        # Generate vertical lines for true values
        if true_vals is not None:
            lam_true_label = (
                f"$\lambda^{{\dagger}}_{param_idx} = "
                + f"{true_vals[0][param_idx]:.4f}$"
            )
            ax.axvline(
                x=true_vals[0][param_idx],
                linewidth=3,
                color="orange",
                label=lam_true_label,
            )
            labels.append(lam_true_label)

        if plot_mud is not None:
            mud_label = (
                f"$\lambda^{{MUD}}_{param_idx} = " + f"{self.mud_point[param_idx]:.4f}$"
            )
            ax.axvline(
                x=self.mud_point[param_idx],
                linewidth=3,
                color="green",
                linestyle="--",
                label=mud_label,
            )
            labels.append(mud_label)

        if plot_legend:
            ax.legend(
                labels=labels,
                fontsize=12,
                title_fontsize=12,
            )

        return ax, labels

    def plot_obs_state(
        self,
        ax=None,
        state_idx=0,
        plot_pf=True,
        plot_obs=True,
        plot_legend=True,
        obs_col="q_lam",
        ratio_col="ratio",
        figsize=(6, 6),
    ):
        ax, labels = super().plot_obs_state(
            ax=ax,
            state_idx=state_idx,
            plot_pf=plot_pf,
            plot_obs=plot_obs,
            plot_legend=plot_legend,
            obs_col=obs_col,
            ratio_col=ratio_col,
            figsize=figsize,
        )

        return ax, labels
