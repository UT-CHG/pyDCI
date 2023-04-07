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
    Sets up Maxmal Update Density Inverse wroblem for parameter identification

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
        Solve Problem
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
        Plotting functions for DCI Problem Class
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
