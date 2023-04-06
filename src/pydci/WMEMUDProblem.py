"""
WME MUD Problem class
"""
from typing import Callable, List, Optional, Union
import pdb

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import distributions as dist  # type: ignore
from scipy.stats import gaussian_kde as gkde  # type: ignore
from scipy.stats import rv_continuous  # type: ignore

from pydci.MUDProblem import MUDProblem
from pydci.utils import fit_domain, set_shape, get_df, put_df
from pydci.pca import pca
from matplotlib.patches import Patch

from pydci.log import logger, enable_log, disable_log


class WMEMUDProblem(MUDProblem):
    """
    Sets up Maxmal Update Density Inverse wroblem for parameter identification

    """
    def __init__(
        self,
        lam,
        q_lam,
        data,
        std_dev,
        init_dist=None,
        weights: ArrayLike = None,
        normalize: bool = False,
        max_nc: int = None
    ):
        # Since we aggregate -> Observed distribution fixed
        super().__init__(lam, q_lam, data, std_dev,
                         init_dist=init_dist,
                         weights=weights,
                         normalize=normalize)

    def q_wme(self):
        """
        Build QoI Map Using Data and Measurements

        Aggregate q_lam data with observed data for MUD convergence.
        """
        if 'q_wme_0' in self.state.columns:
            return
        # TODO: Verify/reshape data appropriately
        # TODO: sub sample data/qoi 
        residuals = np.subtract(self.data.T, self.q_lam) / self.std_dev
        qoi = (np.sum(residuals, axis=1) /
               np.sqrt(self.n_samples)).reshape(-1, 1)
        self.q_lam = qoi
        self.state = put_df(self.state, 'q_wme', qoi, size=1)

    def solve(self):
        """
        Solve problem
        """
        # Determine best MUD estimate
        self.q_wme()
        self.dists['predicted'] = None
        self.dists['observed'] = dist.norm(loc=[0], scale=1)
        super().solve()

    def plot_param_state(
        self,
        true_vals=None,
        ax=None,
        nc=1,
        param_idx=0,
        plot_mud=True,
        plot_initial=True,
        plot_legend=True,
    ):
        """
        Plotting functions for DCI Problem Class
        """
        ax, labels = super().plot_param_state(
            true_vals=true_vals,
            ax=ax,
            param_idx=param_idx,
            plot_mud=True,
            plot_initial=True,
            plot_legend=True,
            figsize=(8, 8),
            ratio_col='ratio',
        )

        return ax, labels

    def plot_obs_state(
        self,
        ax=None,
        nc=1,
        state_idx=0,
        plot_pf=True,
        plot_obs=True,
        plot_legend=True,
        figsize=(8, 8),
    ):
        """
        Plotting function for DCI Problem Class
        """
        ax, labels = super().plot_obs_state(
                ax=ax,
                state_idx=state_idx,
                plot_pf=plot_pf,
                plot_obs=plot_obs,
                plot_legend=plot_legend,
                obs_col='q_wme',
                ratio_col='ratio',
                figsize=figsize)

        return ax, labels
