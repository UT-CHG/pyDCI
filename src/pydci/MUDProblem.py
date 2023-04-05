from typing import Callable, List, Optional, Union
import pdb

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import distributions as dist  # type: ignore
from scipy.stats import gaussian_kde as gkde  # type: ignore
from scipy.stats import rv_continuous  # type: ignore

from pydci.DCIProblem import DCIProblem
from pydci.utils import fit_domain, set_shape, get_df, put_df
from pydci.pca import pca
from matplotlib.patches import Patch

from pydci.log import logger, enable_log, disable_log


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
        aggregate='pca',
        init_dist = None,
        weights: ArrayLike = None,
        normalize: bool = False,
        max_nc: int = None
    ):
        # Since we aggregate -> Observed distribution fixed
        obs_dist = dist.norm(loc=0, scale=1)
        super().__init__(lam, q_lam, obs_dist,
                         init_dist=init_dist,
                         weights=weights,
                         normalize=normalize)
        self.data = data
        self.std_dev = std_dev
        if aggregate not in ['wme', 'pca']:
            ValueError(f"Unrecognized QoI Map type {method}")
        self.method = aggregate
        self.max_nc = self.n_params if max_nc is None else max_nc

        logger.info('Initialized MUD Problem')

    def solve(self):
        """
        Solve Problem
        """
        self.aggregate()
        super().solve()

    def mud_point(self):
        """Maximal Updated Density (MUD) Point

        Returns the Maximal Updated Density or MUD point as the parameter
        sample from the initial distribution with the highest update density
        value:

        .. math::
            \\lambda^{MUD} := \\mathrm{argmax} \\pi_{up}(\\lambda)
            :label: mud

        Note if the updated distribution has not been computed yet, this
        function will call :meth:`fit` to compute it.

        Parameters
        ----------

        Returns
        -------
        mud_point : np.ndarray
            Maximal Updated Density (MUD) point.
        """
        self.solve()
        m = np.argmax(self.state['pi_up'])
        mud_pt = get_df(self.state.loc[[m]], 'lam', size=self.n_params)[0]
        return mud_pt

    def aggregate(self):
        """
        Build QoI Map Using Data and Measurements

        Aggregate q_lam data with observed data for MUD convergence.
        """
        if f'q_{self.method}_0' in self.state.columns:
            return
        # TODO: Verify/reshape data appropriately
        # TODO: sub sample data/qoi 
        residuals = np.subtract(self.data.T, self.q_lam) / self.std_dev
        n_qoi = 1
        if self.method == "wme":
            qoi = (np.sum(residuals, axis=1) /
                   np.sqrt(self.n_samples)).reshape(-1, 1)
        elif self.method == "pca":
            # Learn qoi to use using PCA
            pca_res, X_train = pca(residuals, n_components=self.max_nc)
            self.pca = {
                "X_train": X_train,
                "vecs": pca_res.components_,
                "var": pca_res.explained_variance_,
            }
            qoi = residuals @ pca_res.components_.T
            n_qoi = self.max_nc

        self.q_lam = qoi
        self.state = put_df(self.state, f'q_{self.method}', qoi, size=n_qoi)

    def plot_param_state(
        self,
        true_vals=None,
        ax=None,
        param_idxs=None,
        plot_initial=False,
        plot_legend=True,
    ):
        """
        Plotting functions for DCI Problem Class
        """
        mud_pt = self.mud_point()
        ax = super().plot_param_state(true_vals=true_vals,
                               ax=ax,
                               param_idxs=param_idxs,
                               mud_val=mud_pt,
                               plot_initial=plot_initial,
                               plot_legend=plot_legend)

    def plot_obs_state(
        self,
        ax=None,
        state_idxs=None,
        plot_pf=False,
        plot_legend=True,
        obs_label='q_lam',
    ):
        """
        Plotting function for DCI Problem Class
        """
        ax = super().plot_obs_state(
                ax=ax,
                state_idxs=state_idxs,
                plot_pf=plot_pf,
                plot_legend=plot_legend,
                obs_label=f'q_{self.method}')
