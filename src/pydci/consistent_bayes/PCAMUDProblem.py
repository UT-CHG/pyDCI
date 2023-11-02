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
from pydci.utils import KDEError, closest_factors, fit_domain, get_df, put_df, set_shape

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
        weights=None,
    ):
        self.init_prob(samples, data, std_dev, pi_in=pi_in, weights=weights)

    @property
    def n_qoi(self):
        return self.qoi.shape[1]

    def init_prob(self, samples, data, std_dev, pi_in=None, weights=None):
        """
        Initialize problem

        Initialize problem by setting the lambda samples, the values of the
        samples pushed through the forward map, and the observe distribution
        on the data. Can optionally as well set the initial and predicteed
        distributions explicitly, and pass in weights to incorporate prior
        beliefs on the `lam` sample sets.
        """
        # Assume gaussian error around mean of data with assumed noise
        super().init_prob(
            samples,
            data,
            std_dev,
            pi_in=pi_in,
            pi_pr=None,
            weights=weights,
        )
        self.qoi = self.q_lam

    def q_pca(self, mask=None, max_nc=None):
        """
        Build QoI Map Using Data and Measurements

        Aggregate q_lam data with observed data for MUD convergence.
        TODO: Cache the results of this for when it is repeatedly computed.
        """
        mask = np.arange(self.n_qoi) if mask is None else mask
        residuals = np.subtract(self.data[mask].T, self.qoi[:, mask]) / self.std_dev
        max_nc = self.n_params if max_nc is None else max_nc
        min_shape = min(residuals.shape)
        max_nc = max_nc if max_nc < min_shape else min_shape

        # Standarize and perform linear PCA
        logger.debug(f"Computing PCA using {max_nc} components")
        sc = StandardScaler()
        pca = PCA(n_components=max_nc)
        X_train = pca.fit_transform(sc.fit_transform(residuals))
        self.pca = {
            "X_train": X_train,
            "vecs": pca.components_,
            "var": pca.explained_variance_,
        }
        logger.debug(f"PCA Variance: {pca.explained_variance_}")

        # Compute Q_PCA
        self.q_lam = residuals @ pca.components_.T
        self.state = put_df(self.state, "q_pca", self.q_lam, size=max_nc)

    def solve(
        self,
        pca_mask: List[int] = None,
        pca_components: List[int] = [0],
    ):
        """
        Solve the parameter estimation problem

        This extends the `MUDProblem` solution class by using the `q_pca()` map
        to aggregate data between the observed and predicted values and
        determine the best MUD estimate that fits the data.

        Parameters
        ----------
        pca_mask: List[int], default=None
            Used control what subset of the observed data is used in the data
            constructed map `q_pca()`
        pca_components: List[int], default=[0]
            Used control what subset of pca components are used in Q_PCA map.
        max_nc: int, default=None
            Specifies the max number of principal components to use when doing
            the PCA transformation on the residuals between the observed and
            simulated data. If not specified, defaults to the min of the number
            of states and the number of parameters.
        """
        pca_components = (
            [pca_components] if isinstance(pca_components, int) else pca_components
        )
        self.q_pca(mask=pca_mask)
        all_qoi = self.q_lam
        self.q_lam = self.q_lam[:, pca_components]
        self.dists["pi_obs"] = norm(loc=len(pca_components) * [0], scale=1)
        self.dists["pi_pr"] = None
        try:
            super().solve()
        except ZeroDivisionError as z:
            logger.error(
                f"({pca_mask}: {pca_components}): "
                + "Predictabiltiy assumption violated"
            )
            raise z
        except KDEError as k:
            logger.error(
                f"({pca_mask}: {pca_components}): "
                + "Unable to perform kernel density estimates"
            )
            raise k
        else:
            self.result["pca_components"] = str(pca_components)
            self.result["pca_mask"] = str(pca_mask)
            self.q_lam = all_qoi

    def plot_L(
        self,
        df=None,
        lam_true=None,
        param_idx=0,
        param_col="lam",
        ratio_col="ratio",
        weight_col="weight",
        plot_legend=True,
        initial_kwargs={},
        update_kwargs={},
        mud_kwargs={},
        ax=None,
        figsize=(6, 6),
    ):
        """
        Plot Lambda Space Distributions

        Plot distributions over parameter space. This includes the initial and
        the updated distributions. Extends `MUDProblem` methods by allowing
        an `nc` argument to plot the solutions that use only `nc` number of
        principal components in the `q_pca` map. If `nc` is not specified, the
        best choice of `nc` will be picked from those tried, with the best
        being chosen according to the `best_method` set in `solve()`.

        Parameters
        ----------
        param_idx : int, default=0
            Index of parameter, `lam` to plot.
        lam_true: ArrayLike, default=None
            If specified, a vertical line for the true parameter solution will
            be added. Note this value must be the same dimension as the
            parameter space, even if it only the value at the `param_idx`
            specified is only used.
        plot_mud: bool, default=True
            Whether to add a vertical line for the computed MUD point solution.

        Returns
        -------
        ax, labels : Tuple
            Tuple of (1) matplotlib axis object where distributions where
            plotted and (2) List of labels that were plotted, in order plotted.
        """

        ax, labels = super().plot_L(
            lam_true=lam_true,
            df=df,
            param_idx=param_idx,
            param_col=param_col,
            ratio_col=ratio_col,
            weight_col=weight_col,
            plot_legend=plot_legend,
            initial_kwargs=initial_kwargs,
            update_kwargs=update_kwargs,
            mud_kwargs=mud_kwargs,
            ax=ax,
            figsize=figsize,
        )

        return ax, labels

    def plot_D(
        self,
        df=None,
        state_idx=0,
        state_col="q_pca",
        ratio_col="ratio",
        weight_col="weight",
        plot_obs=True,
        plot_pf=True,
        plot_legend=True,
        ax=None,
        figsize=(6, 6),
    ):
        """
        Plot Q(lambda) Space Distributions

        Plot distributions over observable space `q_lam`. This includes the
        observed distribution `pi_obs`, the predicted distribtuion `pi_pr`, and
        the push-forward of the updated distribution `pi_pf`. Extends
        `MUDProblem` methods by allowing an `nc` argument to plot the solutions
        that use only `nc` number of principal components in the `q_pca` map.
        If `nc` is not specified, the best choice of `nc` will be picked from
        those tried, with the best being chosen according to the `best_method`
        set in `solve()` (and corresponding to the solution stored in the
        `result` class attribute). See parent class for more info on relevant
        parameters.

        Parameters
        ----------
        nc: int, default=None
            Plot solution that only uses `nc` components. By default will plot
            the solution that corresponds to the best as determined in the
            `solve()` method and stored in the `result` attribute.

        Returns
        -------
        ax, labels : Tuple
            Tuple of (1) matplotlib axis object where distributions where
            plotted and (2) List of labels that were plotted, in order plotted.
        """
        if df is None:
            df = self.state

        ax, labels = super().plot_D(
            df=df,
            state_idx=state_idx,
            state_col=state_col,
            ratio_col=ratio_col,
            weight_col=weight_col,
            plot_obs=False,
            plot_pf=plot_pf,
            plot_legend=plot_legend,
            ax=ax,
            figsize=figsize,
        )

        if plot_obs:
            # Plot N(0, 1) distribution over axis range using seaborn
            x = np.linspace(*ax.get_xlim(), 100)
            y = norm.pdf(x, loc=0, scale=1)
            sns.lineplot(
                x=x,
                y=y,
                ax=ax,
                label="$\pi^\mathrm{obs}_\mathcal{D} = \mathcal{N}(0, 1)$",
            )

        return ax, labels

    def density_plots(
        self,
        nc=None,
        lam_true=None,
        lam_kwargs=None,
        q_lam_kwargs=None,
        axs=None,
        figsize=(14, 6),
    ):
        """
        Plot param and observable space onto sampe plot
        """
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        elif len(axs) != 2:
            len(axs) != self.n_params
        lam_kwargs = {} if lam_kwargs is None else lam_kwargs
        q_lam_kwargs = {} if q_lam_kwargs is None else q_lam_kwargs
        lam_kwargs["ax"] = axs[0]
        q_lam_kwargs["ax"] = axs[1]
        self.plot_L(**lam_kwargs)
        self.plot_D(**q_lam_kwargs)
        lam_true = lam_kwargs.get("lam_true", None)
        fig = axs[0].get_figure()
        fig.suptitle(
            self._parse_title(
                result=self.result,
                lam_true=lam_true,
            )
        )
        fig.tight_layout()

        return axs

    def param_density_plots(
        self,
        lam_true=None,
        base_size=4,
        max_np=9,
        **kwargs,
    ):
        base_size = 4
        n_params = self.n_params if self.n_params <= max_np else max_np
        grid_plot = closest_factors(n_params)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, ax in enumerate(ax.flat):
            self.plot_L(param_idx=i, lam_true=lam_true, ax=ax, **kwargs)
            ax.set_title(f"$\lambda_{{{i}}}$")

        fig.suptitle(
            self._parse_title(
                result=self.result,
                nc=True,
                lam_true=lam_true,
            )
        )
        fig.tight_layout()

    def learned_qoi_plot(self, nc_mask=None):
        """
        Scatter plots of learned `q_pca` components.
        """
        max_nc = len(self.pca["vecs"])
        nc_mask = np.arange(max_nc) if nc_mask is None else nc_mask
        max_nc = self.n_states if self.n_states <= max_nc else max_nc
        g = sns.PairGrid(self.state[[f"q_pca_{i}" for i in nc_mask]], diag_sharey=False)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot)

    def _parse_title(
        self,
        result=None,
        lam_true=None,
    ):
        """
        Parse title for plots
        """
        result = self.result if result is None else result
        title = super()._parse_title(result=result, lam_true=lam_true)

        return title
