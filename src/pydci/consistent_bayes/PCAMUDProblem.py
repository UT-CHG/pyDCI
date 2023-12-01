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

import pydci.notation as dcin
from pydci.consistent_bayes.MUDProblem import MUDProblem
from pydci.log import disable_log, enable_log, log_table, logger
from pydci.utils import (KDEError, closest_factors, fit_domain,
                         get_df, put_df, set_shape, print_rich_table)


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
        - Find a way to store pca vectors?

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
    
    def __str__(self) -> str:
        info_df = self.get_info_table()
        return print_rich_table(
            info_df,
            columns=['num_samples', 'num_params', 'num_qoi', 'num_states',
                     'mem_usage', 'solved', 'lam_mud', 'mud_idx', 'e_r', 'kl', 'error'],
            vertical=True,
            title='PCAMUDProblem',
        )
    
    def get_info_table(self) -> pd.DataFrame:
        """
        Get Info Table
        """
        info_df = super().get_info_table()
        info_df['num_qoi'] = str(self.n_qoi)

        return info_df

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
        self.pca = None

    def save_state(self, vals):
        """
        Save current state, adding columns with values in vals dictionary
        """
        super().save_state(vals)

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
        pr_kwargs={},
        obs_kwargs={},
        pf_kwargs={},
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
            pr_kwargs=pr_kwargs,
            obs_kwargs=None,
            pf_kwargs=pf_kwargs,
            plot_legend=False if obs_kwargs is not None else plot_legend,
            ax=ax,
            figsize=figsize,
        )

        if obs_kwargs is not None:
            # Plot N(0, 1) distribution over axis range using seaborn
            x = np.linspace(*ax.get_xlim(), 100)
            y = norm.pdf(x, loc=0, scale=1)
            obs_args = dict(
                x=x,
                y=y,
                ax=ax,
                color='r',
                label= dcin.pi('ob') +  "$ = \mathcal{N}(0, 1)$",
            )
            obs_args.update(obs_kwargs)
            ax = sns.lineplot(
                **obs_args
            )
            labels.append(obs_args["label"])
        
        # Set plot specifications
        ax.set_xlabel(dcin.q(idx=state_idx))
        # ax.set_xlabel(fr"$\mathbf{{q}}_{state_idx}$")
        # Center x axis around 0, +- abs(max/min x value)
        lims = ax.get_xlim()
        max_range = np.max([np.abs(lims[0]), np.abs(lims[1])])
        ax.set_xlim([-max_range, max_range])
        
        if plot_legend:
            ax.legend(
                labels=labels,
                fontsize=12,
                title_fontsize=12,
            )

        return ax, labels

    def param_space_scatter_plot(
        self,
        param_x=0,
        param_y=1,
        type='qoi',
        state_idx=0,
        cb_range=None,
        weighted=False,
        figsize=(6,6),
        ax=None,
        **kwargs,
    ):
        """
        Create scatter plots of the parameter space, contoured by learned QoI or update ratio.

        Parameters
        ----------
        param_x : int, optional
            The index of the parameter to plot on the x-axis, by default 0.
        param_y : int, optional
            The index of the parameter to plot on the y-axis, by default 1.
        type : str, optional
            The type of plot to create, either 'qoi' for QoI or 'ratio' for update ratio, by default 'qoi'.
        state_idx : int, optional
            The index of the state to plot, by default 0.
        weighted : bool, optional
            Whether to plot the weighted update ratio, by default False.
        figsize : tuple, optional
            The figure size, by default (6,6).
        ax : matplotlib.axes, optional  
            The axis to plot on, by default None.

        Returns
        -------
        None
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if type == 'qoi':
            hue_col = f'q_pca_{state_idx}'
        elif type == 'ratio':
            hue_col = 'ratio' if not weighted else 'weighted_ratio'
        else:
            raise ValueError(f"Invalid type: {type}")

        if cb_range is not None:
            vmin, vmax = cb_range
        else:
            vmin = self.state[hue_col].min()
            vmax = self.state[hue_col].max()
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )
        sm._A = []

        plot_args = dict(
            x=self.state[f"lam_{param_x}"],
            y=self.state[f"lam_{param_y}"],
            hue=self.state[hue_col],
            palette="viridis",
            ax=ax,
            s=100,
            hue_norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )
        plot_args.update(kwargs)

        sns.scatterplot(**plot_args)
        dcin
        ax.set_xlabel(dcin.lam(idx=param_x))
        ax.set_ylabel(dcin.lam(idx=param_y))
        if type == 'qoi':
            title = 'Learned QoI '
            title += dcin.q() + '$=$'
            title += dcin.q_pca(arg=dcin.lam())  
        else:
            title = 'Update Ratio '
        ax.set_title(title, pad=10)
        ax.get_legend().remove()

        # Add colorbar
        fig = plt.gcf()
        cbar_ax = fig.add_axes([
            ax.get_position().x1+0.01,
            ax.get_position().y0,
            0.03, ax.get_position().height])
        fig.colorbar(sm, cax=cbar_ax)

        if type == 'qoi':
            title = dcin.q(idx=state_idx)
            # title = rf'$q_{state_idx}(\mathbf{{\lambda}})$'
        else:
            if weighted:
                title = rf'$w*r(\mathbf{{\lambda}})$'
            else:
                title = rf"$r(\mathbf{{\lambda}})$"
        cbar_ax.set_title(title)

        return ax
    
    def plot_qoi_over_params(
        self,
        df=None,
        param_x=0,
        param_y=1,
        state_idxs=None,
        weighted=False,
        same_colorbar=True,
        scatter_kwargs=None,
        cbar_pos=[0.01, 0.03],
        axs=None,
    ):
        """
        Create scatter plots over the parameter space, contoured by different learned QoI.

        Parameters
        ----------
        state_idx : int, optional
            The index of the state to plot, by default 0.
        weighted : bool, optional
            Whether to plot the weighted update ratio, by default False.
        same_colorbar : bool, optional
            Whether to use the same colorbar for all subplots, by default True.

        Returns
        -------
        None
        """
        if self.n_params < 2:
            raise ValueError("Must have at least 2 parameters to plot") 

        if len(cbar_pos) != 2:
            raise ValueError(f"cbar_pos must be of length 2: {cbar_pos}")
        
        df = self.state if df is None else df

        pca_cols = [col for col in df.columns if col.startswith("q_pca")]
        if state_idxs is not None:
            # only plot states in state_idxs list
            pca_cols = [col for col in pca_cols if int(col.split("_")[-1]) in state_idxs]
            if len(state_idxs) > len(pca_cols):
                not_in = [i for i in state_idxs if i not in range(self.n_states)]
                raise ValueError(f"state_idxs must be a subset of the number of states: {not_in}")
        else:
            state_idxs = range(len(pca_cols))

        if axs is None:
            base_size = 5
            grid_plot = closest_factors(len(state_idxs))
            fig, axs = plt.subplots(
                grid_plot[0],
                grid_plot[1],
                figsize=(grid_plot[1] * (base_size), grid_plot[0] * (base_size)),
                sharex=True,
                sharey=True,
            )
        else:
            if len(axs) != len(state_idxs):
                raise ValueError(f"axs must be of length states: {len(axs)} != {len(state_idxs)}")
            fig = plt.gcf()

        vmin = np.inf
        vmax = -np.inf
        if same_colorbar:
            # Get vmin accros q_pca_cols
            vmin = df[pca_cols].min().min()
            vmax = df[pca_cols].max().min()
            sm = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
            )
            sm._A = []
        else:
            vmin = np.inf
            vmax = -np.inf

            
        for i, ax in enumerate(axs.flat):
            hue_col = f'q_pca_{state_idxs[i]}'
            
            if not same_colorbar:
                vmin = df[hue_col].min()
                vmax = df[hue_col].max()
                sm = plt.cm.ScalarMappable(
                    cmap="viridis",
                    norm=plt.Normalize(vmin=vmin, vmax=vmax),
                )
                sm._A = []
            kwargs = dict(
                x=df[f"lam_{param_x}"],
                y=df[f"lam_{param_y}"],
                hue=df[hue_col],
                palette="viridis",
                ax=ax,
                s=100,
                hue_norm=plt.Normalize(vmin=vmin, vmax=vmax),   
            ) 
            kwargs.update(scatter_kwargs or {})
            sns.scatterplot(**kwargs)
            ax.set_xlabel(dcin.lam(idx=param_x))
            ax.set_ylabel(dcin.lam(idx=param_y))
            ax.set_title(dcin.q(idx=state_idxs[i]))
            ax.get_legend().remove()
            if not same_colorbar:
                cbar_loc = [ax.get_position().x1+cbar_pos[0],
                     ax.get_position().y0,
                     cbar_pos[1], ax.get_position().height]
                cbar_ax = fig.add_axes(cbar_loc)
                fig.colorbar(sm, cax=cbar_ax)
                cbar_ax.set_title(dcin.q(idx=state_idxs[i]))

        if same_colorbar:
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
            fig.colorbar(sm, cax=cbar_ax)
            title = dcin.q_pca(idx='j')
            cbar_ax.set_title(title)

        return axs
    
    def learned_qoi_plots(
        self,
        df=None,
        state_x=0,
        state_y=1,
        hue_cols=['weighted_ratio'],
        same_colorbar=True,
        scatter_kwargs=None,
        cbar_pos=[0.01, 0.03],
        axs=None,
    ):
        """
        Create scatter plots of the learned_qoi space.

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.n_states < 2:
            raise ValueError("Must have at least 2 parameters to plot") 

        if len(cbar_pos) != 2:
            raise ValueError(f"cbar_pos must be of length 2: {cbar_pos}")
        
        df = self.state if df is None else df

        bad_cols = [col for col in hue_cols if col not in df.columns]
        if len(bad_cols) > 0:
            raise ValueError(f"Invalid cols to color qoi by: {bad_cols}")
        if axs is None:
            base_size = 5
            grid_plot = closest_factors(len(hue_cols))
            fig, axs = plt.subplots(
                grid_plot[0],
                grid_plot[1],
                figsize=(grid_plot[1] * (base_size), grid_plot[0] * (base_size)),
                sharex=True,
                sharey=True,
            )
        else:
            if len(axs) != len(hue_cols):
                raise ValueError(f"axs must be of length different hue_cols: {len(axs)} != {len(hue_cols)}")
            fig = plt.gcf()

        vmin = np.inf
        vmax = -np.inf
        if same_colorbar:
            # Get vmin accros q_pca_cols
            vmin = df[hue_cols].min().min()
            vmax = df[hue_cols].max().min()
            sm = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
            )
            sm._A = []
        else:
            vmin = np.inf
            vmax = -np.inf

            
        ax_list = [axs] if len(hue_cols) == 1 else axs.flat
        for i, ax in enumerate(ax_list):
            hue_col = hue_cols[i]
            
            if not same_colorbar:
                vmin = df[hue_col].min()
                vmax = df[hue_col].max()
                sm = plt.cm.ScalarMappable(
                    cmap="viridis",
                    norm=plt.Normalize(vmin=vmin, vmax=vmax),
                )
                sm._A = []
            kwargs = dict(
                x=df[f"q_pca_{state_x}"],
                y=df[f"q_pca_{state_y}"],
                hue=df[hue_col],
                palette="viridis",
                ax=ax,
                s=100,
                hue_norm=plt.Normalize(vmin=vmin, vmax=vmax),   
            ) 
            kwargs.update(scatter_kwargs or {})
            sns.scatterplot(**kwargs)
            ax.set_xlabel(dcin.q(idx=state_x))
            ax.set_ylabel(dcin.q(idx=state_y))
            # ax.set_title(dcin.q(idx=state_idxs[i]))
            ax.get_legend().remove()
            if not same_colorbar:
                cbar_loc = [ax.get_position().x1+cbar_pos[0],
                        ax.get_position().y0,
                        cbar_pos[1], ax.get_position().height]
                cbar_ax = fig.add_axes(cbar_loc)
                fig.colorbar(sm, cax=cbar_ax)
                if hue_col == 'weighted_ratio':
                    # TODO: Add weighted ratio option
                    cbar_ax.set_title(r'$wr$')
                elif hue_col == 'ratio':
                    cbar_ax.set_title(r'$r$')

        if same_colorbar:
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
            fig.colorbar(sm, cax=cbar_ax)
            if hue_col == 'weighted_ratio':
                # TODO: Add weighted ratio option
                cbar_ax.set_title(r'$wr$')
            elif hue_col == 'ratio':
                cbar_ax.set_title(r'$r$')

        return axs

    
    def learned_qoi_plot_joint_plot(self, nc_mask=None):
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
        title=None,
    ):
        """
        Parse title for plots
        """
        result = self.result if result is None else result
        title = super()._parse_title(result=result, lam_true=lam_true, title=title)

        return title

#     def density_plots(
#         self,
#         param_idx=0,
#         state_idx=0,
#         lam_true=None,
#         lam_kwargs=None,
#         q_lam_kwargs=None,
#         axs=None,
#         figsize=(14, 6),
#     ):
#         """
#         Plot param and observable space onto sampe plot
#         """
#         if axs is None:
#             fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#         elif len(axs) != 2:
#             len(axs) != self.n_params
#         lam_kwargs = {} if lam_kwargs is None else lam_kwargs
#         q_lam_kwargs = {} if q_lam_kwargs is None else q_lam_kwargs
#         lam_kwargs["param_idx"] = param_idx
#         lam_kwargs["ax"] = axs[0]
#         q_lam_kwargs["ax"] = axs[1]
#         q_lam_kwargs["state_idx"] = state_idx
#         self.plot_L(**lam_kwargs)
#         self.plot_D(**q_lam_kwargs)
#         lam_true = lam_kwargs.get("lam_true", None)
#         fig = axs[0].get_figure()
#         fig.suptitle(
#             self._parse_title(
#                 result=self.result,
#                 lam_true=lam_true,
#             )
#         )
#         # fig.tight_layout()
# 
#         return axs
# 
#    def param_density_plots(
#        self,
#        lam_true=None,
#        base_size=4,
#        max_np=9,
#        **kwargs,
#    ):
#        base_size = 4
#        n_params = self.n_params if self.n_params <= max_np else max_np
#        grid_plot = closest_factors(n_params)
#        fig, ax = plt.subplots(
#            grid_plot[0],
#            grid_plot[1],
#            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
#        )
#
#        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
#        for i, ax in enumerate(ax.flat):
#            self.plot_L(param_idx=i, lam_true=lam_true, ax=ax, **kwargs)
#            ax.set_title(f"$\lambda_{{{i}}}$")
#
#        fig.suptitle(
#            self._parse_title(
#                result=self.result,
#                lam_true=lam_true,
#            )
#        )
#        # fig.tight_layout()