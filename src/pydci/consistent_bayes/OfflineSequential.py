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
    - Sequential (should be renamed split sequential?) - Plot destribution
    methods for plotting distributions per iteration, qoi combinations,
    or pca values
    Dynamic Problem -> Finish

"""
import pdb
import random
from itertools import cycle
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import alive_bar
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike
from rich.table import Table
from scipy.stats import rv_continuous, entropy  # type: ignore
from scipy.stats.distributions import norm
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from pydci import PCAMUDProblem
from pydci.log import disable_log, enable_log, log_table, logger
from pydci.utils import KDEError, closest_factors, fit_domain, get_df, put_df, set_shape, print_rich_table
import pydci.notation as dcin

sns.color_palette("bright")
sns.set_style("darkgrid")

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"


class OfflineSequential(PCAMUDProblem):
    """
    Offline Sequential Estimation

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
            
    def __str__(self) -> str:
        info_df = self.get_info_table()
        return print_rich_table(
            info_df,
            columns=['num_samples', 'num_params', 'num_qoi', 'num_states', 'num_iter',
                     'mem_usage', 'solved', 'pca_components', 'pca_mask', 'lam_mud', 'mud_idx', 'e_r', 'kl', 'error'],
            vertical=True,
            title='OfflineSequential',
        )
    
    @property
    def n_iters(self):
        """
        Number of curent iterations
        """
        if self.states is None:
            return 0
        return self.states['iteration'].max()
    
    def get_info_table(self) -> pd.DataFrame:
        """
        Get Info Table
        """
        info_df = super().get_info_table()
        info_df['num_iter'] = self.n_iters
        if self.result is not None:
            info_df['pca_components'] = self.result['pca_components'].values[0]
            info_df['pca_mask'] = self.result['pca_mask'].values[0]
        return info_df

    def solve(
        self,
        pca_components=1,
        pca_mask: List[int] = None,
        pca_splits: List[int] = 1,
        exp_thresh: float = 0.5,
        fail_on_partial: bool = True,
    ):
        """
        Solve the parameter estimation problem

        This extends the `MUDProblem` solution class by using the `q_pca()` map
        to aggregate data between the observed and predicted values and
        determine the best MUD estimate that fits the data.

        Parameters
        ----------
        """
        if exp_thresh <= 0:
            msg = f"Expected ratio thresh must be a float > 0: {exp_thresh}"
            logger.error(msg)
            raise ValueError(msg)

        num_splits = 1
        pca_components = [list(range(pca_components))] if isinstance(
            pca_components, int) else pca_components
        if isinstance(pca_splits, int) or pca_splits is None:
            # Make even number of splits of all qoi if mask is not specified
            pca_mask = np.arange(self.n_qoi) if pca_mask is None else pca_mask
            num_splits = pca_splits if pca_splits is not None else 1
            pca_splits = [
                range(x[0], x[-1] + 1) for x in np.array_split(pca_mask, num_splits)
            ]
        elif isinstance(pca_splits, list):
            num_splits = len(pca_splits)
            if pca_mask is not None:
                raise ValueError(
                    "Cannot specify both pca_mask and non-integer pca_splits"
                )
        iterations = [(i, j) for i in pca_splits for j in pca_components]
        if len(iterations) == 0:
            raise ValueError(f"No iterations specified: {pca_splits}, {pca_mask}")

        weights = []
        prev_in = self.dists["pi_in"]
        it_results = []
        for i, (pca_mask, pca_cs) in enumerate(iterations):
            str_val = pca_mask if pca_mask is not None else "ALL"
            logger.debug(f"Iteration {i}: Solving using ({str_val}, {pca_cs})")

            reason = None
            try:
                super().solve(
                    pca_mask=pca_mask,
                    pca_components=pca_cs,
                )
            except ZeroDivisionError as z:
                reason = str(z)
            except KDEError as k:
                reason = f"({i}): KDE estimation failed:\n{str(k)}"
            except LinAlgError as l:
                reason = f"({i}): LinalError on pi_in, pi_pr distributions:\n{str(l)}"
            else:
                e_r = self.result["e_r"].values[0]
                if (diff := np.abs(e_r - 1.0)) > exp_thresh:
                    reason = f"|E(r) - 1| = {diff} > {exp_thresh} - Stopping"

            self.save_state({'iteration': i + 1})
            it_results.append(self.result.copy())
            it_results[-1]["i"] = len(it_results)
            it_results[-1]["I"] = int(num_splits)

            if reason is not None:
                if i == 0 or fail_on_partial:
                    reason = f"Failed on iteration {i + 1}:\n{reason}"
                    it_results[-1]["solved"] = False
                    it_results[-1]["error"] = reason
                else:
                    logger.debug(f"Resetting to last solution at {iterations[i-1]}")
                    self.set_weights(weights[:-1])
                    self.dists["pi_in"] = prev_in
                    self.solve(
                        pca_mask=iterations[i - 1][0],
                        pca_components=[iterations[i - 1][1]],
                    )
                    solved_idx = i
                    it_results.append(self.result.copy())
                    it_results[-1]["i"] = solved_idx
                    it_results[-1]["I"] = int(num_splits)
                    reason = None
                break
            else:
                if i != len(iterations) - 1:
                    logger.debug("Updating weights")
                    weights.append(self.state["ratio"].values)
                    prev_in = self.dists["pi_in"]
                    self.set_weights(weights)

        self.results = it_results
        self.result = self.results[-1]

        if reason is not None:
            logger.error(reason)
            raise RuntimeError(reason)

    def expected_ratio(self, iteration: int = -1) -> float:
        """Expectation Value of R

        Returns the expectation value of the R, the ratio of the observed to
        the predicted density values.

        If the predictability assumption for the data-consistent framework is
        satisfied, then this values should be close to 1.0.

        Returns
        -------
        expected_ratio : float
            Value of the E(r). Should be close to 1.0.
        """
        df = self.get_iteration_state(iteration=iteration)
        return np.average(df["ratio"], weights=df["weight"])

    def divergence_kl(self, iteration: int = -1):
        """KL-Divergence Between observed and predicted.

        Parameters
        ----------

        Returns
        -------
        kl: float
            Value of the kl divergence.
        """
        df = self.get_iteration_state(iteration=iteration)
        return entropy(df["pi_obs"], df["pi_pr"])

    def get_mud_point(self, state_df: pd.DataFrame = None, iteration: int = -1) -> tuple:
        """
        Get MUD Point from DataFrame.

        Get MUD point from DataFrame. If DataFrame is not passed in, use the
        `result` attribute of the class.

        Parameters
        ----------
        iteration : int, optional
            The iteration to get the MUD point from. If not provided, the method
            will use the last iteration.
        state_df : DataFrame, optional
            The state DataFrame. If not provided, the method will use the `state`
            attribute of the class.

        Returns
        -------
        tuple
            A tuple containing the index of the maximum "pi_up" value in the
            DataFrame and the MUD point calculated using the `get_df` method
            with the "lam" column and `n_params` attribute of the class.

        """
        state_df = state_df if state_df is not None \
            else self.get_iteration_state(iteration=iteration)

        m = np.argmax(state_df["pi_up"])
        mud_point = get_df(state_df.iloc[[m]], "lam", size=self.n_params)

        return m, mud_point

    def get_iteration_state(self, iteration=-1):
        """
        Retrieve the state of the system at the specified iteration

        Note states are one-indexed in the state table, but assuming arguments here are 0 indexed. 
        Subtract one from states in the table to get.
        Pass in -1 to get the current iteration.

        TODO: Review the indexing here
        """
        if len(self.states) == 0 or iteration == -1:
            df = self.state
        else:
            iterations = self.states["iteration"].unique()
            if iteration not in iterations:
                msg = f'Iteration {iteration} not in saved states: {iterations}'
                logger.error(msg)
                raise ValueError(msg)
            df = self.states[self.states["iteration"] == iterations[iteration - 1]]

        return df

    def plot_L(
        self,
        iteration=-1,
        lam_true=None,
        df=None,
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
        df = df if df is not None else self.get_iteration_state(iteration=iteration)

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
        iteration=-1,
        df=None,
        state_idx=0,
        state_col="q_pca",
        ratio_col="ratio",
        weight_col="weight",
        pr_kwargs={},
        pf_kwargs={},
        obs_kwargs={},
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
        df = df if df is not None else self.get_iteration_state(iteration=iteration)

        ax, labels = super().plot_D(
            df=df,
            state_idx=state_idx,
            state_col=state_col,
            ratio_col=ratio_col,
            weight_col=weight_col,
            pr_kwargs=pr_kwargs,
            obs_kwargs=obs_kwargs,
            pf_kwargs=pf_kwargs,
            plot_legend=plot_legend,
            ax=ax,
            figsize=figsize,
        )

        return ax, labels

    def plot_iterations(
        self,
        param_idx=0,
        iterations=None,
        lam_true=None,
        shade=True,
        color=None,
        linestyle=None,
        ax=None,
    ):
        """
        Plot PCA iterations.

        Plots the initial distribution, the iterative updates, and the final solution
        as stored in the self.results and self.states attributes of the
        PCAMUDselflem object, which are updated during a solve() call.
        The iterative updates correspond to using a re-weighted sequential data-consistent
        update, also known as "offline" sequential estimation, since iterations
        are performed on a static set of data/simulations.

        Parameters
        ----------
        param_idx : int
            Index of the parameter to plot.
        iterations : list of int
            List of iterations to plot. If None, iterations are not plotted, just initial and final.
        lam_true : list of float
            True value of the parameter. Plotted as vertical orange line. If None, no true value is plotted.
        shade : bool
            If True, the iterative updates will be plotted with increasing transparancy as the iteration number
            increases. Otherwise all plotted with the same transparancy (alpha).
        color : str
            Color to use for all lines. If None, a color palette is used, so all lines are different colors.
        linestyle : str
            Linestyle to use for all lines. If None, linestyles are cycled through ['--','-.'] for the
            iterative updates. Initial is always plotted with ':' and final with '-'.
        ax :
            Matplotlib axes object to use. If None, a new figure and axes are created.

        Results
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes object used.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot initial distribution
        _, labels = self.plot_L(
            param_idx=param_idx,
            iteration=1,
            initial_kwargs={
                "color": "black",
                "linestyle": ":",
                "label": "$\pi^\mathrm{init}$",
            },
            update_kwargs=None,
            mud_kwargs=None,
            lam_true=None,
            ax=ax,
        )

        iterations = [] if iterations is None else iterations
        if isinstance(color, str):
            colors = [color] * (len(iterations) + 1)
        else:
            colors = sns.color_palette(None, len(iterations) + 1)

        # Plot iterative updates, for each iteration specified
        if len(iterations) > 0:
            ls = ["--", "-."]
            ls = [linestyle] * (len(iterations) + 1) if linestyle is not None else ls
            linecycler = cycle(ls)

            alphas = (
                np.ones(len(iterations))
                if not shade
                else np.linspace(0.1, 0.9, len(iterations))
            )
            line_opts = {"fill": False}
            for i, it in enumerate(iterations):
                line_opts["alpha"] = alphas[i]
                line_opts['label'] = dcin.pi('up', iteration=i, idx=param_idx)
                line_opts["color"] = colors[i]
                line_opts["linestyle"] = next(linecycler)
                _, l = self.plot_L(
                    param_idx=param_idx,
                    iteration=it + 1,
                    initial_kwargs=None,
                    update_kwargs=line_opts,
                    mud_kwargs=None,
                    lam_true=None,
                    ax=ax,
                )
                labels += l

        # Plot final solution, with mud argument and true lambda
        line_opts = {"fill": True}
        line_opts['label'] = dcin.pi('up', iteration=i, idx=param_idx)
        line_opts["linestyle"] = "-"
        line_opts["color"] = colors[-1]
        mud_args = {"alpha": 1.0, "linestyle": "--", "label": dcin.mud_pt()}
        # if 'color' in kwargs.keys():
        #     mud_args['color'] = kwargs['color']
        _, l = self.plot_L(
            param_idx=param_idx,
            iteration=len(self.results),
            initial_kwargs=None,
            update_kwargs=line_opts,
            mud_kwargs=mud_args,
            lam_true=lam_true,
            ax=ax,
        )
        labels += l
        ax.legend(labels=labels, loc="upper right", fontsize=14)

        return ax

    def param_density_plots(
        self,
        lam_true=None,
        base_size=4,
        max_np=9,
        ax=None,
        **kwargs,
    ):
        base_size = 4
        n_params = self.n_params if self.n_params <= max_np else max_np
        grid_plot = closest_factors(n_params)
        if ax is None:
            fig, ax = plt.subplots(
                grid_plot[0],
                grid_plot[1],
                figsize=(grid_plot[1] * (base_size + 2), grid_plot[0] * base_size),
            )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, axes in enumerate(ax.flat):
            self.plot_L(param_idx=i, lam_true=lam_true, ax=axes, **kwargs)
            axes.set_title(dcin.lam(idx=i))

        fig = plt.gcf()
        fig.suptitle(
            self._parse_title(
                result=self.result,
                nc=True,
                lam_true=lam_true,
            )
        )

        return ax

    def _parse_title(
        self,
        result=None,
        nc=True,
        lam_true=None,
        title=None,
    ):
        """
        Parse title for plots
        """
        result = self.result if result is None else result
        title = super()._parse_title(result=result, lam_true=lam_true, title=title)
        num_splits = len(self.states["iteration"].unique())
        if nc:
            title = f"# Splits = {num_splits}: " + title

        return title

    def plot_qoi_over_params(
        self,
        param_x=0,
        param_y=1,
        iteration=-1,
        state_idxs=None,
        weighted=False,
        same_colorbar=True,
        scatter_kwargs=None,
        cbar_pos=[0.01, 0.03],
        axs=None,
    ):
        """
        """
        state_df = self.get_iteration_state(iteration=iteration)
        axs = super().plot_qoi_over_params(
            df=state_df,
            param_x=param_x,
            param_y=param_y,
            state_idxs=state_idxs,
            weighted=weighted,
            same_colorbar=same_colorbar,
            scatter_kwargs=scatter_kwargs,
            cbar_pos=cbar_pos,
            axs=axs,
        )

        return axs

    def plot_qoi_over_params_by_iteration(
        self,
        type='qoi',
        param_x=0,
        param_y=0,
        state_idx=0,
        weighted=False,
        same_colorbar=True
    ):
        """
        Create scatter plots of the parameter space over iterations.

        Parameters
        ----------
        type : str, optional
            The type of plot to create, either 'qoi' for QoI or 'ratio' for update ratio, by default 'qoi'.
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
        base_size = 5
        grid_plot = closest_factors(self.n_iters)
        fig, axs = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[1] * (base_size + 2), grid_plot[0] * base_size),
            sharex=True,
            sharey=True,
        )

        if type == 'qoi':
            hue_col = f'q_pca_{state_idx}'
        elif type == 'ratio':
            hue_col = 'ratio' if not weighted else 'weighted_ratio'
        else:
            raise ValueError(f"Invalid type: {type}")

        iteration = 0
        iterations = self.states["iteration"].unique()
        vmin = np.inf
        vmax = -np.inf
        if same_colorbar:
            vmin = min([df[hue_col].min() for idx, df in self.states.groupby('iteration')])
            vmax = max([df[hue_col].max() for idx, df in self.states.groupby('iteration')])
            sm = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
            )
            sm._A = []
        else:
            vmin = np.inf
            vmax = -np.inf

            
        for i, ax in enumerate(axs.flat):
            df = self.states[self.states["iteration"] == iterations[i]]
            
            if not same_colorbar:
                vmin = min(vmin, df[hue_col].min())
                vmax = max(vmax, df[hue_col].max())
                sm = plt.cm.ScalarMappable(
                    cmap="viridis",
                    norm=plt.Normalize(vmin=vmin, vmax=vmax),
                )
                sm._A = []
            sns.scatterplot(
                x=df["lam_0"],
                y=df["lam_1"],
                hue=df[hue_col],
                palette="viridis",
                ax=ax,
                s=100,
                hue_norm=plt.Normalize(vmin=vmin, vmax=vmax),   
            )
            ax.set_xlabel(dcin.lam(idx=param_x))
            ax.set_ylabel(dcin.lam(idx=param_y))
            ax.set_title(f"i = {i + 1}")
            ax.get_legend().remove()
            if not same_colorbar:
                # Add colorbar
                cbar_ax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.03, ax.get_position().height])
                fig.colorbar(sm, cax=cbar_ax)

                if type == 'qoi':
                    title = dcin.q(iteration=i, idx=state_idx)
                else:
                    title = dcin.r(weighted=weighted, iteration=i)

                cbar_ax.set_title(title)

        if same_colorbar:
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
            fig.colorbar(sm, cax=cbar_ax)
            if type == 'qoi':
                title = dcin.q(iteration=i, idx=state_idx)
            else:
                title = dcin.r(weighted=weighted, iteration=i)
            cbar_ax.set_title(title)
        
        return axs

    def learned_qoi_plots(
        self,
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

        bad_cols = [col for col in hue_cols if col not in self.state.columns]
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
            vmin = self.state[hue_cols].min().min()
            vmax = self.state[hue_cols].max().min()
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
                vmin = self.state[hue_col].min()
                vmax = self.state[hue_col].max()
                sm = plt.cm.ScalarMappable(
                    cmap="viridis",
                    norm=plt.Normalize(vmin=vmin, vmax=vmax),
                )
                sm._A = []
            kwargs = dict(
                x=self.state[f"q_pca_{state_x}"],
                y=self.state[f"q_pca_{state_y}"],
                hue=self.state[hue_col],
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

    