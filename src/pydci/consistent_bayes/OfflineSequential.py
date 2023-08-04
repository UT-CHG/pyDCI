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
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle
from alive_progress import alive_bar
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike
from rich.table import Table
from scipy.stats import rv_continuous  # type: ignore
from scipy.stats.distributions import norm
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from pydci import PCAMUDProblem
from pydci.log import disable_log, enable_log, log_table, logger
from pydci.utils import KDEError, fit_domain, get_df, put_df, set_shape, closest_factors

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
    ):
        self.init_prob(samples, data, std_dev, pi_in=pi_in)

    def init_prob(self, samples, data, std_dev, pi_in=None):
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
        )
        self.pca_states = None

    def save_state(self, vals):
        """
        Save current state, adding columns with values in vals dictionary
        """
        keys = vals.keys()
        cols = [c for c in self.state.columns if c.startswith("lam_")]
        cols += [c for c in self.state.columns if c.startswith("q_pca_")]
        cols += ["weight", "pi_in", "pi_obs", "pi_pr", "ratio", "pi_up"]
        state = self.state[cols].copy()
        for key in keys:
            state[key] = vals[key]
        if self.pca_states is None:
            self.pca_states = state
        else:
            self.pca_states = pd.concat([self.pca_states, state], axis=0)

    def solve(
        self,
        weights=None,
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
        weights = [] if weights is None else weights
        failed = False
        if exp_thresh <= 0:
            msg = f"Expected ratio thresh must be a float > 0: {exp_thresh}"
            logger.error(msg)
            raise ValueError(msg)
        num_splits = 1
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
        prev_in = self.dists['pi_in']
        if len(iterations) == 0:
            raise ValueError(f'No iterations specified: {pca_splits}, {pca_mask}')
        for i, (pca_mask, pca_cs) in enumerate(iterations):
            str_val = pca_mask if pca_mask is not None else "ALL"
            logger.info(f"Iteration {i}: Solving using ({str_val}, {pca_cs})")

            self.set_weights(weights)
            try:
                super().solve(
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
                # * Thrown when pdf() method fails on pi_in from previous iteration
                # TODO: Should we raise this error if on first iteration? Test this error better
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
                    if i == 0:
                        raise RuntimeError('No solution found within exp_thresh')
                    failed = True

            if failed:
                logger.info(f"Resetting to last solution at {iterations[i-1]}")
                self.set_weights(weights[:-1])
                self.dists["pi_in"] = prev_in
                logger.debug(f"dists: {self.dists}")
                self.solve(
                    pca_mask=iterations[i - 1][0],
                    pca_components=[iterations[i - 1][1]],
                )
                break
            else:
                state_vals = {
                    "iteration": len(it_results),
                    "pca_components": str(pca_cs),
                    "pca_mask": str(pca_mask),
                    "pca_splits": num_splits,
                }
                if state_extra is not None:
                    state_vals.update(state_extra)
                self.save_state(state_vals)
                it_results.append(self.result.copy())
                it_results[-1]["i"] = len(it_results) - 1
                it_results[-1]["num_splits"] = num_splits
                if i != len(iterations) - 1:
                    logger.info("Updating weights")
                    weights.append(self.state["ratio"].values)
                    prev_in = self.dists['pi_in']

        self.it_results = pd.concat(it_results)
        self.result = self.it_results.iloc[[-1]]
        self.result = self.result.drop(columns=['i', 'pca_mask'])

    def get_iteration_state(self, iteration=-1):
        """
        Retrieve the state of the system at the specified iteration
        """
        if self.pca_states is None:
            df = self.state
        else:
            iterations = self.pca_states["iteration"].unique()
            df = self.pca_states[self.pca_states["iteration"] == iterations[iteration]]

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
        df = df if df is not None else self.get_iteration_state(iteration=iteration)

        ax, labels = super().plot_D(
            df=df,
            state_idx=state_idx,
            state_col=state_col,
            ratio_col=ratio_col,
            weight_col=weight_col,
            plot_obs=plot_obs,
            plot_pf=plot_pf,
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
        as stored in the self.it_results and self.pca_states attributes of the
        PCAMUDselflem object, which are updated during a PCAMUDselflem.solve_it() call.
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
            iteration=0,
            initial_kwargs={'color': 'black', 'linestyle': ':', 'label': '$\pi^\mathrm{init}$'},
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
            ls = ['--', '-.']
            ls = [linestyle] * (len(iterations) + 1) if linestyle is not None else ls
            linecycler = cycle(ls)

            alphas = np.ones(len(iterations)) if not shade else np.linspace(0.1, 0.9, len(iterations))
            line_opts = {'fill': False}
            for i, it in enumerate(iterations):
                line_opts['alpha'] = alphas[i]
                line_opts['label'] = f"$(\pi^\mathrm{{up}}_{{\lambda_{param_idx}}})_{{{i}}}$"
                line_opts['color'] = colors[i]
                line_opts['linestyle'] = next(linecycler)
                _, l = self.plot_L(
                    param_idx=param_idx,
                    iteration=it,
                    initial_kwargs=None,
                    update_kwargs=line_opts,
                    mud_kwargs=None,
                    lam_true=None,
                    ax=ax,
                )
                labels += l

        # Plot final solution, with mud argument and true lambda
        line_opts = {'fill': True}
        line_opts['label'] = f"$\pi^\mathrm{{up}}_{{\lambda_{param_idx}}}$"
        line_opts['linestyle'] = '-'
        line_opts['color'] = colors[-1]
        mud_args = {'alpha': 1.0, 'linestyle': '--', 'label': '$\lambda^\mathrm{MUD}$'}
        # if 'color' in kwargs.keys():
        #     mud_args['color'] = kwargs['color']
        _, l = self.plot_L(
            param_idx=param_idx,
            iteration=len(self.it_results) - 1,
            initial_kwargs=None,
            update_kwargs=line_opts,
            mud_kwargs=mud_args,
            lam_true=lam_true,
            ax=ax,
        )
        labels += l
        ax.legend(labels=labels, loc='upper right', fontsize=14)

#     def density_plots(
#         self,
#         lam_true=None,
#         lam_kwargs=None,
#         q_lam_kwargs=None,
#         axs=None,
#         figsize=(14, 6),
#     ):
#         """
#         Plot param and observable space onto sampe plot
# 
#         TODO:
#             - Update this method 
#         """
#         rasie NotImplementedError("This method is not implemented yet")
#         # if axs is None:
        #     fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # elif len(axs) != 2:
        #     len(axs) != self.n_params
        # lam_kwargs = {} if lam_kwargs is None else lam_kwargs
        # q_lam_kwargs = {} if q_lam_kwargs is None else q_lam_kwargs
        # lam_kwargs["ax"] = axs[0]
        # lam_kwargs["nc"] = nc
        # q_lam_kwargs["ax"] = axs[1]
        # q_lam_kwargs["nc"] = nc
        # self.plot_L(**lam_kwargs)
        # self.plot_D(**q_lam_kwargs)
        # lam_true = lam_kwargs.get("lam_true", None)
        # fig = axs[0].get_figure()
        # fig.suptitle(
        #     self._parse_title(
        #         result=self.result if nc is None else self.pca_results.loc[[nc]],
        #         lam_true=lam_true,
        #     )
        # )
        # fig.tight_layout()

        # return axs

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
            ax.set_title(f"$\lambda_{i}$")

        fig.suptitle(
            self._parse_title(
                result=self.result,
                nc=True,
                lam_true=lam_true,
            )
        )
        fig.tight_layout()

    def _parse_title(
        self,
        result=None,
        nc=True,
        lam_true=None,
    ):
        """
        Parse title for plots
        """
        result = self.result if result is None else result
        title = super()._parse_title(result=result, lam_true=lam_true)
        num_splits = len(self.pca_states["iteration"].unique())
        if nc:
            title = f"# Splits = {num_splits}: " + title

        return title
