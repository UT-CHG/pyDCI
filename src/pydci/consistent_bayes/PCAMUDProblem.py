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
        self.init_prob(samples, data, std_dev, pi_in=pi_in)

    @property
    def n_qoi(self):
        return self.qoi.shape[1]

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
            pi_pr=None,
        )
        self.qoi = self.q_lam
        self.pca_states = None
        self.pca_results = None

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
        components_mask: List[int], default=[0]
            Used control what subset of pca components are used in Q_PCA map.
        max_nc: int, default=None
            Specifies the max number of principal components to use when doing
            the PCA transformation on the residuals between the observed and
            simulated data. If not specified, defaults to the min of the number
            of states and the number of parameters.
        """
        self.q_pca(mask=pca_mask)
        all_qoi = self.q_lam
        self.q_lam = self.q_lam[:, pca_components]
        self.dists["pi_obs"] = norm(loc=len(pca_components) * [0], scale=1)
        self.dists["pi_pr"] = None
        try:
            super().solve()
        except ZeroDivisionError as z:
            logger.exception(
                f"({pca_mask}: {pca_components}): "
                + "Predictabiltiy assumption violated"
            )
            raise z
        except KDEError as k:
            logger.exception(
                f"({pca_mask}: {pca_components}): "
                + "Unable to perform kernel density estimates"
            )
            raise k
        else:
            self.result["pca_components"] = str(pca_components)
            self.result["pca_mask"] = str(pca_mask)
            self.q_lam = all_qoi

    def solve_it(
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
            # ! We are doing online iteration, this whipes our previously compute pi_up
            # ! So we suffer from sampling error twice on each iteration....
            # ! Fix for now is to change set_weights() to only whipe dists dictionary
            # ! on first iteration since first iteration passes in [] for weights
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
                self.dists["pi_in"] = prev_in
                logger.debug(f"dists: {self.dists}")
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
                    prev_in = self.dists['pi_in']

        self.it_results = pd.concat(it_results)
        self.result = self.it_results.iloc[[-1]]

    def solve_search(
        self,
        search_list,
        def_args = None,
        exp_thresh: float = 0.5,
        best_method: str = "closest",
    ):
        """
        Search through different iterations of solvign the PCA problem

        Parameters
        ----------
        """
        am = ["closest", "min_kl", "max_kl"]
        if best_method not in am:
            msg = f"Unrecognized best method {best_method}. Allowed: {am}"
            logger.error(msg)
            raise ValueError(msg)
        if exp_thresh <= 0:
            msg = f"Expected ratio thresh must be a float > 0: {exp_thresh}"
            logger.error(msg)
            raise ValueError(msg)

        all_search_results = []
        all_results = []
        with alive_bar(
            len(search_list),
            title="Solving for different combinations",
            force_tty=True,
            receipt=True,
            length=40,
        ) as bar:
            for idx, args in enumerate(search_list):
                args.update(def_args if def_args is not None else {})
                logger.info(f"Solving with args:\n{args}")

                # Solve -> Saves states in state dictionary
                try:
                    self.solve_it(**args, state_extra={"search_index": idx})
                except ZeroDivisionError or KDEError as e:
                    logger.error(f"Failed: Ill-posed problem: {e}")
                except RuntimeError as r:
                    if "No solution found within exp_thresh" in str(r):
                        logger.error(f"Failed: No solution in exp_thresh: {r}")
                else:
                    # ! What state do we need to whipe here to ensure back to original conditions of search on next iteration?
                    # Store results per each iteration and final result
                    # This will be erased the next iteration if we don't store it
                    all_search_results.append(self.it_results.copy())
                    all_search_results[-1]["index"] = idx
                    all_results.append(self.result.copy())
                    all_results[-1]["index"] = idx

                bar()

        # Parse DataFrame with results of mud estimations for each ts choice
        res_df = pd.concat(all_results)
        res_df["predict_delta"] = np.abs(res_df["e_r"] - 1.0)
        res_df["within_thresh"] = res_df["predict_delta"] <= exp_thresh
        res_df["closest"] = np.logical_and(
            res_df["predict_delta"]
            <= res_df[res_df["within_thresh"]]["predict_delta"].min(),
            res_df["within_thresh"],
        )
        res_df["max_kl"] = np.logical_and(
            res_df["kl"] >= res_df[res_df["within_thresh"]]["kl"].max(),
            res_df["within_thresh"],
        )
        res_df["min_kl"] = np.logical_and(
            res_df["kl"] <= res_df[res_df["within_thresh"]]["kl"].min(),
            res_df["within_thresh"],
        )

        # Set to best
        self.search_results = res_df
        self.all_search_results = pd.concat(all_search_results)
        self.result = res_df[res_df[best_method]]

        if len(self.result) == 0:
            raise RuntimeError(f'No solution found within exp_thresh')
        else:
            # Re-solve Using Best
            self.solve_it(**search_list[self.result['index'].values[0]])

    def plot_L(
        self,
        iteration=-1,
        lam_true=None,
        df=None,
        param_idx=0,
        param_col="lam",
        ratio_col="ratio",
        weight_col="weight",
        plot_initial=True,
        plot_mud=False,
        plot_legend=True,
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
        if self.pca_states is None:
            df = self.state
        else:
            iterations = self.pca_states["iteration"].unique()
            df = self.pca_states[self.pca_states["iteration"] == iterations[iteration]]

        ax, labels = super().plot_L(
            lam_true=lam_true,
            df=df,
            param_idx=param_idx,
            param_col=param_col,
            ratio_col=ratio_col,
            weight_col=weight_col,
            plot_initial=plot_initial,
            plot_legend=plot_legend,
            plot_mud=plot_mud,
            ax=ax,
            figsize=figsize,
        )

        return ax, labels

    def plot_D(
        self,
        nc=None,
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
            if nc is not None:
                if nc > len(self.pca_results):
                    msg = f"{nc} greater than max number of components used"
                    logger.error(msg)
                    raise ValueError(msg)
                df = self.state.join(
                    self.pca_states[self.pca_states["nc"] == nc][["ratio"]].add_suffix(
                        f"_nc={nc}"
                    )
                )
                ratio_col = f"ratio_nc={nc}"
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
        lam_kwargs["nc"] = nc
        q_lam_kwargs["ax"] = axs[1]
        q_lam_kwargs["nc"] = nc
        self.plot_L(**lam_kwargs)
        self.plot_D(**q_lam_kwargs)
        lam_true = lam_kwargs.get("lam_true", None)
        fig = axs[0].get_figure()
        fig.suptitle(
            self._parse_title(
                result=self.result if nc is None else self.pca_results.loc[[nc]],
                lam_true=lam_true,
            )
        )
        fig.tight_layout()

        return axs

    def param_density_plots(
        self,
        lam_true=None,
        plot_mud=True,
        base_size=4,
        max_np=9,
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
            self.plot_L(param_idx=i, lam_true=lam_true, ax=ax, plot_mud=plot_mud)
            ax.set_title(f"$\lambda_{i}$")

        fig.suptitle(
            self._parse_title(
                result=self.result,
                nc=True,
                lam_true=lam_true,
            )
        )
        fig.tight_layout()

    def nc_param_density_plots(
        self,
        nc_mask=None,
        param_idx=0,
        lam_true=None,
        base_size=4,
        max_np=9,
    ):
        if nc_mask is None:
            nc_mask = np.arange(1, len(self.pca_results) + 1)
        nc = len(nc_mask)
        nc = nc if nc <= max_np else max_np
        grid_plot = closest_factors(nc)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, ax in enumerate(ax.flat):
            self.plot_L(nc=nc_mask[i], param_idx=param_idx, lam_true=lam_true, ax=ax)
            result = self.pca_results.loc[[nc_mask[i]]]
            ax.set_title(self._parse_title(result=result, lam_true=lam_true, nc=True))

        fig.suptitle("MUD Estimates by Number of PCA Components Used")
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
        nc=True,
        lam_true=None,
    ):
        """
        Parse title for plots
        """
        result = self.result if result is None else result
        title = super()._parse_title(result=result, lam_true=lam_true)
        if nc:
            title = f"nc = {result.index[0]}: " + title

        return title
