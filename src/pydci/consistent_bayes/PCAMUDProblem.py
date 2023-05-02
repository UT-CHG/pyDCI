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
from numpy.typing import ArrayLike
from numpy.linalg import LinAlgError
from rich.table import Table
from scipy.stats import distributions as dist  # type: ignore
from scipy.stats import rv_continuous  # type: ignore
from scipy.stats import entropy
from scipy.stats import gaussian_kde as gkde  # type: ignore
from scipy.stats.distributions import norm
from alive_progress import alive_bar
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from pydci.consistent_bayes.MUDProblem import MUDProblem
from pydci.log import disable_log, enable_log, log_table, logger
from pydci.utils import fit_domain, get_df, put_df, set_shape

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
            samples, data, std_dev, pi_in=pi_in, pi_pr=None, weights=weights
        )
        self.qoi = self.q_lam
        self.pca_states = None
        self.pca_results = None

    def q_pca(self, mask=None, max_nc=None):
        """
        Build QoI Map Using Data and Measurements

        Aggregate q_lam data with observed data for MUD convergence.
        """
        mask = np.arange(self.n_qoi) if mask is None else mask
        residuals = np.subtract(self.data[mask].T, self.qoi[:, mask]) / self.std_dev
        max_nc = self.n_params if max_nc is None else max_nc
        min_shape = min(residuals.shape)
        max_nc = max_nc if max_nc < min_shape else min_shape

        # Standarize and perform linear PCA
        logger.info(f'Computing PCA using {max_nc} components')
        sc = StandardScaler()
        pca = PCA(n_components=max_nc)
        X_train = pca.fit_transform(sc.fit_transform(residuals))
        self.pca = {
            "X_train": X_train,
            "vecs": pca.components_,
            "var": pca.explained_variance_,
        }
        logger.info(f'PCA Variance: {pca.explained_variance_}')

        # Compute Q_PCA
        self.q_lam = residuals @ pca.components_.T
        self.state = put_df(self.state, "q_pca", self.q_lam, size=max_nc)

    def save_state(self, vals):
        """
        Save current state, adding columns with values in vals dictionary
        """
        keys = vals.keys()
        cols = [c for c in self.state.columns if c.startswith('lam_')]
        cols += [c for c in self.state.columns if c.startswith('q_pca_')]
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
        self.dists["pi_obs"] = dist.norm(loc=len(pca_components) * [0], scale=1)
        self.dists["pi_pr"] = None
        try:
            super().solve()
        except ValueError as v:
            if "array must not contain infs or NaNs" in str(v):
                logger.error(f"Solve with {pca_components} components failed")
            else:
                raise v
        except LinAlgError as le:
            if "data appears to lie in a lower-dimensional" in str(le):
                logger.error(f"GKDE failed using {pca_components} components")
                var = [self.pca['var'][x] for x in pca_components]
                logger.error(f"Variance: {var}")
            else:
                raise le
        self.result['pca_components'] = str(pca_components)
        self.result['pca_mask'] = str(pca_mask)
        self.q_lam = all_qoi

    def solve_it(
        self,
        pca_components=[[0]],
        pca_splits: List[int] = [None],
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
        exp_thresh: float, default=0.5
            Threshold to accept a solution to the MUD problem as a valid
            solution. Any solution more than `exp_thresh` away from 1.0 will
            be deemed as violating the predictability assumption and not valid.
        """
        it_results = []
        weights = []
        for j, pca_mask in enumerate(pca_splits):
            str_val = pca_mask if pca_mask is not None else 'ALL'
            logger.info(f'Using data for pca: {str_val}')
            for i, pca_cs in enumerate(pca_components):
                logger.info(f'Solving using pca components: {pca_cs}')
                self.set_weights(weights)
                self.solve(
                    pca_mask=pca_mask,
                    pca_components=pca_cs,
                )
                e_r = self.result['e_r'].values[0]
                state_vals = {'iteration': len(it_results),
                              'pca_components': str(pca_cs),
                              'pca_mask': str(pca_mask)}
                if state_extra is not None:
                    state_vals.update(state_extra)
                self.save_state(state_vals)
                it_results.append(self.result.copy())
                it_results[-1]['i'] = len(it_results) - 1

                if (diff := np.abs(e_r - 1.0)) > exp_thresh:
                    pdb.set_trace()
                    logger.info(f'|E(r) - 1| = {diff} > {exp_thresh} - Stopping')
                    break

                if i != len(pca_components) - 1:
                    logger.info('Updating weights')
                    weights.append(self.state['ratio'].values)
                    self.dists["pi_in"] = None

            if (diff := np.abs(e_r - 1.0)) > exp_thresh:
                pdb.set_trace()
                logger.info(f'|E(r) - 1| = {diff} > {exp_thresh} - Stopping')
                break

            if j != len(pca_splits) - 1:
                logger.info('Updating weights')
                weights.append(self.state['ratio'].values)
                self.dists["pi_in"] = None

        self.it_results = pd.concat(it_results)
        self.result = self.it_results.iloc[[-1]]

    def solve_search(
        self,
        search_list,
        exp_thresh: float = 0.1,
        best_method: str = "closest",
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
        max_nc: int, default=None
            Specifies the max number of principal components to use when doing
            the PCA transformation on the residuals between the observed and
            simulated data. If not specified, defaults to the min of the number
            of states and the number of parameters.
        exp_thresh: float, default=0.5
            Threshold to accept a solution to the MUD problem as a valid
            solution. Any solution more than `exp_thresh` away from 1.0 will
            be deemed as violating the predictability assumption and not valid.
        best_method: str, default="closest"
            One of "closest", "min_k", or "max_kl", this specifies which
            solution should be deemed the "best". Closest is for the MUD
            estimate corresponding to the E(r) value closest to 1, while min/max
            kl specify the MUD estimate with the corresponding min/max KL
            divergence, indicating the least/most informative update. Note this
            is of estimates that are within `exp_thresh` of E(r). If none are,
            then no solution will be returned.
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
        self.exp_thresh = exp_thresh
        self.best_method = best_method

        all_search_results = []
        all_results = []
        with alive_bar(
            len(search_list),
            title="Solving for different combinations",
            force_tty=True,
            receipt=True,
            length=40,
        ) as bar:
            for idx, (pca_splits, pca_components) in enumerate(search_list):
                logger.info(f'Solving with:\npca_splits: {pca_splits}\n' +
                            f'pc: {pca_components}')
                self.solve_it(pca_components=pca_components,
                              pca_splits=pca_splits,
                              exp_thresh=exp_thresh,
                              state_extra={'index': idx})
                all_search_results.append(self.it_results.copy())
                all_search_results[-1]['index'] = idx
                all_results.append(self.result.copy())
                all_results[-1]['index'] = idx
                bar()

        # Parse DataFrame with results of mud estimations for each ts choice
        res_df = pd.concat(all_results)  # , keys=nc_list, names=['nc'])
        res_df["predict_delta"] = np.abs(res_df["e_r"] - 1.0)
        res_df["within_thresh"] = res_df["predict_delta"] <= self.exp_thresh
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
        self.result = res_df[res_df[self.best_method]]

        # Re-solve Using Best
        self.solve(
            pca_mask=eval(self.result['pca_mask'].values[0]),
            pca_components=eval(self.result['pca_components'].values[0]),
        )

    def solve_seq(
        self,
        chunk_size: None,
        pca_components=[0],
        exp_thresh: float = 0.1,
        stop_thresh: float = 0.99,
        state_extra: dict = None,
    ):
        """
        Sequentially solve problem, suing chunk_sizes. Up to when exp_thresh
        is violated, and then update from there.
        """
        chunk_size = self.n_params if chunk_size is None else chunk_size

        it_results = []
        weights = []
        prev_weights = []
        pca_mask = range(chunk_size)
        e_r = 1.0
        best_chunk = None
        prev_best_chunk = None
        while max(pca_mask) < self.n_qoi:
            logger.info(f'Solving for : {pca_mask}, {pca_components}')
            self.solve(
                pca_mask=pca_mask,
                pca_components=pca_components,
            )
            e_r = self.result['e_r'].values[0]
            if np.abs(e_r - 1.0) <= exp_thresh:
                logger.info(
                    f'E(r) within thresh, adding {chunk_size} more data.')
                best_chunk = pca_mask
                pca_mask = range(min(pca_mask), max(pca_mask) + 1 + chunk_size)
            if np.abs(e_r - 1.0) <= stop_thresh:
                logger.info(f'E(r)={e_r} outside of threshold. Reseting to chunk ' +
                            f'{best_chunk}, updating weights.')
                # Resolve
                self.solve(
                    pca_mask=best_chunk,
                    pca_components=pca_components,
                )
                if prev_best_chunk != best_chunk:
                    # First time we get to this solution, update and go to next chunk
                    state_vals = {'iteration': len(it_results),
                                  'pca_components': str(pca_components),
                                  'pca_mask': str(best_chunk)}
                    if state_extra is not None:
                        state_vals.update(state_extra)
                    self.save_state(state_vals)

                    # Update weights and next iteration
                    weights.append(self.state['ratio'].values)
                    it_results.append(self.result.copy())
                    it_results[-1]['i'] = len(it_results) - 1
                    pca_mask = range(max(best_chunk) + 1,
                                     max(best_chunk) + 1 + chunk_size)
                    prev_best_chunk = best_chunk
                    prev_weights = weights
                    self.set_weights(weights)
                else:
                    # Second time we get here, skip chunk
                    logger.info(f'Skipping chunk {pca_mask}')
                    pca_mask = range(max(pca_mask) + 1,
                                     max(pca_mask) + 1 + chunk_size)
            else:
                logger.info(f'E(r)={e_r} outside of stop threshold. Setting to last' +
                            'best and stopping')
                if prev_best_chunk is not None:
                    self.set_weights(prev_weights)
                    self.solve(
                        pca_mask=prev_best_chunk,
                        pca_components=pca_components,
                    )
                    break

        if len(it_results) < 1:
            logger.error(f'No estimate for chunk size {chunk_size}.' +
                         ' Try reducing the chunk_size, decrasing ' +
                         'number of components, or increasing sample ' +
                         'size')
        else:
            self.it_results = pd.concat(it_results)
            self.result = self.it_results.iloc[[-1]]

    def plot_L(
        self,
        iteration=-1,
        lam_true=None,
        mud_point=None,
        df=None,
        param_idx=0,
        param_col="lam",
        ratio_col="ratio",
        weight_col="weight",
        plot_initial=True,
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
        if df is None:
            df = self.state
        else:
            df = self.pca_states[self.pca_states['iteration'] == iteration]
            mud_point = get_df(self.pca_results.loc[iteration],
                               "lam_MUD", self.n_params)[0]

        ax, labels = super().plot_L(
            lam_true=lam_true,
            mud_point=mud_point,
            df=df,
            param_idx=param_idx,
            param_col=param_col,
            ratio_col=ratio_col,
            weight_col=weight_col,
            plot_initial=plot_initial,
            plot_legend=plot_legend,
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
        figsize=(14, 6),
    ):
        """
        Plot param and observable space onto sampe plot
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        lam_kwargs = {} if lam_kwargs is None else lam_kwargs
        q_lam_kwargs = {} if q_lam_kwargs is None else q_lam_kwargs
        lam_kwargs["ax"] = axs[0]
        lam_kwargs["nc"] = nc
        q_lam_kwargs["ax"] = axs[1]
        q_lam_kwargs["nc"] = nc
        self.plot_L(**lam_kwargs)
        self.plot_D(**q_lam_kwargs)
        lam_true = lam_kwargs.get("lam_true", None)
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
        nc=None,
        lam_true=None,
        base_size=4,
        max_np=9,
    ):
        base_size = 4
        n_params = self.n_params if self.n_params <= max_np else max_np
        grid_plot = self._closest_factors(n_params)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, ax in enumerate(ax.flat):
            self.plot_L(nc=nc, param_idx=i, lam_true=lam_true, ax=ax)
            ax.set_title(f"$\lambda_{i}$")

        fig.suptitle(
            self._parse_title(
                result=self.result if nc is None else self.pca_results.loc[[nc]],
                nc=nc,
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
        grid_plot = self._closest_factors(nc)
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
