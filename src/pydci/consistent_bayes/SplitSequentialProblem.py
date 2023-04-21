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
from rich.table import Table
from scipy.stats import distributions as dist  # type: ignore
from scipy.stats import rv_continuous  # type: ignore
from scipy.stats import entropy
from scipy.stats import gaussian_kde as gkde  # type: ignore
from scipy.stats.distributions import norm
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from pydci import PCAMUDProblem
from pydci.log import disable_log, enable_log, log_table, logger
from pydci.utils import fit_domain, get_df, put_df, set_shape

sns.color_palette("bright")
sns.set_style("darkgrid")

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"


class SplitSequentialProblem(PCAMUDProblem):
    """
    Class defining a SequentialDensity Problem for parameter estimation on.

    To initialize the class, a forward model model, and parameters need to be
    sepcified. The main entrypoint for solving the estimation problem is the
    `seq_solve()` method, with the `search_params` class attribute controlling
    how the the sequential algorithm behaves.

    Attributes
    ----------
    forward_model : callable
        Function that runs the forward model. Should be callable using
    x0 : ndarray
        Initial state of the system.
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
    def it(self):
        return len(self.states["results"])

    @property
    def num_it(self):
        return len(self.states["data"])

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
        super().init_prob(samples, data, std_dev, pi_in=pi_in, weights=weights)
        self.states = {
            "qoi": np.array_split(self.qoi, 1, axis=1),
            "data": np.array_split(self.data, 1, axis=0),
            "results": [],
            "states": [],
        }

    def _create_binary_string(self, lst, max_int):
        binary_string = ""
        for i in range(max_int):
            if i in lst:
                binary_string += "1"
            else:
                binary_string += "0"
        return binary_string

    def _get_qoi_combinations(
        self,
        max_tries=10,
    ):
        """
        Utility function to determine sets of ts combinations to iterate through
        """
        min_num = self.n_params if self.n_params <= self.n_qoi else self.n_qoi
        if self.qoi_method == "all":
            combs = [list(np.arange(self.n_qoi))]
        elif self.qoi_method == "linear":
            combs = [list(np.arange(0, i)) for i in range(min_num, self.n_qoi)]
        elif self.qoi_method == "random":
            # Divide the max#tries amongs the number of timesteps available
            if self.n_qoi < max_tries:
                num_ts_list = range(min_num, self.n_qoi + 1)
                tries_per = int(max_tries / self.n_qoi)
            else:
                num_ts_list = range(
                    min_num, self.n_qoi + 1, int(self.n_qoi / max_tries)
                )
                tries_per = 1

            combs = []
            qoi_choices = range(0, self.n_qoi)
            for num_ts in num_ts_list:
                psble = list(
                    [list(x) for x in itertools.combinations(qoi_choices, num_ts)]
                )
                tries_per = tries_per if tries_per < len(psble) else len(psble)
                combs += random.sample(psble, tries_per)

        return combs

    def solve(
        self,
        num_splits: int = 1,
        qoi_method: str = "all",
        min_weight_thresh: float = 1e-4,
        exp_thresh: float = 0.5,
        best_method: str = "closest",
        max_nc: int = None,
    ):
        """
        Detect shift and determine next action.
        """
        self.qoi_method = qoi_method
        self.min_weight_thresh = min_weight_thresh

        if self.qoi_method not in ["all", "linear", "random"]:
            msg = f"Unrecognized qoi method: {qoi}. Allowed: {am}"
            logger.error(msg)
            raise ValueError(msg)

        self.states["qoi"] = np.array_split(self.qoi, num_splits, axis=1)
        self.states["data"] = np.array_split(self.data, num_splits, axis=0)
        pi_in = self.dists["pi_in"]
        weights = self.state["weight"]
        it_results = []
        it_states = []
        best_it_results = []
        logger.info(f"Starting sequential algorithm with {num_splits}")
        for it in range(self.num_it):
            logger.info(f"Re-initializing problem with split #{it}")
            # Use parent method here, don't want to erase states dictionary
            super().init_prob(
                (self.lam, self.states["qoi"][it]),
                self.states["data"][it],
                self.std_dev,
                pi_in=pi_in,
                weights=weights,
            )
            qoi_combs = self._get_qoi_combinations()

            results = []
            pca_states = []
            logger.info(f"{qoi_method}: Trying {len(qoi_combs)} qoi combs.")
            for q_idx, qc in enumerate(qoi_combs):
                super().solve(
                    pca_mask=qc,
                    max_nc=max_nc,
                    exp_thresh=exp_thresh,
                    best_method=best_method,
                )
                res_df = self.pca_results
                res_df["qoi_comb"] = q_idx

                # actions = []
                # for nc, res in res_df.groupby("nc"):
                #     actions.append(self._get_action(res))
                # res_df["action"] = actions
                results.append(res_df.set_index("qoi_comb", append=True))
                temp = self.pca_states.copy()
                temp["qoi_comb"] = q_idx
                pca_states.append(temp)

            res_df = pd.concat(results)
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
            # TODO: replace this extra call with just setting appropriately
            # to what's saved in history to not reproduce work.
            idx_max = res_df[self.best_method].idxmax()
            super().solve(
                pca_mask=qoi_combs[idx_max[1]],
                max_nc=max_nc,
                exp_thresh=self.exp_thresh,
                best_method=self.best_method,
            )
            it_results.append(res_df.copy())
            best_it_results.append(res_df.loc[[idx_max]].copy())

            pca_states = pd.concat(pca_states, axis=0)
            pca_states["split"] = it
            it_states.append(
                pca_states[
                    [
                        "split",
                        "qoi_comb",
                        "nc",
                        "weight",
                        "pi_obs",
                        "pi_pr",
                        "ratio",
                        "pi_up",
                    ]
                ]
            )

            if it + 1 < self.num_it:
                # TODO: Implement weight inflation if weights < min thresh?
                logger.info("Update: setting pi_up -> pi_in, ratio -> weights")
                pi_in = self.dists["pi_up"]
                weights = self.state["ratio"]

        best_it_result = pd.concat(
            best_it_results, keys=np.arange(self.num_it), names=["split"]
        )
        self.split_results = pd.concat(
            it_results, keys=np.arange(self.num_it), names=["split"]
        )
        self.split_states = pd.concat(it_states, axis=0)
        self.result = best_it_result.iloc[[num_splits - 1]]

    def get_summary_table(
        self,
    ):
        """
        Get a summary table of results to print
        """
        # TODO: Implement
        fields = ["Iteration", "NC", "E(r)", "D_KL"]

        table = Table(show_header=True, header_style="bold magenta")
        cols = ["Key", "Value"]
        for c in cols:
            table.add_column(c)

        res_df = self.results[-1]
        best_idx = res_df[best].argmax()
        row = (
            str(len(self.mud_res)),
            f"{res_df.loc[best_idx]['action']}",
            f"{res_df.loc[best_idx]['nc']:1.0f}",
            f"{res_df.loc[best_idx]['e_r']:0.3f}",
            f"{res_df.loc[best_idx]['kl']:0.3f}",
        )
        for i in range(len(fields)):
            table.add_row(fields[i], row[i])

        return table

    def plot_L(
        self,
        idx=None,
        lam_true=None,
        mud_point=None,
        df=None,
        param_idx=0,
        param_col="lam",
        ratio_col="ratio",
        weight_col="ratio",
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
            if idx is not None:
                res = self._get_plot_df(idx, cols=["ratio", "weight"])
                df = res[0]
                mud_point = res[1]
                ratio_col = res[2][0]
                weight_col = res[2][1]

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

    def param_density_plots(
        self,
        idx=None,
        lam_true=None,
        base_size=4,
        max_np=9,
    ):
        n_params = self.n_params if self.n_params <= max_np else max_np
        grid_plot = self._closest_factors(n_params)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, ax in enumerate(ax.flat):
            self.plot_L(idx=idx, param_idx=i, lam_true=lam_true, ax=ax)
            ax.set_title(f"$\lambda_{i}$")

        fig.suptitle(
            self._parse_title(
                result=self.result if idx is None else self.pca_results.loc[[idx]],
                nc=True,
                lam_true=lam_true,
            )
        )
        fig.tight_layout()

    def splits_param_density_plots(
        self,
        split_mask=None,
        nc=None,
        qoi_comb=None,
        param_idx=0,
        lam_true=None,
        base_size=4,
        max_splits=9,
        figsize=None,
    ):
        base_size = 4
        if split_mask is None:
            split_mask = np.arange(self.num_it)
        ns = len(split_mask)
        ns = ns if ns <= max_splits else max_splits
        grid_plot = self._closest_factors(ns)
        if figsize is None:
            figsize = (grid_plot[0] * (base_size + 2), grid_plot[0] * base_size)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=figsize,
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, ax in enumerate(ax.flat):
            result = self.split_results.loc[pd.IndexSlice[i, :, :], :]
            best_result = result.iloc[[result[self.best_method].argmax()]]
            best_idx = best_result.index.values[0]
            self.plot_L(idx=best_idx, param_idx=param_idx, lam_true=lam_true, ax=ax)
            ax.set_title(self._parse_title(result=result, lam_true=lam_true, nc=True))

        fig.suptitle(f"Best MUD Estimates by Split For $\lambda_{{param_idx}}$")
        fig.tight_layout()

    def get_full_df(
        self,
        df="state",
        iterations=None,
    ):
        """
        Concatenate stored df
        """

        if df not in self.dfs.keys():
            raise ValueError(f"{df} not one of {self.dfs.keys()}")

        dfs = self.dfs[df]
        if iterations is not None:
            dfs = [dfs[i] for i in range(len(dfs)) if i in iterations]

        return pd.concat(dfs, axis=0)

    def _get_plot_df(self, idx=None, cols=["ratio"]):
        """
        Helper function to get sub df to plot

        We use seaborn's kde plot on the dataframe of lambda samples to plot
        initial and updated distributions, but just weighted appropriately with
        the ratio. If want to plot solution using a different number of
        components as the optimal one stored in the 'ratio' column of the
        samples dataframe, then we have to get it from the pca_states dataframe
        which stores the results from the `solve` routine. In this helper method
        we extract those columns if necessary and append them to the state
        dataframe for plotting.
        """
        state_cols = self.split_states[
            (self.split_states["nc"] == idx[1])
            & (self.split_states["split"] == idx[0])
            & (self.split_states["qoi_comb"] == idx[2])
        ][cols].add_suffix("_plot")
        df = self.state.join(state_cols)
        col_names = [f"{c}_plot" for c in cols]
        mud_point = get_df(self.split_results.loc[[idx]], "lam_MUD", self.n_params)[0]

        return df, mud_point, col_names

    def _parse_title(
        self,
        result=None,
        lam_true=None,
        nc=True,
        qoi_comb=False,
        split=True,
    ):
        """
        Parse title for plots
        """
        result = self.result if result is None else result
        title = super()._parse_title(result=result, lam_true=lam_true, nc=False)
        if nc:
            title = f"{result.index[0][1]} NC, " + title
        if qoi_comb:
            title = f"qoi_comb = {result.index[0][2]}, " + title
        if split:
            title = f"Split {result.index[0][0]}: " + title

        return title


# class DynamicSequentialProblem(SequentialProblem):
#     """
#     Dynamic Seqential MUD Parameter Estimation Problem
#
#     To initialize the class, a forward model model, and parameters need to be
#     sepcified. The main entrypoint for solving the estimation problem is the
#     `seq_solve()` method, with the `search_params` class attribute controlling
#     how the the sequential algorithm behaves.
#
#     Attributes
#     ----------
#     forward_model : callable
#         Function that runs the forward model. Should be callable using
#     x0 : ndarray
#         Initial state of the system.
#     """
#
#     def __init__(
#         self,
#         model,
#     ):
#         self.model = model
#         self.push_forwards = []
#
#     def _detect_shift(
#         self,
#         res,
#     ):
#         """ """
#         shift = False
#         prev = self.get_prev_best()
#         if prev is None:
#             return False
#         if prev["action"] == "RESET":
#             return False
#
#         # Mean condition - Shift in the mean exp_r value detected
#         shift = True
#         if self.e_r_delta is not None:
#             condition = np.abs(prev["e_r"] - res["e_r"].values[0]) <= self.e_r_delta
#             shift = shift if condition else False
#
#         # KL Divergence Condition - If exceeds threshold then shift
#         if self.kl_thresh is not None:
#             condition = res["kl"].values[0] < self.kl_thresh
#             shift = shift if condition else False
#
#         return shift
#
#     def _get_action(
#         self,
#         res,
#     ):
#         """ """
#         action = None
#         if np.abs(1.0 - res["e_r"].values[0]) <= self.exp_thresh:
#             if self.min_weight_thresh is not None:
#                 r_min = self.state[f"ratio"].min()
#                 r_min = r_min[0] if not isinstance(r_min, np.float64) else r_min
#                 if r_min >= self.min_weight_thresh:
#                     action = "RE-WEIGHT"
#             if action != "RE-WEIGHT":
#                 action = "UPDATE"
#         elif self._detect_shift(res):
#             action = "RESET"
#
#         return action
#
#     def solve(
#         self,
#         time_windows,
#         diff=0.5,
#         num_samples=1000,
#         seed=None,
#         max_nc: int = None,
#         splits_per: int = 1,
#         qoi_method: str = "all",
#         e_r_delta: float = 0.5,
#         kl_thresh: float = 3.0,
#         min_weight_thresh: float = 1e-4,
#         exp_thresh: float = 0.5,
#         best_method: str = "closest",
#     ):
#         """
#         Iterative Solver
#
#         Iterative between solving and pushing model forward using sequential
#         MUD algorithm for parameter estimation.
#
#         Parameters
#         ----------
#
#         Returns
#         -------
#
#         Note
#         ----
#         This will reset the state of the class and erase its previous dataframes.
#         """
#         bad = []
#         if self.e_r_delta < 0.5:
#             bad += [f"Shift detection delta(E(r)) must be >= 0.5: {self.e_r}"]
#         if self.kl_thresh < 3.0:
#             bad += [f"Shift detection D_KL_thresh(r) must be >= 3.0: {kl}"]
#         if len(bad) > 0:
#             msg = "Bad args:\n" + "\n".join(bad)
#             logger.error(msg)
#             raise ValueError(msg)
#
#         self.diff = diff
#         if self.samples is not None:
#             yn = input("Previous run exists. Do you want to reset state? y/(n)")
#             if yn == "n":
#                 return
#             self.push_forwards = []
#             self.states = []
#
#         np.random.seed(seed)  # Initial seed for sampling
#         self.samples = self.model.get_uniform_initial_samples(
#             scale=diff, num_samples=num_samples
#         )
#         if len(time_windows) < 2:
#             time_windows.insert(0, 0)
#         time_windows.sort()
#         self.model.t0 = time_windows[0]
#
#         logger.info(f"Starting solve over time : {time_windows}")
#         for it, tf in enumerate(time_windows[1:]):
#             t0 = time_windows[it]
#             logger.info(f"Iteration {it} [{t0}, {tf}]: ")
#             args = self.forward_solve(tf, samples=self.samples)
#             mud_args = self.get_mud_args()
#             self.mud_prob.init_prob(
#                 *[mud_args[x] for x in ["lam", "q_lam", "data", "std_dev"]]
#             )
#             super().solve(
#                 num_splits=num_splits,
#                 qoi_method=qoi_method,
#                 min_weight_thresh=min_weight_thresh,
#                 exp_thresh=exp_thresh,
#                 best_method=best_method,
#                 max_nc=max_nc,
#             )
#             self.iteration_update()
#             logger.info(f" Summary:\n{log_table(self.get_summary_row())}")
#
#     def iteration_update(
#         self,
#     ):
#         """
#         Perform an update after a Sequential MUD estimation
#         """
#         action = self.mud_prob.result["action"].values[0]
#         if action == "UPDATE":
#             logger.info("Drawing from updated distribution")
#             self.samples = self.mud_prob.sample_update(self.n_samples)
#             self.sample_weights = None
#         elif action == "RESET":
#             logger.info("Reseting to initial distribution")
#             self.samples = self.get_uniform_initial_samples(
#                 scale=self.diff, num_samples=self.n_samples
#             )
#         elif action == "RE-WEIGHT":
#             logger.info("Re-weighting current samples")
#             self.sample_weights = (
#                 self.mud_prob.state["weight"] * self.mud_prob.state["ratio"]
#             )
#         else:
#             logger.info("No action taken, continuing with current samples")
#
#     def get_summary_row(
#         self,
#     ):
#         """ """
#         best = self.search_params["best"]
#
#         fields = ["Iteration", "Action", "NC", "E(r)", "D_KL"]
#
#         table = Table(show_header=True, header_style="bold magenta")
#         cols = ["Key", "Value"]
#         for c in cols:
#             table.add_column(c)
#
#         res_df = self.results[-1]
#         best_idx = res_df[best].argmax()
#         row = (
#             str(len(self.mud_res)),
#             f"{res_df.loc[best_idx]['action']}",
#             f"{res_df.loc[best_idx]['nc']:1.0f}",
#             f"{res_df.loc[best_idx]['e_r']:0.3f}",
#             f"{res_df.loc[best_idx]['kl']:0.3f}",
#         )
#         for i in range(len(fields)):
#             table.add_row(fields[i], row[i])
#
#         return table
#
#     def get_full_df(
#         self,
#         df="state",
#         iterations=None,
#     ):
#         """
#         Concatenate stored df
#         """
#
#         if df not in self.dfs.keys():
#             raise ValueError(f"{df} not one of {self.dfs.keys()}")
#
#         dfs = self.dfs[df]
#         if iterations is not None:
#             dfs = [dfs[i] for i in range(len(dfs)) if i in iterations]
#
#         return pd.concat(dfs, axis=0)
