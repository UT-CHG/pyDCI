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
import math
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
from scipy.stats import rv_continuous  # type: ignore
from scipy.stats.distributions import norm
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from pydci import OfflineSequential
from pydci.log import disable_log, enable_log, log_table, logger
from pydci.utils import KDEError, closest_factors, fit_domain, get_df, put_df, set_shape, get_search_combinations

sns.color_palette("bright")
sns.set_style("darkgrid")

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"


class OfflineSequentialSearch:
    """
    Offline Sequential Estimation

    Attributes
    ----------

    Methods
    -------
    solve()

    TODO:
        - Document
    """

    def __init__(
        self,
        samples,
        data,
        std_dev,
        pi_in=None,
        store=True,
    ):
        self.data = data
        self.samples = samples
        self.std_dev = std_dev
        self.pi_in = pi_in
        self.store = store

        self.n_params = len([c for c in self.samples.columns if c.startswith("lam_")])
        self.n_states = len([c for c in self.samples.columns if c.startswith("q_lam_")])
        self.n_samples = len(self.samples)

        if isinstance(self.data, pd.DataFrame):
            # Pandas dataframe with ts and q_lam_obs columns for observations over time
            num_obs = len([c for c in self.data.columns if c.startswith("q_lam_obs")])
            self.measurements = get_df(self.data.dropna(), "q_lam_obs", num_obs).ravel()
        else:
            # Array/matrix of observations - Make 1D
            self.measurements = self.data.ravel()
        if self.n_meas != self.n_states:
            raise ValueError(
                f"Number of measurements {self.n_meas} must match "
                + f"number of states {self.n_states}"
            )

        self.full_search_results = None
        self.search_results = None
        self.result = None
        self.probs = []
        self.best = None

    @property
    def n_meas(self) -> int:
        return len(self.measurements)

    def solve(
        self,
        search_list=None,
        exp_thresh: float = 0.5,
        best_method: str = "closest",
        fail_on_partial: bool = True,
        pi_in=None,
        search_exp_thresh: float = 1e10,
        all_data=True,
        pca_range=None,
        mask_range=None,
        split_range=None,
        max_nc=5,
        data_chunk_size=None,
        max_num_combs=20,
    ):
        """
        Search through different iterations of solvign the PCA problem

        Thea idea of this method is, given a chunk of data, and a list of
        different iterative solve arguments, solve them and determine
        the "best" solution

        Parameters
        ----------
        """
        am = ["closest", "min_kl", "max_kl"]
        if best_method not in am:
            msg = f"Unrecognized best method {best_method}. Allowed: {am}"
            raise ValueError(msg)
        if exp_thresh <= 0:
            msg = f"Expected ratio thresh must be a float > 0: {exp_thresh}"
            raise ValueError(msg)

        # TODO: Move this call to utility function, print pandata DataFrame if logger set
        search_list = (
            get_search_combinations(
                self.n_meas,
                self.n_params,
                self.n_samples,
                exp_thresh=search_exp_thresh,
                all_data=all_data,
                pca_range=pca_range,
                mask_range=mask_range,
                split_range=split_range,
                max_nc=max_nc,
                data_chunk_size=data_chunk_size,
                max_num_combs=max_num_combs,
            ) if search_list is None else search_list
        )
        logger.info(f'Searching through combinations:\n{pd.DataFrame(search_list)}')

        pi_in = self.pi_in if pi_in is None else pi_in
        all_search_results = []
        all_results = []
        probs = []

        def _append_result(p, i):
            all_search_results.append(prob.it_results.copy())
            all_search_results[-1]["search_index"] = idx
            all_results.append(prob.result.copy())
            all_results[-1]["search_index"] = idx

        with alive_bar(
            len(search_list),
            title="Solving for different combinations",
            force_tty=True,
            receipt=False,
            length=40,
        ) as bar:
            for idx, args in enumerate(search_list):
                prob = OfflineSequential(
                    self.samples,
                    self.measurements,
                    self.std_dev,
                    pi_in=pi_in,
                )

                args.update(dict(
                    fail_on_partial=fail_on_partial
                ))
                logger.debug(f"Attempting solve with args: {args}")
                try:
                    prob.solve(**args, state_extra={"search_index": idx})
                except ZeroDivisionError or KDEError or LinAlgError as e:
                    logger.error(f"Failed: Ill-posed problem: {e}")
                except RuntimeError as r:
                    if "No solution found within exp_thresh" in str(r):
                        logger.error(f"Failed: No solution in exp_thresh: {r}")
                        # _append_result(prob, idx)
                    if "Failed to solve problem through all iterations" in str(r):
                        logger.error(f"Failed: No solution found for all data: {r}")
                        _append_result(prob, idx)
                    else:
                        raise r
                else:
                    _append_result(prob, idx)

                probs.append(prob)
                bar()

        if self.store:
            self.probs = probs

        failed = False
        if len(all_results) == 0:
            msg = (
                "All combinations tried failed. If store set to True, "
                + "check probs attribute for individual results, or turn on "
                + "logging using pydci.log.enable_log()"
            )
            failed = True
        else:
            # Parse DataFrame with results of mud estimations for each ts choice
            self.full_search_results = self._process_search_results(
                all_search_results, exp_thresh
            )
            self.search_results = self._process_search_results(all_results, exp_thresh)
            self.result = self.search_results[self.search_results[best_method]]
            logger.debug(f'Search results:\n{self.search_results}')
            self.best = (
                None
                if len(self.result) == 0
                else probs[self.result["search_index"].values[0]]
            )

            if self.best is None:
                msg = f"No solution found within exp_thresh {exp_thresh} for any solve"
                failed = True

        if failed:
            logger.error(msg)
            raise RuntimeError(msg)

    def _process_search_results(
        self,
        dfs,
        exp_thresh,
    ):
        """ """
        res_df = pd.concat(dfs)
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
        return res_df

    def plot_param_updates(
        self,
        probs=None,
        param_idx=0,
        search_idxs=None,
        lam_true=None,
        color=None,
        linestyle=None,
        max_plot=10,
        ax=None,
    ):
        """
        Plot PCA iterations.

        TODO: Document

        Parameters
        ----------
        param_idx : int
            Index of the parameter to plot.
        search_idxs : list of int
            List of search indices to plot updated solution for.
        lam_true : list of float
            True value of the parameter. Plotted as vertical orange line. If None, no true value is plotted.
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

        probs = self.probs if probs is None else probs
        if len(probs) == 0:
            raise ValueError("No probs found. Run solve() first.")

        # Plot initial distribution
        _, labels = probs[0].plot_L(
            param_idx=param_idx,
            iteration=0,
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

        search_idxs = list(range(len(probs))) if search_idxs is None else search_idxs
        if len(search_idxs) > max_plot:
            search_idxs = list(
                range(0, len(search_idxs), int(len(search_idxs) / max_plot))
            )
        if isinstance(color, str):
            colors = [color] * (len(search_idxs) + 1)
        else:
            colors = sns.color_palette(None, len(search_idxs) + 1)

        # Plot iterative updates, for each iteration specified
        if len(search_idxs) > 0:
            ls = ["-", "--", "-."]
            ls = [linestyle] * (len(search_idxs) + 1) if linestyle is not None else ls
            linecycler = cycle(ls)

            line_opts = {"fill": False}
            for i, si in enumerate(search_idxs):
                ls = next(linecycler)
                line_opts[
                    "label"
                ] = f"$(\pi^\mathrm{{up}}_{{\lambda_{param_idx}}})_{{s = {si}}}$"
                line_opts["color"] = colors[i]
                line_opts["linestyle"] = ls
                mud_args = {
                    "color": colors[i],
                    "linestyle": ls,
                    "linewidth": 2,
                    "label": f"$(\lambda^\mathrm{{MUD}})_{{s = {si}}} =  $"
                    + f"{probs[si].mud_point[param_idx]:.2e}",
                }
                _, l = probs[si].plot_L(
                    param_idx=param_idx,
                    initial_kwargs=None,
                    update_kwargs=line_opts,
                    mud_kwargs=mud_args,
                    lam_true=None if i < len(search_idxs) - 1 else lam_true,
                    ax=ax,
                )
                labels += l

        ax.legend(labels=labels, loc="upper right", fontsize=14)

        return ax

    def param_density_plots(
        self,
        lam_true=None,
        base_size=5,
        search_idxs=None,
        figure_size=None,
        max_np=9,
        **kwargs,
    ):
        n_params = self.n_params if self.n_params <= max_np else max_np
        grid_plot = closest_factors(n_params)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=figure_size
            if figure_size is not None
            else (grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, ax in enumerate(ax.flat):
            self.plot_param_updates(
                param_idx=i, search_idxs=search_idxs, lam_true=lam_true, ax=ax, **kwargs
            )
            # Double size of xaxis range by reading existing limits
            xlims = ax.get_xlim()
            ax.set_xlim(
                xlims[0] - (xlims[1] - xlims[0]), xlims[1] + (xlims[1] - xlims[0])
            )
            ax.set_title(f"$\lambda_{i}$")

        fig.suptitle(
            self._parse_title(
                lam_true=lam_true,
            )
        )
        fig.tight_layout()

    def metric_plot(
        self,
        metric="e_r",
        x_vals=None,
        x_label="Iteration",
        e_r_thresh=0.2,
        kl_thresh=4.5,
        ax=None,
        **kwargs,
    ):
        """
        Plot the expected ratio
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if metric not in ["e_r", "kl"]:
            raise ValueError('metric must be one of "e_r" or "kl"')

        # Add one row for first iteration of each option searched
        sr = self.full_search_results.copy()
        sr[x_label] = (sr["i"] + 1) / sr["num_splits"]
        first = sr[sr["i"] == 0].copy()
        first[x_label] = 0
        plot_df = pd.concat([first, sr])

        x_vals = x_label if x_vals is None else x_vals

        args = dict(
            x=x_vals,
            y=metric,
            hue="search_index",
            marker="o",
            ax=ax,
        )
        args.update(kwargs)
        sns.lineplot(plot_df, **args)

        if metric == "e_r":
            ax.hlines(
                [1],
                xmin=0.0,
                xmax=1.0,
                color="black",
                linestyle=":",
                label="$\mathbb{E}(r)$ ≈ 1",
            )
            if e_r_thresh is not None:
                ax.hlines(
                    [1 + e_r_thresh, 1 - e_r_thresh],
                    xmin=0.0,
                    xmax=1.0,
                    color="blue",
                    linestyle=":",
                    label=f"$\pm \epsilon_\mathrm{{pred}} = {e_r_thresh}$",
                )
        if metric == "kl":
            ax.hlines(
                [kl_thresh],
                xmin=0.0,
                xmax=1.0,
                color="red",
                linestyle=":",
                label=f"$\epsilon_\delta = {kl_thresh}$",
            )
        ax.set_xlabel(x_label)
        labels = {
            "kl": "$\mathrm{KL}(\pi^\mathrm{up}_i | \pi^\mathrm{up}_{i-1})$",
            "e_r": "$\mathbb{E}(r_i)$",
        }
        ax.set_ylabel(labels[metric])
        ax.set_xlabel("i / # iterations")

        ax.set_xlim(-0.4, 1.1)
        ax.legend(loc="upper left")

        return ax

    def joint_metric_plot(
        self,
        e_r_thresh=0.2,
        kl_thresh=4.5,
        figsize=(12, 8),
    ):
        """
        TODO: Document
        """
        fig, ax = plt.subplots(2, 1, figsize=figsize)

        self.metric_plot(metric="e_r", e_r_thresh=e_r_thresh, ax=ax[0])

        self.metric_plot(metric="kl", kl_thresh=kl_thresh, ax=ax[1])

    def _parse_title(
        self,
        search_index=None,
        lam_true=None,
    ):
        """
        Parse title for plots
        """
        prob = self.best if search_index is None else self.probs[search_index]
        title = prob._parse_title(lam_true=lam_true)

        # TODO: Add best index to title

        return title
