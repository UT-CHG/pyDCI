"""
Dynamic Model Class

TODO: 
 - Document and add tests

"""
import math
import pdb
import random
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import alive_bar
from matplotlib.patches import Rectangle
from numpy.linalg import LinAlgError
from rich.table import Table
from scipy.stats import multivariate_normal, gaussian_kde
from scipy.stats.distributions import uniform

from pydci import OfflineSequential, OfflineSequentialSearch
from pydci.log import disable_log, enable_log, log_table, logger
from pydci.utils import (
    KDEError,
    add_noise,
    closest_factors,
    get_df,
    get_uniform_box,
    put_df,
    set_seed,
    set_shape,
)


class OnlineSequential:
    """
    Class defining an Online Sequential Estimation Problem suing DCI methods.

    Attributes
    ----------
    forward_model : callable
        Function that runs the forward model. Should be callable using
    x0 : ndarray
        Initial state of the system.
    lam_true : ndarray
        True parameter value for creating the reference data using the passed
        in forward_model.
    """

    def __init__(
        self,
        model,
        time_step=1,
        model_file=None,
    ):
        self.model = model
        self.time_step = time_step
        self.model_file = model_file

        if self.model_file is not None:
            # Initialize model (which is a class not an instance in this case)
            logger.deubg(f'Loading model from file {model_file}')
            self.model = model(file=self.model_file)

        self.probs = []
        self.it_results = []
        self.result = None

    @property
    def n_params(self) -> int:
        return len(self.model.lam_true)

    @property
    def n_states(self) -> int:
        return len(self.model.x0)

    @property
    def n_sensors(self) -> int:
        return len(self.model.state_idxs)

    def get_num_measurements(self, data_idx=-1) -> int:
        """
        Get number of measurements for a given data chunk index.
        """
        data_idx = data_idx if data_idx != -1 else len(self.model.data) - 1
        if data_idx < 0 or data_idx >= len(self.model.data):
            raise ValueError(f"Invalid data_idx: {data_idx} > {len(self.model.data)}.")
        return self.model.data[data_idx].dropna().shape[0] * self.n_sensors

    def plot_iterations(self, base_size=5):
        """
        Plot states over time

        TODO: Document
        """
        grid_plot = closest_factors(self.n_params)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )
        for prob in len(self.probs):
            for i, ax in enumerate(ax.flat):
                prob.plot_L(param_idx=i, ax=ax)

    def plot_param_density(
        self, probs=None, param_idx=0, idxs=None, figsize=(5, 5), lam_true=None, ax=None
    ):
        """
        Plot states over time

        TODO: Document
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        probs = self.probs if probs is None else probs

        # Plot initial at first iteration
        labels = []
        idxs = np.arange(len(probs)) if idxs is None else idxs
        _ = probs[idxs[0]].plot_L(
            ax=ax,
            param_idx=param_idx,
            initial_kwargs={"color": "black", "linestyle": ":", "fill": True},
            update_kwargs=None,
            plot_legend=False,
            mud_kwargs=None,
            lam_true=None,
        )
        labels += [f"$\pi^{{in}}$"]
        if len(idxs) > 2:
            alphas = np.linspace(0.1, 0.9, len(idxs))
            for i, j in enumerate(idxs[1:-1]):
                if isinstance(probs[j], OfflineSequential):
                    _, l = probs[j].plot_L(
                        ax=ax,
                        param_idx=param_idx,
                        initial_kwargs=None,
                        update_kwargs={
                            "color": "blue",
                            "alpha": alphas[i],
                            "linestyle": "--",
                            "fill": False,
                        },
                        plot_legend=False,
                        mud_kwargs=None,
                        lam_true=None,
                    )
                    labels += [f"$\pi^{{up}}_{{{j}}}$"]
        # plot update at final iteration
        _, l = probs[idxs[-1]].plot_L(
            ax=ax,
            param_idx=param_idx,
            initial_kwargs=None,
            update_kwargs={"color": "blue", "linestyle": "-", "fill": True},
            plot_legend=False,
            mud_kwargs={"color": "blue"},
            lam_true=None,
        )
        labels += [f"$\pi^{{up}}$", "$\lambda^{mud}$"]
        for l in lam_true:
            colors = ["orange", "brown", "purple"]
            if len(l) == 2:
                ax.axvline(
                    x=l[1][param_idx],
                    linewidth=3,
                    color=colors[l[0]],
                )
                labels += [f"$\lambda^{{\dagger}}_{{{l[0]}}}$"]
            else:
                ax.axvline(
                    x=l[param_idx],
                    linewidth=3,
                    color="orange",
                )
                labels += [f"$\lambda^{{\dagger}}$"]
        labels += l

        ax.legend(labels)

        return ax

    def plot_param_densities(
        self, probs=None, idxs=None, figsize=None, base_size=5, lam_true=None
    ):
        """
        TODO: FIx to general case when num_params != 4. Use grid_plot
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        probs = self.probs if probs is None else probs

        # idxs = np.arange(1, len(probs)-1, 1 if )
        grid_plot = self._closest_factors(self.n_params)
        fig, axs = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=figsize
            if figsize is None
            else (grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )
        for i, ax in enumerate(axs.flat):
            self.plot_param_density(
                probs, param_idx=i, idxs=idxs, ax=ax, lam_true=lam_true
            )

        return axs

    # plot_iterations(probs, idxs=np.arange(0, 10, 2), lam_true=[SEIRS_P2])

    def e_r_plot(
        self, results=None, e_r_thresh=None, x_vals='data_idx', x_label="Iteration", ax=None
    ):
        """
        Plot the expected ratio
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if results is None:
            results = pd.concat(self.it_results)

        sns.lineplot(
            results.dropna(), x=x_vals, y='e_r', ax=ax, label="Iterative Expected Ratio", marker="o"
        )
        xlims = ax.get_xlim()
        if e_r_thresh is not None:
            ax.hlines(
                [1 + e_r_thresh, 1 - e_r_thresh],
                xmin=xlims[0],
                xmax=xlims[1],
                color="blue",
                linestyle=":",
                label="Threshold $|1 - \mathbb{E}(r)|$",
            )
        ax.hlines(
            [1],
            xmin=xlims[0],
            xmax=xlims[1],
            color="black",
            linestyle=":",
            label="Predictability Assumption $\mathbb{E}(r)$ ≈ 1",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("$\mathbb{E}(r)$")

    def kl_plot(
        self, results=None, kl_thresh=None, x_vals='data_idx', x_label="Iteration", ax=None
    ):
        """
        Plot the expected ratio
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if results is None:
            results = pd.concat(self.it_results)

        sns.lineplot(
            results.dropna(),
            x=x_vals,
            y='kl',
            color="green",
            ax=ax,
            label="$\mathrm{KL}(\pi^{up}_i | \pi^{up}_{i-1})$",
            marker="o",
        )
        if kl_thresh is not None:
            ax.hlines(
                [kl_thresh],
                xmin=xlims[0],
                xmax=xlims[1],
                color="orange",
                linestyle=":",
                label="KL Threshold",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("$\mathrm{KL}()$")

        return ax

    def kl_delta_plot(
        self, results=None, kl_thresh=None, x_vals='data_idx', x_label="Iteration", ax=None
    ):
        """
        Plot the expected ratio
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if results is None:
            results = pd.concat(self.it_results)

        # Calculate change in KL from previous iteration pandas df
        results['kl_delta'] = results['kl'].shift(1) - results['kl']

        label = "$\Delta \mathrm{KL}(\pi^{up}_i | \pi^{up}_{i-1})$"
        sns.lineplot(
            results.dropna(), x=x_vals, y='kl_delta', color="purple", ax=ax, label=label, marker="o"
        )

        if kl_thresh is not None:
            ax.hlines(
                [kl_thresh],
                xmin=xlims[0],
                xmax=xlims[1],
                color="orange",
                linestyle=":",
                label="KL Threshold",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("$\Delta \mathrm{KL}()$")

        return ax

    def joint_metrics_plot(
        self,
        probs=None,
        e_r_thresh=None,
        kl_thresh=None,
        y1="e_r",
        y2="kl",
        x_vals='data_idx',
        x_label="Iteration",
        ax=None,
    ):
        """
        Plot the expected ratio and KL divergence metrics for a set of problems
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        probs = self.probs if probs is None else probs

        # Check y1 and y1 are iwthin set ['e_r', 'kl', 'kl_delta']
        if y1 not in ["e_r", "kl", "kl_delta"] or y2 not in ["e_r", "kl", "kl_delta"]:
            raise ValueError('y1 and y2 must be in set ["e_r", "kl", "kl_delta"]')
        if y1 == y2:
            raise ValueError("y1 and y2 must be different")

        results = pd.concat(self.it_results)

        axs = [ax]
        for i, y in enumerate([y1, y2]):
            if i > 0:
                ax = ax.twinx()
                axs.append(ax)
            if y == "e_r":
                self.e_r_plot(
                    results=results, e_r_thresh=e_r_thresh, x_vals=x_vals, x_label=x_label, ax=ax
                )
            if y == "kl":
                self.kl_plot(
                    results=results, kl_thresh=kl_thresh, x_vals=x_vals, x_label=x_label, ax=ax
                )
            if y == "kl_delta":
                self.kl_delta_plot(
                    results=results, kl_thresh=kl_thresh, x_vals=x_vals, x_label=x_label, ax=ax
                )

        axs[0].legend(loc="upper left")
        axs[1].legend(loc="upper right")

        return axs

    def make_summary_plots(self, data_idx: int = -1, save_dir=None, save_name=None):
        """
        Makes summary plots for the given iteration of the problem,
        returns True or False if the user wants to continue or not.
        """
        self.probs[data_idx].best.param_density_plots()
        self.probs[data_idx].param_density_plots()
        plt.show()
        response = input('Press enter to continue, or q to quit.')
        if response == 'q':
            logger.info(f"User break at iteration {it}")
            return False

        return True

    def solve_till_thresh(
        self,
        data_idx,
        exp_thresh=0.1,
        start_sample_size=1000,
        max_sample_size=None,
        samples_inc=None,
        sampling_args={},
        min_eff_sample_size=1.0,
        reset=False,
        solver_args=dict(
            search_list=None,
            best_method="closest",
            fail_on_partial=True,
            pi_in=None,
            all_data=True,
            pca_range=[2],
            mask_range=None,
            split_range=[1],
            max_nc=2,
            data_chunk_size=None,
            max_num_combs=10,
        ),
    ):
        """
        Solves a problem with a given set of arguments, increasing the number
        of samples used until the E(r) threshold is met for any solve.
        """
        data_idx = data_idx if data_idx != -1 else len(self.model.data) - 1
        max_sample_size = start_sample_size if max_sample_size is None \
            else max_sample_size
        samples_inc = max(1, int((max_sample_size - start_sample_size)/10)) \
            if samples_inc is None else samples_inc
        logger.debug(f'Max sample size: {max_sample_size}, samples_inc: {samples_inc}')

        def _get_samples(init, ss):
            """
            Helper function to get samples from either a distribution or KDE.
            or to draw new samples from the model if no distribution is given.
            """
            if init is None:
                init, samples = self.model.get_initial_samples(
                    num_samples=ss, **sampling_args
                )
            else:
                if not isinstance(init, gaussian_kde):
                    samples = init.rvs((ss, self.model.n_params))
                else:
                    samples = init.resample(ss).T

            return init, samples

        sample_size = start_sample_size

        # * First see if there is information from a previous solve
        weights = None
        if data_idx > 0 and len(self.it_results) > 0 and not reset:
            # * Use last pi_up from previous as pi_in
            pi_in = self.probs[data_idx-1].best.dists["pi_up"]

            # * Determine whether we keep samples from past iteration
            weights = self.probs[data_idx-1].best.state['ratio'].values
            eff_sample_ratio = len(np.where(weights > 1e-10)[0]) / len(
               self.model.samples[data_idx-1])
            logger.info(f"Effective sample ratio: {eff_sample_ratio}")
            if eff_sample_ratio <= min_eff_sample_size:
                # * Weights from last iteration are too refined
                logger.info(f"Re-setting weights: {eff_sample_ratio} < " +
                            f"{min_eff_sample_size}).")

                logger.info(f"Re-sampling from pi_up_{data_idx-1}")
                samples = self.probs[data_idx-1].best.sample_dist(start_sample_size, dist="pi_up")
                weights = None
            else:
                sample_size = len(self.model.samples[data_idx-1])
                if sample_size > max_sample_size:
                    raise ValueError(
                        f"Number of samples from previous iteration ({sample_size})" +
                        f" exceeds max_sample_size ({max_sample_size})."
                    )

                logger.info(f"Using samples {sample_size} from previous" +
                            f" iteration {data_idx-1}: {eff_sample_ratio} >=" +
                            f"{min_eff_sample_size}).")
                # ! Note: Passing None to forward_solve will use the samples from the previous iteration
                samples = None
        else:
            # * IF reset set, or first iteration, start from initial
            logger.info(f'Starting from initial Reset:{reset}, Data_Idx: ' +
                        f'{data_idx}\nSampling Args: {sampling_args}\n' +
                        f'Solver Args: {solver_args}')
            pi_in, samples = self.model.get_initial_samples(
                num_samples=start_sample_size, **sampling_args
            )

        # * Advance forward model for first set of samples.
        logger.info(f"Advancing forward model for {sample_size} samples")
        self.model.forward_solve(samples=samples, append=False, data_idx=data_idx)

        solved = False
        results = []
        while not solved:
            prob = OfflineSequentialSearch(
                self.model.samples[data_idx],
                self.model.data[data_idx],
                self.model.measurement_noise,
                pi_in=pi_in,
            )
            logger.debug(
                f"Solving {'WEIGHTED' if weights is not None else 'UN-WEIGHTED'}" +
                f" {sample_size} samples. Args: {solver_args}"
            )
            solver_args['weights'] = weights
            solver_args['search_exp_thresh'] = exp_thresh
            try:
                prob.solve(**solver_args)
            except RuntimeError as r:
                if 'No solution found within' in str(r) or 'All combinations tried failed.' in str(r):
                    logger.debug(f"Failed with {sample_size} samples:\n{prob.search_results}")
                else:
                    raise r
            else:
                solved = True

            # TODO: Choose which of these to use
            weighted_flag = False if weights is None or len(weights) == 0 else True
            try:
                results.append(prob.search_results if prob.search_results
                            is not None else pd.concat([p.result for p in prob.probs]))
            except ValueError as v:
                if 'All objects passed were None' in str(v):
                    solved = False
                    results.append(
                        pd.DataFrame(
                            np.repeat(
                                np.array([[sample_size,
                                           weighted_flag]]),
                                len(prob.probs), axis=0
                            ), columns=['num_samples', 'weighted']
                        )
                    )
                else:
                    raise v
            else:
                results[-1]['num_samples'] = sample_size
                results[-1]['weighted'] = weighted_flag

            if not solved:
                if weights is not None and len(weights) > 0:
                    logger.info("Failed using weighted samples from previous " +
                                "iteration -> Re-starting with a fresh set of " +
                                "unweighted samples from the previous distribution")
                    pi_in, samples = _get_samples(
                        self.probs[data_idx-1].best.dists['pi_up'], start_sample_size
                    )
                    weights = None
                elif sample_size >= max_sample_size:
                    logger.debug(f"Reached max sample size of {max_sample_size}")
                    break
                else:
                    logger.debug(f"Drawing {samples_inc} more samples.")
                    _, samples = _get_samples(pi_in, samples_inc)
                    sample_size += samples_inc
                    logger.debug(f"Solving forward model for {samples_inc} more samples")
                    self.model.forward_solve(samples=samples, append=True, data_idx=data_idx)
        
        results = pd.concat(results)
        logger.info(
            f"Solved {'FAILED' if not solved else 'SUCCEEDED'} (s = {sample_size}). Results:\n{results}"
        )

        return solved, results, prob

    def solve(
        self,
        num_its=1,
        max_t=None,
        num_samples=1000,
        max_sample_size=None,
        samples_inc=None,
        time_step=None,
        exp_thresh=0.1,
        kl_thresh=3.0,
        min_eff_sample_size=0.5,
        num_tries_per_it=2,
        sampling_args={
            "scale": 0.5,
        },
        solver_args=dict(
            search_list=None,
            best_method="closest",
            fail_on_partial=True,
            pi_in=None,
            weights=None,
            search_exp_thresh=1e10,
            all_data=True,
            pca_range=[2],
            mask_range=None,
            split_range=[1],
            max_nc=None,
            data_chunk_size=None,
            max_num_combs=10,
        ),
        reset_model=False,
        reset_samples=False,
        make_plots=True,
        seed=None,
    ):
        """
        Online Iterative Solve

        TODO: Document
        """
        if seed is not None:
            logger.info(f"Setting seed to {seed}")
            set_seed(seed)

        t0 = 0
        if len(self.model.data) == 0:
            logger.info("No previous data -> Starting from initial")
            start_idx = 0
            self.probs= []
            self.it_results = []
        elif reset_model:
            logger.info("Resetting model -> Starting from initial")
            start_idx = 0
            self.model.reset(warn=False)
            self.probs = []
            self.it_results = []
        else:
            if reset_samples or len(self.probs) == 0:
                logger.info("Resetting samples -> Starting from beginning using same data.")
                self.model.samples = []
                self.model.samples_xf = []
                start_idx = 0
                self.probs = []
                self.it_results = []
            else:
                start_idx = len(self.probs)
                logger.info(f"Starting from iteration {start_idx}")
                t0 = self.model.data[start_idx-1]['ts'].max()

        time_step = time_step if time_step is not None else self.time_step
        if time_step < self.model.sample_ts:
            raise ValueError(
                f"time_step too small (> sampe_ts): {time_step} > {self.model.sample_ts}"
            )
        num_its = num_its if max_t is None else int(max_t - t0/ time_step) + 1
        if num_its < 1:
            raise ValueError(
                f"Num iterations too small: {num_its} < 1"
            )
        if not (isinstance(make_plots, bool) or isinstance(make_plots, list)):
            raise ValueError(
                f"make_plots must be bool or list of iterations to plot: {make_plots}"
            )
        make_plots = range(num_its) if isinstance(make_plots, bool) and make_plots else make_plots
        logger.debug(f'make_plots: {make_plots}')

        reset = False
        best_flag = pd.DataFrame(np.empty((num_samples, 1), dtype=bool))
        logger.info(f"Starting online solve with {num_samples} samples")
        data_idx = start_idx
        num_tries_per_it = min(2, num_tries_per_it)
        tf = t0 + time_step
        for i in range(num_its):
            data_idx = start_idx + i
            logger.debug(f"Getting measurements over time window {t0} to {tf}")
            try:
                self.model.get_data(tf=tf)
            except Exception as e:
                if 'must be greater than' in str(e):
                    pass
                else:
                    logger.error(f"Error getting data for iteration {data_idx}: {e}")
                    raise e

            num_tries = 0
            solved = False
            logger.debug(f'Starting solves for iteration {data_idx}')
            while not solved and num_tries < num_tries_per_it:
                solved, results, prob = self.solve_till_thresh(
                    data_idx,
                    exp_thresh=exp_thresh,
                    start_sample_size=num_samples,
                    max_sample_size=max_sample_size,
                    samples_inc=samples_inc,
                    sampling_args=sampling_args,
                    min_eff_sample_size=min_eff_sample_size,
                    reset=reset,
                    solver_args=solver_args,
                )
                self.it_results.append(results)
                self.it_results[-1]['data_idx'] = data_idx

                if make_plots and i in make_plots:
                    self.make_summary_plots(data_idx=data_idx)

                if solved:
                    self.probs.append(prob)
                    best_flag = pd.DataFrame(
                        np.empty((len(self.model.samples[data_idx]), 1), dtype=bool),
                        columns=["best_flag"],
                    )
                    best_flag[:] = False
                    best_flag.iloc[prob.best.mud_arg] = True
                    self.model.samples[data_idx]["best_flag"] = best_flag['best_flag']
                    reset = False
                else:
                    if kl := np.mean(results.dropna()['kl']) > kl_thresh:
                        # ! Change point detection -> E(r) bad with large KL divergence
                        logger.info(f"Avg KL value of {kl} > {kl_thresh} -> shifting time window")
                        reset = True
                    num_tries += 1

                    if num_tries >= num_tries_per_it:
                        logger.info(f"Reached max number of tries ({num_tries_per_it}) Skipping Iteration")
                        # TODO: Fail on first iteration? Or continue simulating until a good iteration found?
                        # if i == 0:
                        #     RuntimeError("No solution found for first iteration")
                        self.probs.append(prob)
                        reset = True

            # Advanced Time Steps
            t0 = tf
            tf += time_step
