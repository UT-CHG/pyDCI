"""
Dynamic Model Class

TODO: 
 - Document and add tests

"""
import pdb
import random
from typing import Callable, List, Optional, Tuple, Union

import math 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from matplotlib.patches import Rectangle
from rich.table import Table
from scipy.stats.distributions import uniform
from scipy.stats import multivariate_normal
from numpy.linalg import LinAlgError

from pydci.log import log_table, logger, enable_log, disable_log
from pydci.utils import add_noise, get_df, get_uniform_box, \
    put_df, set_shape, KDEError, set_seed, closest_factors
from pydci import OfflineSequential, OfflineSequentialSearch


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
        def_init='uniform',
    ):
        self.model = model
        self.time_Step = time_step
        self.model_file = model_file
        self.def_init = def_init



        self.probs = []

    @property
    def n_params(self) -> int:
        return len(self.model.lam_true)

    @property
    def n_states(self) -> int:
        return len(self.model.x0)

    @property
    def n_sensors(self) -> int:
        return len(self.model.state_idxs)

    def get_initial_samples(
        self,
        dist=None,
        num_samples=100,
        **kwargs
    ):
        """
        Wrapper method around different methods for generating initial samples.
        Model class by default has a uniform distribution over the parameter
        space surronding the 'true' parameter, or initial parameter guess.

        Parameters
        ----------
        dist : str
            Distribution to use for generating initial samples. Options are
            'uniform' and 'normal'.
        num_samples : int
            Number of samples to generate.
        kwargs : dict
            Additional keyword arguments to pass to the distribution specific method.

        Returns
        -------
        dist, samples: tuple
            Tuple of the distribution object and the samples generated from it.
        """
        dist = self.def_init if dist is None else dist
        if dist == 'uniform':
            return self.get_uniform_initial_samples(num_samples=num_samples, **kwargs)
        elif dist == 'normal':
            return self.get_normal_initial_samples(num_samples=num_samples, **kwargs)
        else:
            raise ValueError(f"Unrecognized distribution: {dist}")

    def get_uniform_initial_samples(
        self,
        domain=None,
        center=None,
        scale=0.5,
        num_samples=1000
    ):
        """
        Generate initial samples from uniform distribution over domain set by
        `self.set_domain`.
        """
        if domain is None:
            center = self.lam_true if center is None else center
            domain = get_uniform_box(
                self.lam_true, factor=scale, mins=self.param_mins, maxs=self.param_maxs
            )
        loc = domain[:, 0]
        scale = domain[:, 1] - domain[:, 0]
        logger.info(
            f"Drawing {num_samples} from uniform at:\n"
            + f"\tloc: {loc}\n\tscale: {scale}"
        )
        dist = uniform(loc=loc, scale=scale)
        samples = dist.rvs(size=(num_samples, self.n_params))
        return dist, samples

    def get_normal_initial_samples(
        self,
        num_samples=100,
        mean=1.0,
        std_dev=1.0
    ):
        """
        Generate initial samples from uniform distribution over domain set by
        `self.set_domain`.
        """
        # Draw from n-dimensional Gaussian centered at `mean` with `std_dev` variance
        # Dimension equal to dimension of `self.n_params`
        logger.info(
            f"Drawing {num_samples} from multivariate normal at:\n"
            + f"\tmean: {mean}\n\tstd_dev: {std_dev}"
        )
        mean = np.ones(self.n_params) * mean if isinstance(mean, (int, float)) else mean
        std_dev = np.ones(self.n_params) * std_dev if isinstance(std_dev, (int, float)) else std_dev
        dist = multivariate_normal(mean=mean, cov=std_dev)
        samples = dist.rvs(size=num_samples)
        return dist, samples

    def check_overwrite(self, attr='probs', overwrite=False):
        """
        """
        # See if probs and data already exist, if so prompt user to continue if we want to delete them to start fresh
        already_yes = False
        attr_val = getattr(self, attr)
        if len(attr_val) > 0:
            if not overwrite and not already_yes:
                logger.warning(
                    "This model already has a set of samples/data/probs. Continuing will delete these and start fresh."
                )
                if not already_yes:
                    if input("Continue? (y/n): ") != "y":
                        return
            setattr(self, attr, [])

    def solve(
        self,
        num_steps=1,
        time_step=None,
        diff=0.5,
        num_samples=100,
        nc=1,
        resample_thresh=0.2,
        shift_thresh=0.9,
        min_eff_sample_size=0.5,
        weights=None,
        seed=None,
    ):
        """
        Online solve

        If problem has not been initialized (no self.probs[] array), then the problem
        is initialized with a uniform distribution over the parameter space around the
        true value, with a scale of `diff` controlling the size of the uniform distribution
        around the true value we search for, and hence the problem difficulty.
        Solve inverse problem for `num_its` consuming `time_step` data at each iteration.
        At each iteration, a set of possible sovle parameters will be searched for, using
        varying number of PCA components, data points, and splits. The best solution will
        be determined by the `best_method` argument.

        """
        logger.debug(f'Running online iterative solve over time window {time_windows}')

        if seed is not None:
            logger.info(f'Setting seed to {seed}')
            set_seed(seed)

        time_step = time_step if time_step is not None else self.time_step
        if self.time_step < self.model.sample_ts:
            raise ValueError(
                f"time_step too small (> sampe_ts): {time_step} > {self.model.sample_ts}"
            )
        if weights is not None and len(weights) != num_samples:
            raise ValueError(f"weights must be None or of length {num_samples}")

        t0 = None
        samples = None
        pi_in = None
        if self.model.n_intervals == 0:
            logger.debug(f'Drawing {num_samples} samples from {self.model.def_init[0]} with args {self.model.def_init[1]}')
            pi_in, samples = self.model.get_initial_samples(num_samples=num_samples)
            t0 = 0.0

        weights = [] if weights is None else weights
        best_flag = np.empty((num_samples, 1), dtype=bool)
        sample_groups = []   # List of lists of data chunks groups used by common set of samples
        sample_group = []    # List for current iteration of data chunks used by common set of samples
        skip_intervals = []  # List of intervals where no solution was found 
        for i, t in enumerate(range(num_steps)):

            sample_group += [i]
            tf = t0 + (i + 1) * time_step
            logger.debug(f"Getting measurements over time window {t0} to {tf}")
            self.model.get_data(t0=t0, tf=tf)
            measurements = get_df(
                self.model.data[-1].dropna(),
                'q_lam_obs', self.model.n_sensors
            )

            num_tries = 0
            solution_found = False
            prev_pi_in = pi_in
            while not solution_found and num_tries < 2:
                # Solve -> Saves states in state dictionary
                self.model.forward_solve(samples=samples)

                prob = OfflineSequential(
                    self.model.samples[-1],
                    measurements,
                    self.measurement_noise,
                    pi_in=pi_in,
                )
                prob.set_weights(weights)

                try:
                    prob.solve(pca_components=list(range(nc)))
                except ZeroDivisionError as z:
                    # Zero division means predictabiltiy assumption violated
                    # -> Param shift may have occured as predicted prob
                    #    of a sample was set to zero where observed data was non-zero
                    e_r_delta = -1.0
                    logger.error(f"Failed: Ill-posed problem: {z}. Suspected param shift.")
                else:
                    e_r = prob.result["e_r"].values[0]
                    e_r_delta = np.abs(e_r - 1.0)
                    logger.info(f"Succesfully solved problem - e_r_delta = {e_r_delta}, kl = {prob.divergence_kl()}")

                # If failed to solve problem because we have refined our weights to much
                # On the current set of samples, then resample from previous iterations updated distribution
                # To start with a fresh set of samples and no weights
                # This occurs when
                #   1. Weights vector is too refined, zero-ing out too many samples so we don't have enough variability
                #       in our samples to solve the problem usinG KDEs -> error through by prob.solve() which we catch by setting e_r_delta = 1.0
                #   2. The e_r_delta we get is above our resampling threshold, but not greater than the shift threshold where we may
                #       think that the true params have shifted and a violation of the predictabiltiy assumption is occuring instead
                #       of jus a resolution issue due to weighting of the curent samples.
                # over-ref
                if (e_r_delta > resample_thresh) and (e_r_delta < shift_thresh):
                    if i == 0:
                        # Won't be able to sample from previous if this is the first iteration
                        raise ValueError("Problem is ill-posed and cannot be solved from the first iteration.")
                    logger.info(f"|E(r) - 1| = {e_r_delta} : < 0 or > {resample_thresh} -> Resampling from previous pi_up and retrying.")
                    samples = self.probs[-1].sample_dist(num_samples, dist='pi_up')
                    pi_in = self.probs[-1].dists['pi_up']
                    logger.info(f"Zeroing out weights and retrying solve.")
                    weights = []
                    num_tries += 1
                elif e_r_delta > shift_thresh or e_r_delta < 0.0:
                    logger.info(f"|E(r) - 1| = {e_r_delta} > {shift_thresh} --> Shift.")
                    logger.info(f"Drawing {num_samples} samples from uniform +- {diff} around true value")
                    pi_in, samples = self.get_uniform_initial_samples(num_samples=num_samples, scale=diff)
                    weights = []
                    num_tries += 1
                else:
                    logger.info(f"|E(r) - 1| = {e_r_delta} < {resample_thresh} - Keeping solution.")
                    logger.info(f"{prob.result}")
                    self.probs.append(prob)

                    best_flag = np.empty((num_samples, 1), dtype=bool)
                    best_flag[:] = False
                    best_flag[prob.mud_arg] = True
                    self.samples[-1]['best_flag'] = best_flag

                    solution_found = True

                    # Determine if new set of weights is too refined -> Calculate effective sample size
                    weights.append(prob.state["ratio"].values)
                    net_weights = np.prod(np.array(weights).T, axis=1)
                    eff_num_samples = len(np.where(net_weights > 1e-10)[0])
                    logger.info(f"Effective sample size: {eff_num_samples}")
                    if eff_num_samples/num_samples < min_eff_sample_size:
                        logger.info(f"Getting new set of samples ({eff_num_samples} < {min_eff_sample_size}).")
                        samples = prob.sample_dist(num_samples, dist='pi_up')
                        pi_in = prob.dists['pi_up']
                        weights = []
                        sample_groups.append(sample_group)
                        sample_group = []
                    else:
                        if num_tries > 0:
                            # Got here after a retry -> Clear sample groups
                            sample_groups.append(sample_group)
                            sample_group = []
                        logger.info(f"Keeping samples.")
                        samples = None

            if not solution_found:
                # TODO: Here could try increasing sample size
                logger.info("No good solution found. Skipping to next time window.")
                pi_in = prev_pi_in
                samples = None
                skip_intervals.append(i)

            logger.info(f'Sample groups {sample_group}')
            t0 = t

        return sample_groups, probs

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
        self,
        probs=None,
        param_idx=0,
        idxs=None,
        figsize=(5, 5),
        lam_true=None,
        ax=None
    ):
        """
        Plot states over time

        TODO: Document
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        probs =self.probs if probs is None else probs

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
            lam_true=None)
        labels += [f'$\pi^{{in}}$']
        if len(idxs) > 2:
            alphas = np.linspace(0.1,0.9,len(idxs))
            for i, j in enumerate(idxs[1:-1]):
                if isinstance(probs[j], OfflineSequential):
                    _, l = probs[j].plot_L(
                        ax=ax,
                        param_idx=param_idx,
                        initial_kwargs=None,
                        update_kwargs={"color": "blue", "alpha": alphas[i],
                                       "linestyle": "--", "fill": False},
                        plot_legend=False,
                        mud_kwargs=None,
                        lam_true=None
                    )
                    labels += [f'$\pi^{{up}}_{{{j}}}$']
        # plot update at final iteration
        _, l = probs[idxs[-1]].plot_L(
            ax=ax,
            param_idx=param_idx,
            initial_kwargs=None,
            update_kwargs={"color": "blue", "linestyle": "-", "fill": True},
            plot_legend=False,
            mud_kwargs={'color': 'blue'},
            lam_true=None,
        )
        labels += [f'$\pi^{{up}}$', '$\lambda^{mud}$']
        for l in lam_true:
            colors = ["orange", "brown", "purple"]
            if len(l) == 2:
                ax.axvline(
                    x=l[1][param_idx],
                    linewidth=3,
                    color=colors[l[0]],
                )
                labels += [f'$\lambda^{{\dagger}}_{{{l[0]}}}$']
            else:
                ax.axvline(
                    x=l[param_idx],
                    linewidth=3,
                    color="orange",
                )
                labels += [f'$\lambda^{{\dagger}}$']
        labels += l

        ax.legend(labels)

        return ax

    def plot_param_densities(self, probs=None, idxs=None, figsize=None, base_size=5, lam_true=None):
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
            figsize=figsize if figsize is None else 
            (grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )
        for i, ax in enumerate(axs.flat):
            self.plot_param_density(probs, param_idx=i, idxs=idxs, ax=ax, lam_true=lam_true)
        
        return axs
    # plot_iterations(probs, idxs=np.arange(0, 10, 2), lam_true=[SEIRS_P2])

    def e_r_plot(self, probs=None, e_r_thresh=None, x_vals=None, x_label='Iteration', ax=None):
        """
        Plot the expected ratio
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        probs = self.probs if probs is None else probs

        e_r = [p.expected_ratio() for p in probs]
        x_vals = np.arange(len(e_r)) if x_vals is None else x_vals

        sns.lineplot(x=x_vals, y=e_r, ax=ax, label='Iterative Expected Ratio', marker="o")
        xlims = ax.get_xlim()
        if e_r_thresh is not None:
            ax.hlines([1 + e_r_thresh, 1 - e_r_thresh], xmin=xlims[0], xmax=xlims[1], color='blue', linestyle=':', label='Threshold $|1 - \mathbb{E}(r)|$')
        ax.hlines([1], xmin=xlims[0], xmax=xlims[1], color='black', linestyle=':', label='Predictability Assumption $\mathbb{E}(r)$ ≈ 1')
        ax.set_xlabel(x_label)
        ax.set_ylabel('$\mathbb{E}(r)$')

    def kl_plot(self, probs=None, kl_thresh=None, x_vals=None, x_label='Iteration', ax=None):
        """
        Plot the expected ratio
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        probs = self.probs if probs is None else probs

        d_kl = [p.divergence_kl() for p in probs]
        x_vals = np.arange(len(d_kl)) if x_vals is None else x_vals

        sns.lineplot(x=x_vals, y=d_kl, color='green', ax=ax, label='$\mathrm{KL}(\pi^{up}_i | \pi^{up}_{i-1})$', marker="o")
        if kl_thresh is not None:
            ax.hlines([kl_thresh], xmin=xlims[0], xmax=xlims[1], color='orange', linestyle=':', label='KL Threshold')

        ax.set_xlabel(x_label)
        ax.set_ylabel('$\mathrm{KL}()$')

        return ax

    def kl_delta_plot(self, probs=None, kl_thresh=None, x_vals=None, x_label='Iteration', ax=None):
        """
        Plot the expected ratio
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        probs = self.probs if probs is None else probs

        d_kl = [p.divergence_kl() for p in probs]
        kl_delta = np.abs(np.array(d_kl[1:]) - np.array(d_kl[:-1]))
        x_vals = np.arange(1, len(kl_delta) + 1) if x_vals is None else x_vals

        label = '$\Delta \mathrm{KL}(\pi^{up}_i | \pi^{up}_{i-1})$'
        sns.lineplot(x=x_vals, y=kl_delta, color='purple', ax=ax, label=label, marker="o")

        if kl_thresh is not None:
            ax.hlines([kl_thresh], xmin=xlims[0], xmax=xlims[1], color='orange', linestyle=':', label='KL Threshold')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('$\Delta \mathrm{KL}()$')

        return ax

    def joint_metrics_plot(self, probs=None, e_r_thresh=None, kl_thresh=None, y1='e_r', y2='kl', x_vals=None, x_label='Iteration', ax=None):
        """
        Plot the expected ratio and KL divergence metrics for a set of problems
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        probs = self.probs if probs is None else probs

        # Check y1 and y1 are iwthin set ['e_r', 'kl', 'kl_delta']
        if y1 not in ['e_r', 'kl', 'kl_delta'] or y2 not in ['e_r', 'kl', 'kl_delta']:
            raise ValueError('y1 and y2 must be in set ["e_r", "kl", "kl_delta"]')
        if y1 == y2:
            raise ValueError('y1 and y2 must be different')

        e_r = [p.expected_ratio() for p in probs]
        d_kl = [p.divergence_kl() for p in probs]

        axs = [ax]
        for i, y in enumerate([y1, y2]):
            if i > 0:
                ax = ax.twinx()
                axs.append(ax)
            if y == 'e_r':
                self.e_r_plot(probs, e_r_thresh=e_r_thresh, x_vals=x_vals, x_label=x_label, ax=ax)
            if y == 'kl':
                self.kl_plot(probs, kl_thresh=kl_thresh, x_vals=x_vals, x_label=x_label, ax=ax)
            if y == 'kl_delta':
                self.kl_delta_plot(probs, kl_thresh=kl_thresh, x_vals=x_vals, x_label=x_label, ax=ax)

        axs[0].legend(loc='upper left')
        axs[1].legend(loc='upper right')

        return axs

    def solve_2(
        self,
        max_t=1,
        num_samples=100,
        time_step=1,
        exp_thresh=0.1,
        kl_thresh=3.0,
        comb_args={
            'max_nc': 3,
            'exp_thresh': 0.5,
            'data_chunk_size': 3
        },
        search_args={
            'exp_thresh': 0.1,
            'best_method': 'max_kl',
        },
        sampling_args={
            'scale': 0.5,
        },
    ):
        """
        Online solve

        If problem has not been initialized (no self.probs[] array), then the problem
        is initialized with a uniform distribution over the parameter space around the
        true value, with a scale of `diff` controlling the size of the uniform distribution
        around the true value we search for, and hence the problem difficulty.
        Solve inverse problem for `num_its` consuming `time_step` data at each iteration.
        At each iteration, a set of possible sovle parameters will be searched for, using
        varying number of PCA components, data points, and splits. The best solution will
        be determined by the `best_method` argument.

        """
        max_its = int(max_t/time_step) + 1
        if max_its < 1:
            raise ValueError(f"max_t must be greater than time_step: {max_t} > {time_step}")

        if len(self.probs) == 0:
            logger.info(f'Initializing {num_samples} samples:\n{sampling_args}')
            pi_in, samples = self.get_initial_samples(
                num_samples=num_samples,
                **sampling_args
            )
            it = 1
        else:
            pi_in = self.probs[-1].dists['pi_up']
            samples = self.probs[-1].sample_dist(num_samples=num_samples)
            it = len(self.probs)
            logger.info(f'Continuing at iteration {it} and timestep {self.t0}')

        best_flag = np.empty((num_samples, 1), dtype=bool)
        while it < max_its:
            logger.debug(f"Iteration {it} from {(it-1)*time_step} to {it*time_step}")
            if it > len(self.model.data):
                logger.debug(f"Getting {int(time_step/self.model.sample_ts)}" +
                             f" data for iteration {it}")
                self.model.get_data(time_step)

            if it > len(self.model.samples):
                self.model.forward_solve(samples, restart=True)

            prob = OfflineSequentialSearch(
                self.model.samples[-1],
                self.model.data[-1],
                self.model.measurement_noise,
                pi_in=pi_in,
                store=True,
            )
            search_combs = prob.get_search_combinations(
                **comb_args,
            )
            search_args.update({'pi_in': pi_in})
            logger.debug(f"Searching: {search_combs}")
            prob.solve(
                search_combs,
                **search_args,
            )
            if prob.best is None:
                shift = False
                reason = ''
                if prob.search_results is not None:
                    avg_kl = np.mean(prob.search_results['kl'])
                    logger.info(f'No solution found within exp_thresh: {res}')
                    if avg_kl > kl_thresh:
                        shift = True
                    reason = f'Avg. KL Divergence > threshold: {avg_kl}'
                else:
                    shift = True
                    reason = 'No solution found amongst search options:\n{search_combs}'

                if shift:
                    logger.info(f'Suspected shift in params at {it}.\n{reason}')
                    pi_in, samples = self.get_initial_samples(
                        num_samples=num_samples,
                        **sampling_args
                    )
                else:
                    logger.info(f'KL Divergence within threshold: {avg_kl}.' +
                                'No shift but bad E(r). Skipping interval.')
                    self.probs.append(prob)
                    it += 1
            else:
                logger.info(f"Best solution found:{prob.best.result}")
                self.probs.append(prob)
                best_flag[:] = False
                best_flag[self.probs[-1].best.mud_arg] = True
                self.model.samples[-1]["best_flag"] = best_flag
                try:
                    samples = self.probs[-1].best.sample_dist(num_samples=num_samples)
                except KDEError as e:
                    msg = f"Unable to draw samples from update distribution: {e}"
                    msg += "\nRe-run using more samples and/or append=True."
                    logger.error(msg)
                    raise RuntimeError(msg)

                pi_in = self.probs[-1].best.dists['pi_up']
                it += 1
