"""
Dynamic Model Class

"""
import pdb
import random
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import alive_bar
from matplotlib.patches import Rectangle
from rich.table import Table
from scipy.stats.distributions import uniform

from pydci.log import log_table, logger
from pydci.utils import add_noise, get_df, get_uniform_box, put_df, set_shape
from pydci import PCAMUDProblem


interval_colors = sns.color_palette("muted", n_colors=50)


class DynamicModel:
    """
    Class defining a model for inverse problems. The model

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
        x0,
        lam_true,
        t0=0.0,
        measurement_noise=0.05,
        solve_ts=0.2,
        sample_ts=1,
        param_mins=None,
        param_maxs=None,
        state_mins=None,
        state_maxs=None,
        param_shifts=None,
        num_states=None,
        max_states=10,
    ):
        self.x0 = x0
        self.t0 = t0
        self.samples_x0 = None
        self.lam_true = np.array(lam_true)
        self.measurement_noise = measurement_noise
        self.solve_ts = solve_ts
        self.sample_ts = sample_ts
        self.param_shifts = {} if param_shifts is None else param_shifts
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.state_mins = state_mins
        self.state_maxs = state_maxs

        # TODO: Hard code max number of states
        # ! Justify and elaborte on this limitation - Warning messages maybe?
        num_states = self.n_states if self.n_states < max_states else max_states
        self.state_idxs = np.random.choice(
            self.n_states, size=num_states, replace=False
        )
        self.state_idxs.sort()

        self.samples = []
        self.data = []
        self.probs = []

    @property
    def n_params(self) -> int:
        return len(self.lam_true)

    @property
    def n_states(self) -> int:
        return len(self.x0)

    @property
    def n_sensors(self) -> int:
        return len(self.state_idxs)

    def get_param_intervals(self, t0, t1):
        """
        Given the algorithm's current iteration, determines the set of time
        steps to solve over, and the value of the true parameter at each
        time-step.
        """
        time_window = t1 - t0
        solve_step = int(time_window / self.solve_ts)
        ts = np.linspace(t0, t1, solve_step)
        shift_times = list(self.param_shifts.keys())
        shift_times.sort()
        shift_idx = np.zeros((len(ts)), dtype=int)
        param_vals = np.zeros((len(ts), self.n_params))
        for p_idx in range(self.n_params):
            param_vals[:, p_idx] = self.lam_true[p_idx]
        for i, st in enumerate(shift_times):
            idxs = ts > st
            shift_idx[idxs] = i
            for p_idx in range(self.n_params):
                param_vals[idxs, p_idx] = self.param_shifts[st][p_idx]

        return ts, shift_idx, param_vals

    def forward_solve(self, tf, x0=None, t0=None, samples=None, samples_x0=None):
        """
        Forward Model Solve

        Solve the forward model from t0 to t1. If a set of samples are passed,
        the samples are pushed forward through the forward model as well. Note
        when evaluating the true solution, this function will divide the time
        range into intervals as dictacted by the `param_shifts` attribute. The
        split time range, with the true param values to use in each interval,
        can be accessed at the `intervals` attribute after this method has
        been run.

        Merged with:

        Compute Observable

        Sample the current state of the system (states over time) and set the
        classes's stp attribute to the observed state of the system we can
        then compute MUD estimates off of using data-constructed maps.


        Parameters here:
          - Time step (and window length)
          - sample_ts
          - solve_ts
        """
        if x0 is not None:
            self.x0 = x0
        x0_temp = self.x0
        if t0 is not None:
            self.t0 = t0

        ts, shift_idx, param_vals = self.get_param_intervals(self.t0, tf)
        true_vals = np.zeros((len(ts), self.n_states))

        for i in range(shift_idx[-1] + 1):
            idxs = shift_idx == i
            times = ts[idxs]
            lam_true = param_vals[idxs, :][0]
            true_vals[idxs] = self.forward_model(x0_temp, times, tuple(lam_true))
            x0_temp = true_vals[idxs][-1]

        sample_step = int(self.sample_ts / self.solve_ts)
        sample_ts_flag = np.mod(np.arange(len(ts)), sample_step) == 0
        sample_ts_flag[-1] = True
        sample_ts_idxs = np.where(sample_ts_flag)[0]
        measurements = np.empty((len(ts), self.n_states))
        measurements[:] = np.nan
        measurements[sample_ts_flag] = np.reshape(
            add_noise(true_vals[sample_ts_flag].ravel(), self.measurement_noise),
            true_vals[sample_ts_flag].shape,
        )

        x0_temp = self.x0
        self.t0 = ts[sample_ts_flag][-1]
        self.x0 = measurements[sample_ts_flag][-1]
        logger.info(f"end_point: {self.t0}, {self.x0}")

        push_forwards = None
        if samples is not None:
            if samples_x0 is None:
                if self.samples_x0 is None:
                    self.samples_x0 = self.get_initial_condition(x0_temp, len(samples))
                samples_x0 = self.samples_x0

            push_forwards = np.zeros(
                (len(samples), np.sum(sample_ts_flag), self.n_sensors)
            )
            with alive_bar(
                len(samples),
                title="Solving model sample set:",
                force_tty=True,
                receipt=True,
                length=20,
            ) as bar:
                for j, s in enumerate(samples):
                    push_forwards[j, :, :] = self.forward_model(
                        samples_x0[j], ts, tuple(s)
                    )[sample_ts_idxs][:, self.state_idxs]
                    bar()

            self.samples_x0[:] = self.get_initial_condition(self.x0, len(samples))

        # Store everything in state DF
        data_df = pd.DataFrame(ts, columns=["ts"])
        data_df["shift_idx"] = shift_idx
        data_df["sample_flag"] = sample_ts_flag
        data_df = put_df(data_df, "lam_true", param_vals)
        data_df = put_df(data_df, "q_lam_true", true_vals, size=self.n_states)
        data_df = put_df(data_df, "q_lam_obs", measurements, size=self.n_states)
        # self.states.append(state_df)
        args = {
            "data": measurements[sample_ts_flag][:, self.state_idxs].reshape(
                np.sum(sample_ts_flag) * self.n_sensors, -1
            ),
            "std_dev": self.measurement_noise,
        }
        if push_forwards is not None:
            q_lam_cols = [
                f"q_lam_{x}" for x in range(np.sum(sample_ts_flag * self.n_sensors))
            ]
            full_samples_df = pd.DataFrame(
                np.hstack([samples, push_forwards.reshape(len(samples), -1)]),
                columns=[f"lam_{x}" for x in range(self.n_params)] + q_lam_cols,
            )
            args["samples"] = full_samples_df
            self.data.append(data_df)
            self.samples.append(full_samples_df)

        return args

    def get_initial_condition(self, x0, num_samples):
        """
        Get Initial condition for a number of samples. Initial condition is
        given by populating x0 with measurement noise for each sample.
        """
        init_conds = np.empty((num_samples, len(x0)))
        init_conds[:] = x0
        init_conds = add_noise(init_conds.ravel(), self.measurement_noise)
        # TODO: impose state minimums: for example - IF can't be negative
        if self.state_mins is not None:
            init_conds[init_conds < self.state_mins] = self.state_mins
        if self.state_maxs is not None:
            init_conds[init_conds > self.state_maxs] = self.state_maxs
        init_conds = np.reshape(init_conds, (num_samples, len(x0)))

        return init_conds

    def get_uniform_initial_samples(self, domain=None, center=None, scale=0.5, num_samples=1000):
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

    def forward_model(
        self,
        x0: List[float],
        times: np.ndarray,
        lam: np.ndarray,
    ) -> np.ndarray:
        """
        Forward Model

        Stubb meant to be overwritten by inherited classes.

        Parameters
        ----------
        x0 : List[float]
            Initial conditions.
        times: np.ndarray[float]
            Time steps to solve the model for. Note that times[0] the model
            is assumed to be at state x0.
        parmaeters: Tuple
            Tuple of parameters to set for model run. These should correspond
            to the model parameters being varied.
        """
        raise NotImplementedError("forward_model() base class skeleton.")

    def plot_state(
        self,
        df=None,
        plot_true=True,
        plot_measurements=True,
        plot_samples=True,
        n_samples=10,
        state_idx=0,
        time_col="ts",
        meas_col=None,
        window_type="line",
        plot_shifts=True,
        markersize=100,
        ax=None,
        figsize=(9, 8),
    ):
        """
        Takes a list of observed data dataframes and plots the state at a certain
        index over time. If pf_dfs passed as well, pf_dfs are plotted as well.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot each column (state) of data on a separate subplot
        max_it = len(self.data) - 1
        for iteration, df in enumerate(self.data):
            n_ts = len(df)
            n_states = len([x for x in df.columns if x.startswith("q_lam_true")])

            # plot_vals = df
            # if iteration != max_it:
            #     label = None
            #     plot_vals = df[df["ts"] <= df["ts"][df["sample_flag"] == True].max()]
            if plot_true:
                sns.lineplot(
                    x="ts",
                    y=f"q_lam_true_{state_idx}",
                    ax=ax,
                    color="blue",
                    data=df,
                    linewidth=2,
                    label="True State",
                )
            # Add Measurement Data to the plot
            if plot_measurements:
                label = None if iteration != (max_it) else "Measurements"
                sns.scatterplot(
                    x="ts",
                    y=f"q_lam_obs_{state_idx}",
                    ax=ax,
                    color="black",
                    data=df,
                    s=markersize,
                    marker="*",
                    label=label,
                    zorder=10,
                )
            # Add Push Forward Data to the plot
            if plot_samples:
                self.plot_sample_states(
                    iteration=iteration, state_idx=state_idx,
                    n_samples=n_samples, ax=ax,
                    label=False if iteration != max_it else True)

            if window_type == "line":
                ax.axvline(
                    df["ts"].min(),
                    linestyle="--",
                    color="cyan",
                    alpha=1,
                    label=None,
                )
            elif window_type == "rectangle":
                xmin = df["ts"].min()
                xmax = df["ts"].max()
                ymin, ymax = ax.get_ylim()
                rect = Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=0, alpha=0.3
                )
                rect.set_facecolor(interval_colors[i])
                ax.add_patch(rect)

            # Add Shifts as vertical lines to the plot
            if plot_shifts:
                max_si = df["shift_idx"].max()
                for si, sd in df[df["shift_idx"] > 0].groupby("shift_idx"):
                    ax.axvline(
                        x=sd["ts"].min(),
                        linewidth=3,
                        color="orange",
                        label=None
                        if not ((si == max_si) and (iteration == len(self.data)))
                        else "Shift",
                    )
        if window_type == "line":
            ax.axvline(
                df["ts"].max(),
                linestyle="--",
                color="green",
                alpha=1,
                label="Time Interval",
            )
        ax.legend(fontsize=12)
        ax.set_title(f"State {state_idx} Temporal Evolution")
        ax.set_xlabel("Time Step")
        ax.set_ylabel(f"State {state_idx}")

        plt.tight_layout()

        return ax
    
    def plot_sample_states(self,
                            iteration=0,
                            state_idx=0,
                            n_samples=10,
                            ax=None,
                            label=False,
                            figsize=(9, 8)):
        """
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        sample_df = self.samples[iteration]
        cols = [x for x in sample_df.columns if x.startswith("q_lam_")]
        times = self.data[iteration]["ts"][self.data[iteration]["sample_flag"]].values
        max_samples = len(sample_df)
        n_samples = n_samples if n_samples < max_samples else max_samples
        rand_idxs = random.choices(range(max_samples), k=n_samples)
        plot_data = sample_df[cols].to_numpy().reshape(
            max_samples, len(times), -1)[rand_idxs, :, state_idx].reshape(-1, len(times))

        label = None if not label else f"Samples ({n_samples} random)"
        for i, d in enumerate(plot_data):
            sns.lineplot(
                x=times,
                y=d,
                legend=False,
                color="purple",
                marker="o",
                alpha=0.2,
                label=None if i != (n_samples - 1) else label,
                ax=ax,
            )

        if "best_flag" in sample_df.columns is not None:
            best_sample = np.where(sample_df["best_flag"] == True)[0]
            plot_data = sample_df[cols].to_numpy().reshape(
                max_samples, len(times), -1)[best_sample, :, state_idx].reshape(-1, len(times))

            sns.lineplot(
                x=times,
                y=plot_data[0],
                legend=False,
                color="green",
                linestyle="--",
                marker="o",
                alpha=0.8,
                label=None if label is None else "Best Sample",
                ax=ax,
            )


    def plot_states(self, base_size=5, **kwargs):

        """
        Plot states over time
        """
        grid_plot = self._closest_factors(self.n_states)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )
        for i, ax in enumerate(ax.flat):
            self.plot_state(state_idx=i, ax=ax, **kwargs)
            ax.set_title(f"State {i}: Temporal Evolution")

    def plot_iterations(self, base_size=5):
        """
        Plot states over time
        """
        grid_plot = self._closest_factors(self.n_params)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )
        for prob in len(self.probs):
            for i, ax in enumerate(ax.flat):
                prob.plot_L(param_idx=i, ax=ax)

    def forward_model(
        self,
        x0,
        times,
        lam,
    ):
        """
        Forward model model base function -> To be overwritten

        TODO:
            - Document
        """
        raise NotImplementedError("forward_model() base class skeleton.")

    def estimate_params(
        self,
        time_windows,
        search_params,
        num_samples=100,
        diff=0.5,
    ):
        """
        Iterative estimate

        TODO:
            Updated using PCAMudProblem.solve() search method.
        """
        if num_samples < self.n_params:
            raise ValueError(f"# of samples must be at least > # of params")
        pi_in, samples = self.get_uniform_initial_samples(
            num_samples=num_samples, scale=diff
        )
        best_flag = np.empty((num_samples, 1), dtype=bool)
        self.check_overwrite(attr="probs", overwrite=True)
        for it, t in enumerate(time_windows):
            logger.info(f"Starting iteration from {self.t0} to {t}")
            forward_res = self.forward_solve(t, samples=samples)
            prob = PCAMUDProblem(
                forward_res["samples"], forward_res["data"],
                self.measurement_noise, pi_in=pi_in
            )
            logger.debug(f'Problem dists upon init: {prob.dists}')
            sp = search_params if isinstance(search_params, dict) else search_params[it]
            try:
                prob.solve_search(**sp)
            except RuntimeError as r:
                if "No solution found" in str(r):
                    logger.error(f"No solution found at iteration {it}. Quitting")
                    self.samples = self.samples[:-1]
                    self.data = self.data[:-1]
                    raise r
            else:
                logger.info(f"Solution {prob.result}")
                best_flag[:] = False
                best_flag[prob.mud_arg] = True
                self.samples[it]["best_flag"] = best_flag
                self.probs.append(prob)
                samples = prob.sample_dist(num_samples=num_samples, dist="pi_up")
                pi_in = prob.dists['pi_up']

    
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