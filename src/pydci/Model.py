"""
Dynamic Model Class

TODO: 
 - Document and add tests

"""
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import alive_bar
from matplotlib.patches import Rectangle
from numpy.linalg import LinAlgError
from rich.table import Table
from scipy.stats import multivariate_normal
from scipy.stats.distributions import uniform

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

    MAX_STATES = 1e5

    def __init__(
        self,
        x0=None,
        lam_true=None,
        measurement_noise=0.05,
        solve_ts=0.2,
        sample_ts=1,
        param_mins=None,
        param_maxs=None,
        state_mins=None,
        state_maxs=None,
        param_shifts=None,
        state_idxs=None,
        def_init=["uniform", {"scale": 1.0}],
        file=None,
    ):
        if file is not None:
            self.load(file)
            return

        self.x0 = x0
        self.lam_true = np.array(lam_true)
        self.measurement_noise = measurement_noise
        self.solve_ts = solve_ts
        self.sample_ts = sample_ts
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.state_mins = state_mins
        self.state_maxs = state_maxs
        self.param_shifts = param_shifts
        self.def_init = def_init

        # TODO: Hard code max number of states
        # ! Justify and elaborte on this limitation - Warning messages maybe?
        # TODO: Move this to `observe` method for child classes
        if state_idxs is not None:
            self.state_idxs = state_idxs
        else:
            num_states = (
                self.n_states if self.n_states < self.MAX_STATES else self.MAX_STATES
            )
            self.state_idxs = np.random.choice(
                self.n_states, size=num_states, replace=False
            )
        self.state_idxs.sort()
        self.state_idxs = np.array(self.state_idxs)
        logger.debug(f"State idxs set at {len(self.state_idxs)} of {self.n_states} total indices")

        if 'data' not in dir(self):
            # * Measruements for each data window (i.e. iteration)
            self.data = []

            # * Samples for each data window (i.e. iteration)
            self.samples = []

            # * Full final state for all samples for each iteration for hot-starting
            self.samples_xf = []
        
        self._init_state = self._info_dict()

    @property
    def n_params(self) -> int:
        return len(self.lam_true)

    @property
    def n_states(self) -> int:
        return len(self.x0)

    @property
    def n_sensors(self) -> int:
        return len(self.state_idxs)

    @property
    def n_intervals(self) -> int:
        return len(self.data)

    def reset(self, warn=True):
        """
        Reset model state to original state at initialization
        """
        if warn:
            res = input(
                f"Resetting {self.n_intervals} intervals of data. Continue? (y/N):"
            )
            if res.lower() != "y":
                logger.info("Reset aborted.")
                return
        for k, v in self._init_state.items():
            logger.debug(f"Resetting model attr {k} to {v}")
            setattr(self, k, v)

    def _info_dict(self):
        """
        Get dictionary of model attributes
        """
        allowable_types = [int, float, str, bool, list, pd.DataFrame, np.ndarray]
        info_dict = dict(
            [
                (x, getattr(self, x))
                for x in dir(self)
                if type(getattr(self, x)) in allowable_types
                and not x.startswith("_")
                and not x.startswith("n_")
                and not x.isupper()
            ]
        )
        return info_dict

    def load(self, path=None):
        """
        Load model state from file

        """
        path = f"{self.__class__}.h5" if path is None else path

        # Resolve to absolute path
        path = Path(path).absolute()

        if not path.exists():
            raise ValueError(f"Model file at {path} does not exist.")

        req_attrs = [
            "x0",
            "lam_true",
            "measurement_noise",
            "solve_ts",
            "sample_ts",
            "param_mins",
            "param_maxs",
            "state_mins",
            "state_maxs",
            "param_shifts",
        ]
        attrs = dir(self)
        if any([attr in attrs for attr in req_attrs]):
            msg = "Loading an already initialized model. This will clear all data."
            logger.warning(msg)
            res = input(f"{msg} Continue? (y/N): ")
            if res.lower() != "y":
                logger.info("Load aborted.")
                return
        _ = [setattr(self, attr, None) for attr in req_attrs]

        logger.info(f"Loading model state from state file at {str(path)}")
        with pd.HDFStore(str(path), mode="r") as store:
            for key, val in store.items():
                v_type = type(val)
                logger.debug(
                    f"type({key}) = {v_type} "
                    + f"\n ?= DataframeDF -> {isinstance(store[key], pd.DataFrame)}"
                    + f"\n ? = Series -> {isinstance(store[key], pd.Series)}"
                    + f"\n ? = dict -> {isinstance(store[key], dict)}"
                )
                if isinstance(store[key], pd.DataFrame):
                    val = [
                        d[1].reset_index(level=0, drop=True)
                        for d in store[key].groupby(level=0)
                    ]
                    logger.debug(f"DF: {key}:{[v.head(n=1) for v in val[0:2]]}")
                    setattr(self, key[1:], val)
                if isinstance(store[key], pd.Series) and key.startswith("__dict__"):
                    name = key[len("/__dict__") :]
                    val = store[key].to_dict()
                    logger.debug(f"Dictioanry: {name}:{val}")
                    setattr(self, name, val)

            info_d = store["__attrs__"].to_dict()
            logger.debug(f"Setting info attributes of {len(info_d.keys())}")
            for k, v in info_d.items():
                logger.debug(f"Setting attr {k} to {v}")
                setattr(self, k, v)

    def save(self, path=None, overwrite=False):
        """
        Save model state to file
        """
        path = f"{self.__class__}.h5" if path is None else path

        allowable_types = [int, float, str, bool, list, pd.DataFrame, np.ndarray]
        info_dict = dict(
            [
                (x, getattr(self, x))
                for x in dir(self)
                if type(getattr(self, x)) in allowable_types
                and not x.startswith("_")
                and not x.startswith("n_")
                and not x.isupper()
            ]
        )

        if Path(path).exists() and not overwrite:
            res = input("File exists. Overwrite? (y/N): ")
            if res.lower() != "y":
                logger.info("Save aborted {path} exists. Choose different path name")
                return

        logger.info(f"Saving model to state file at {str(path)}")
        with pd.HDFStore(path, mode="w") as store:
            to_rem = []
            for key, val in info_dict.items():
                if v_type := type(val) == list and len(val) > 0:
                    if all([type(v) == np.ndarray for v in val]):
                        val = [pd.DataFrame(v) for v in val]
                    if all([type(v) == pd.DataFrame for v in val]):
                        val = pd.concat(val, keys=[f"{i}" for i in range(len(val))])
                        v_type = pd.DataFrame
                if v_type == pd.DataFrame:
                    logger.debug(f"Saving {key} as DataFrame")
                    store.put(key, val)
                    to_rem.append(key)
                if v_type == dict:
                    logger.debug(f"Saving {key} as Dictionary")
                    store.put("__dict__{key}", pd.Series(val))

            for key in to_rem:
                _ = info_dict.pop(key)

            store.put("__attrs__", pd.Series(info_dict))

    def get_samples(self, data_idx=-1):
        """
        Get array of lambda samples for a given data index
        """
        try:
            lam = get_df(self.samples[data_idx], "lam", size=self.n_params)
        except IndexError:
            msg = f"Data index {data_idx} is out of range (< {len(self.samples)})"
            logger.error(msg)
            raise ValueError(msg)

        return lam

    def get_param_intervals(self, t0, t1):
        """

        Warning
        -------
        When specifying the `param_shifts` attribute, their must be a key
        0 with the base parameter value.
        TODO: Fix this so its more user friendly
        """
        if self.param_shifts is None:
            self.param_shifts = {t0: self.lam_true}

        # Get list of of param values and times they hold true in order
        shift_times = list(self.param_shifts.keys())
        shift_times.sort()

        if min_time := min(list(self.param_shifts.keys())) > t0:
            logger.warning(
                f"Shift intervals {self.param_shifts} start after t0"
                + f"\nExtending param value at {min_time} to {t0}"
            )
            self.param_shifts[t0] = self.param_shifts[min_time]
            shift_times = max(list(self.param_shifts.keys()))
            shift_times.sort()

        # Determine length of time window
        time_window = t1 - t0
        solve_step = int(time_window / self.solve_ts)
        ts = np.linspace(t0, t1, solve_step)
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

    def search_data(
        self,
        t0,
        tf,
    ):
        """
        Search set of data arrays for data range.
        """
        if len(self.data) == 0:
            return None

        # See if we've solved previous step and data is stored:
        tol = max(self.solve_ts / 100, 1e-6)
        prev_solve = np.where(
            [
                all(
                    [np.abs(tf - x["ts"].max()) < tol, np.abs(t0 - x["ts"].min()) < tol]
                )
                for x in self.data
            ]
        )
        if len(prev_solve[0]) > 0:
            logger.info(f"Found previous solve from {t0} to {tf} at {prev_solve[0][0]}")
            return prev_solve[0][0]

    def get_data(
        self,
        t0=None,
        tf=None,
        x0=None,
    ):
        """
        Get data from system. For synthetic data from models, this method
        solves forward model and adds noise to the results. For real data,
        overwrite thie method to pull the necessary data from the data source.
        """
        last_df = None if len(self.data) == 0 else self.data[-1]

        # * t0 = 0.0 or t0 to start, or last time step if previous solve found
        # * tf = previous time step or # of timesteps to get # data = # params
        if last_df is not None:
            if t0 is not None:
                raise ValueError("Cannot specify t0 with existing data.")

            t0 = last_df["ts"].max()
            tf = tf if tf is not None else t0 + (last_df["ts"].max() - t0)

            if tf <= last_df["ts"].max():
                raise ValueError(f"tf={tf} must be greater than last time-step {t0}.")
        else:
            t0 = 0.0 if t0 is None else t0
            tf = tf if tf is not None else t0 + (self.n_params) * self.sample_ts

        # * x0 = initial condition from last data, or initial for model
        if x0 is None:
            if len(self.data) == 0:
                x0 = self.x0
            else:
                x0 = get_df(last_df.tail(n=1), "q_lam_true", size=self.n_states)[0]

        ts, shift_idx, param_vals = self.get_param_intervals(t0, tf)
        logger.info(f"Getting data for model from {t0} to {tf}")
        true_vals = np.zeros((len(ts), self.n_states))
        for i in range(min(shift_idx), max(shift_idx) + 1):
            idxs = shift_idx == i
            times = ts[idxs]
            lam_true = param_vals[idxs, :][0]
            true_vals[idxs] = self.forward_model(x0, times, tuple(lam_true))
            x0 = true_vals[idxs][-1]

        sample_step = int(self.sample_ts / self.solve_ts)
        sample_ts_flag = np.mod(np.arange(len(ts)), sample_step) == 0
        # TODO: Need this?
        sample_ts_flag[-1] = True
        # sample_ts_flag[0] = True if len(self.data) == 0 else False
        measurements = np.empty((len(ts), self.n_sensors))
        logger.debug(
            f"Shapes: {measurements.shape}, {true_vals.shape}, {sample_ts_flag.shape}"
        )
        measurements[:] = np.nan
        measurements[sample_ts_flag] = np.reshape(
            add_noise(
                true_vals[sample_ts_flag][:, self.state_idxs].ravel(),
                self.measurement_noise,
            ),
            (sum(sample_ts_flag), self.n_sensors),
        )
        # Worked with others but not heat model. Check above works with others
        # measurements[sample_ts_flag] = np.reshape(
        #     add_noise(true_vals[sample_ts_flag, self.state_idxs].ravel(), self.measurement_noise),
        #     (sum(sample_ts_flag), self.n_sensors)
        # )

        # Store everything in state DF
        data_df = pd.DataFrame(ts, columns=["ts"])
        data_df["shift_idx"] = shift_idx
        data_df["sample_flag"] = sample_ts_flag
        data_df = put_df(data_df, "lam_true", param_vals)
        data_df = put_df(data_df, "q_lam_true", true_vals, size=self.n_states)
        data_df = put_df(data_df, "q_lam_obs", measurements, size=self.n_sensors)

        self.data.append(data_df)

    def forward_solve(
        self,
        samples=None,
        append=False,
        data_idx=-1,
    ):
        """
        Forward Model Solve

        Solve the forward model from t0 to t1. If a set of samples are passed,
        the samples are pushed forward through the forward model as well. Note
        when evaluating the true solution, this function will divide the time
        range into intervals as dictacted by the `param_shifts` attribute.

        Parameters here:
          - Time step (and window length)
          - sample_ts
          - solve_ts
        """
        data_df = self.data[
            data_idx := data_idx if data_idx != -1 else len(self.data) - 1
        ]
        x0 = get_df(data_df, "q_lam_true", self.n_states)[0]

        t0 = data_df["ts"].min()
        tf = data_df["ts"].max()
        logger.info(
            f"Beginning forward solve from {t0} to {tf}",
        )

        if samples is None:
            samples = self.get_samples(data_idx=data_idx - 1)
            logger.info(f"No samples passed. Resuming previous {len(samples)} samples")
            if isinstance(self.samples_xf[data_idx - 1], pd.DataFrame):
                self.samples_xf[data_idx - 1] = self.samples_xf[data_idx - 1].to_numpy()
            samples_x0 = self.samples_xf[data_idx - 1]
            append = False
        else:
            logger.debug(f"Starting fresh simulation for {len(samples)}")
            # samples = np.vstack([self.samples[data_idx], samples])
            samples_x0 = self.get_initial_condition(x0, len(samples))

        push_forwards = np.zeros(
            (len(samples), np.sum(data_df["sample_flag"]), self.n_sensors)
        )
        sample_full_state = np.zeros((np.sum(data_df["sample_flag"]), self.n_states))
        samples_xf = np.zeros((len(samples), self.n_states))

        with alive_bar(
            len(samples),
            title="Solving model sample set:",
            force_tty=True,
            receipt=False,
            length=20,
        ) as bar:
            for j, s in enumerate(samples):
                sample_full_state = self.forward_model(
                    samples_x0[j], data_df["ts"].to_numpy(), tuple(s)
                )[data_df["sample_flag"]]
                push_forwards[j, :, :] = sample_full_state[:, self.state_idxs]
                samples_xf[j, :] = sample_full_state[-1, :]
                bar()

        q_lam_cols = [
            f"q_lam_{x}" for x in range(np.sum(data_df["sample_flag"]) * self.n_sensors)
        ]
        full_samples_df = pd.DataFrame(
            np.hstack([samples, push_forwards.reshape(len(samples), -1)]),
            columns=[f"lam_{x}" for x in range(self.n_params)] + q_lam_cols,
        )
        if data_idx >= len(self.samples):
            self.samples.append(full_samples_df)
            self.samples_xf.append(samples_xf)
        else:
            if append:
                self.samples[data_idx] = pd.concat(
                    [self.samples[data_idx], full_samples_df]
                ).reset_index(drop=True)
                self.samples_xf[data_idx] = np.vstack(
                    [self.samples_xf[data_idx], samples_xf]
                )
            else:
                self.samples[data_idx] = full_samples_df
                self.samples_xf[data_idx] = samples_xf

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

    def get_initial_samples(self, num_samples=100, dist=None, **kwargs):
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
        dist = self.def_init[0] if dist is None else dist 
        args = self.def_init[1] if kwargs == {} else kwargs
        if dist == "uniform":
            return self.get_uniform_initial_samples(
                num_samples=num_samples, **args
            )
        elif dist == "normal":
            return self.get_normal_initial_samples(
                num_samples=num_samples, **args
            )
        else:
            raise ValueError(f"Unrecognized distribution: {dist}")

    def get_uniform_initial_samples(
        self, domain=None, center=None, scale=0.5, num_samples=1000
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

    def get_normal_initial_samples(self, num_samples=100, mean=1.0, std_dev=1.0):
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
        std_dev = (
            np.ones(self.n_params) * std_dev
            if isinstance(std_dev, (int, float))
            else std_dev
        )
        dist = multivariate_normal(mean=mean, cov=std_dev)
        samples = dist.rvs(size=num_samples)
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
        plot_true=True,
        plot_measurements=True,
        plot_samples=True,
        n_samples=10,
        state_idx=0,
        iterations=None,
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

        interval_colors = sns.color_palette("muted", n_colors=50)

        # Plot each column (state) of data on a separate subplot
        max_it = len(self.data) - 1
        iterations = np.arange(max_it + 1) if iterations is None else iterations
        for iteration in iterations:
            df = self.data[iteration]
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
                    label="True State" if iteration == max_it else None,
                )
            # Add Measurement Data to the plot
            if plot_measurements and state_idx in self.state_idxs:
                label = None if iteration != (max_it) else "Measurements"
                # Get index of state_idx in list self.state_idxs
                obs_idx = np.where(self.state_idxs == state_idx)[0][0]
                sns.scatterplot(
                    x="ts",
                    y=f"q_lam_obs_{obs_idx}",
                    ax=ax,
                    color="black",
                    data=df,
                    s=markersize,
                    marker="*",
                    label=label,
                    zorder=10,
                )
            # Add Push Forward Data to the plot
            if plot_samples and state_idx in self.state_idxs:
                self.plot_sample_states(
                    iteration=iteration,
                    state_idx=state_idx,
                    n_samples=n_samples,
                    ax=ax,
                    label=False if iteration != max_it else True,
                )

            if window_type == "line":
                ax.axvline(
                    df["ts"].min(),
                    linestyle="--",
                    color="cyan",
                    alpha=0.5,
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
                # Determine shift indices to plot by where df['shift_idx'] increases values, and mark indices where this happens with a flag column
                shift_times = df["ts"][
                    (df["shift_idx"] - df["shift_idx"].shift(1)) == 1
                ]
                if len(shift_times) > 0:
                    for t in shift_times:
                        ax.axvline(
                            x=t,
                            linewidth=3,
                            color="orange",
                            label="Shift"
                            if t >= max(self.param_shifts.keys())
                            else None,
                        )
        if window_type == "line":
            ax.axvline(
                self.data[iterations[-1]]["ts"].max(),
                linestyle="--",
                color="cyan",
                alpha=0.8,
                label="Time Interval",
            )
        ax.legend(fontsize=12)
        ax.set_title(f"State {state_idx} Temporal Evolution")
        ax.set_xlabel("Time Step")
        ax.set_ylabel(f"State {state_idx}")

        plt.tight_layout()

        return ax

    def plot_sample_states(
        self,
        iteration=0,
        state_idx=0,
        n_samples=10,
        ax=None,
        label=False,
        figsize=(9, 8),
    ):
        """ """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        obs_idx = np.where(self.state_idxs == state_idx)[0]
        if len(obs_idx) == 0:
            raise ValueError(f"State {state_idx} not observed. Cannot plot samples.")
        else:
            obs_idx = obs_idx[0]

        sample_df = self.samples[iteration]
        cols = [x for x in sample_df.columns if x.startswith("q_lam_")]
        times = self.data[iteration]["ts"][self.data[iteration]["sample_flag"]].values
        max_samples = len(sample_df)
        n_samples = n_samples if n_samples < max_samples else max_samples
        rand_idxs = random.sample(range(max_samples), n_samples)
        plot_data = (
            sample_df[cols]
            .to_numpy()
            .reshape(max_samples, len(times), -1)[rand_idxs, :, obs_idx]
            .reshape(-1, len(times))
        )

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
            plot_data = (
                sample_df[cols]
                .to_numpy()
                .reshape(max_samples, len(times), -1)[
                    best_sample,
                    :,
                ]
                .reshape(-1, len(times))
            )

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
        grid_plot = closest_factors(self.n_states)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )
        for i, ax in enumerate(ax.flat):
            self.plot_state(state_idx=i, ax=ax, **kwargs)
            ax.set_title(f"State {i}: Temporal Evolution")


# TODO: Below should be moved to appropriate solver problem
#
#     def solve_search(
#         self,
#         search_list,
#         def_args = None,
#         exp_thresh: float = 0.5,
#         best_method: str = "closest",
#         pi_in = None,
#     ):
#         """
#         Search through different iterations of solvign the PCA problem
#
#         Thea idea of this method is, given a chunk of data, and a list of
#         different iterative solve arguments, solve them and determine
#         the "best" solution
#
#         Parameters
#         ----------
#         """
#         am = ["closest", "min_kl", "max_kl"]
#         if best_method not in am:
#             msg = f"Unrecognized best method {best_method}. Allowed: {am}"
#             raise ValueError(msg)
#         if exp_thresh <= 0:
#             msg = f"Expected ratio thresh must be a float > 0: {exp_thresh}"
#             raise ValueError(msg)
#
#         all_search_results = []
#         all_results = []
#         probs = []
#         with alive_bar(
#             len(search_list),
#             title="Solving for different combinations",
#             force_tty=True,
#             receipt=True,
#             length=40,
#         ) as bar:
#             for idx, args in enumerate(search_list):
#                 args.update(def_args if def_args is not None else {})
#
#                 # Get measurements from last data chunk
#                 measurements = get_df(self.data[-1].dropna(), 'q_lam_obs', self.n_sensors)
#
#                 # Solve -> Saves states in state dictionary
#                 prob = PCAMUDProblem(
#                     self.samples[-1],
#                     measurements,
#                     self.measurement_noise,
#                     pi_in=pi_in,
#                 )
#
#                 try:
#                     prob.solve_it(**args, state_extra={"search_index": idx})
#                 except ZeroDivisionError or KDEError  or LinAlgError as e:
#                     logger.error(f"Failed: Ill-posed problem: {e}")
#                     continue
#                 except RuntimeError as r:
#                     if "No solution found within exp_thresh" in str(r):
#                         logger.error(f"Failed: No solution in exp_thresh: {r}")
#                         continue
#                     else:
#                         raise r
#                 else:
#                     all_search_results.append(prob.it_results.copy())
#                     all_search_results[-1]["index"] = idx
#                     all_results.append(prob.result.copy())
#                     all_results[-1]["index"] = idx
#
#                 probs.append(prob)
#                 bar()
#
#         if len(all_results) == 0:
#             return {'best': None,
#                     'probs': probs,
#                     'search_results': None,
#                     'all_search_results': None}
#
#         # Parse DataFrame with results of mud estimations for each ts choice
#         res_df = pd.concat(all_results)
#         res_df["predict_delta"] = np.abs(res_df["e_r"] - 1.0)
#         res_df["within_thresh"] = res_df["predict_delta"] <= exp_thresh
#         res_df["closest"] = np.logical_and(
#             res_df["predict_delta"]
#             <= res_df[res_df["within_thresh"]]["predict_delta"].min(),
#             res_df["within_thresh"],
#         )
#         res_df["max_kl"] = np.logical_and(
#             res_df["kl"] >= res_df[res_df["within_thresh"]]["kl"].max(),
#             res_df["within_thresh"],
#         )
#         res_df["min_kl"] = np.logical_and(
#             res_df["kl"] <= res_df[res_df["within_thresh"]]["kl"].min(),
#             res_df["within_thresh"],
#         )
#
#         # Set to best
#         search_results = res_df
#         all_search_results = pd.concat(all_search_results) # Has internal iterations for each try
#         result = res_df[res_df[best_method]]
#         best = None if len(result) == 0 else probs[result['index'].values[0]]
#
#         # Return best found, results for each tried, and iterative breakdown of each try
#         return {'best': best,
#                 'probs': probs,
#                 'search_results': search_results,
#                 'all_search_results': all_search_results}
#
#     def get_search_combinations(self,
#                                 data_idx=-1,
#                                 exp_thresh=1e10,
#                                 max_nc=5,
#                                 data_chunk_size=None,
#                                 all_data=False,
#                                 ):
#         """
#         Determine search combinations for a given data chunk.
#         By default uses the last data chunk in the data list.
#
#         TODO: Fix and check
#         """
#         if len(self.data) == 0 or data_idx > len(self.data):
#             raise ValueError(f"Invalid data_idx: {data_idx}. Data length: {len(self.data)}")
#         n_data = sum(self.data[data_idx]['sample_flag']) *  self.n_sensors
#         if data_chunk_size is None:
#             data_chunk_size = self.n_params if self.n_params <= n_data else n_data
#
#         def order_of_magnitude(n):
#             return int(math.log10(n)) + 1
#
#         # * 1. # PCA component : Restrict by n_sensors available
#         max_nc = min(order_of_magnitude(len(self.samples[data_idx])), max_nc)
#         pca_range = range(min(max_nc, data_chunk_size))
#         logger.debug(f'PCA search range {pca_range}')
#
#         # * 2. # Data Points to Use : Increasing groups of data_chunk_size.
#         mask_range = [n_data] if all_data else range(data_chunk_size, n_data + 1, data_chunk_size)
#         logger.debug(f'Data chunk end points: {mask_range}')
#
#         # * 3. # Splits : 1 -> (# data/# data_chunk_size). Splits of data_chunk_size.
#         split_range = range(1, int(n_data/data_chunk_size) + 1)
#         logger.debug(f'# of splits: {split_range}')
#
#         search_list = [
#             {
#                 'exp_thresh': exp_thresh,
#                 'pca_components': [list(range(i + 1))],
#                 'pca_mask': range(j),
#                 'pca_splits': k,
#             }
#             for i in pca_range
#             for j in mask_range
#             for k in split_range
#             if j/(k*data_chunk_size) >= 1.0
#         ]
#
#         return search_list
#
#     def check_overwrite(self, attr='probs', overwrite=False):
#         """
#         """
#         # See if probs and data already exist, if so prompt user to continue if we want to delete them to start fresh
#         already_yes = False
#         attr_val = getattr(self, attr)
#         if len(attr_val) > 0:
#             if not overwrite and not already_yes:
#                 logger.warning(
#                     "This model already has a set of samples/data/probs. Continuing will delete these and start fresh."
#                 )
#                 if not already_yes:
#                     if input("Continue? (y/n): ") != "y":
#                         return
#             setattr(self, attr, [])
#
#     def online_iterative(
#         self,
#         num_its=1,
#         num_samples=100,
#         time_step=1,
#         comb_args={
#             'max_nc': 3,
#             'exp_thresh': 0.5,
#             'data_chunk_size': 3
#         },
#         search_args={
#             'exp_thresh': 0.1,
#             'best_method': 'max_kl',
#         },
#         diff=0.5,
#         kl_thresh=3.0,
#     ):
#         """
#         Online solve
#
#         If problem has not been initialized (no self.probs[] array), then the problem
#         is initialized with a uniform distribution over the parameter space around the
#         true value, with a scale of `diff` controlling the size of the uniform distribution
#         around the true value we search for, and hence the problem difficulty.
#         Solve inverse problem for `num_its` consuming `time_step` data at each iteration.
#         At each iteration, a set of possible sovle parameters will be searched for, using
#         varying number of PCA components, data points, and splits. The best solution will
#         be determined by the `best_method` argument.
#
#         """
#         if len(self.probs) == 0:
#             logger.info(f'Initializing problem with difficulty {diff} and {num_samples}')
#             pi_in, samples = self.get_uniform_initial_samples(
#                 num_samples=num_samples,
#                 scale=diff)
#             it = 1
#         else:
#             pi_in = self.probs[-1].dists['pi_up']
#             samples = self.probs[-1].sample_dist(num_samples=num_samples)
#             it = len(self.probs)
#             logger.info(f'Continuing at iteration {it} and timestep {self.t0}')
#
#         max_its = it + num_its
#         best_flag = np.empty((num_samples, 1), dtype=bool)
#         while it < max_its:
#             logger.debug(f"Iteration {it} from {(it-1)*time_step} to {it*time_step}")
#             if it > len(self.data):
#                 logger.debug(f"Getting {int(time_step/self.sample_ts)}. data for iteration {it}")
#                 self.get_data(time_step)
#
#             self.forward_solve(samples, restart=True)
#             search_combs = self.get_search_combinations(
#                 **comb_args,
#             )
#             search_args.update({'pi_in': pi_in})
#             logger.debug(f"Searching: {search_combs}")
#             res = self.solve_search(
#                 search_combs,
#                 **search_args,
#             )
#             if res['best'] is None:
#                 shift = False
#                 reason = ''
#                 if res['search_results'] is not None:
#                     avg_kl = np.mean(res['search_results']['kl'])
#                     logger.info(f'No solution found within exp_thresh: {res}')
#                     if avg_kl > kl_thresh:
#                         shift = True
#                     reason = f'Avg. KL Divergence > threshold: {avg_kl}'
#                 else:
#                     shift = True
#                     reason = 'No solution found amongst search options:\n{search_combs}'
#
#                 if shift:
#                     logger.info(f'Suspected shift in params at {it}.\n{reason}')
#                     pi_in, samples = self.get_uniform_initial_samples(
#                         num_samples=num_samples, scale=diff)
#                 else:
#                     logger.info(f'KL Divergence within threshold: {avg_kl}.' +
#                                 'No shift but bad E(r). Skipping interval.')
#                     self.probs.append(res)
#                     it += 1
#             else:
#                 logger.info(f"Best solution found:{res['best'].result}")
#                 self.probs.append(res['best'])
#                 best_flag[:] = False
#                 best_flag[res['best'].mud_arg] = True
#                 self.samples[-1]["best_flag"] = best_flag
#                 samples = self.probs[-1].sample_dist(num_samples=num_samples)
#                 pi_in = self.probs[-1].dists['pi_up']
#                 it += 1
#
#     def adaptive_online_iterative(
#         self,
#         time_windows,
#         diff=0.5,
#         num_samples=100,
#         nc=1,
#         resample_thresh=0.2,
#         shift_thresh=0.9,
#         min_eff_sample_size=0.5,
#         weights=None,
#         seed=None,
#     ):
#         """
#         Online solve
#
#         If problem has not been initialized (no self.probs[] array), then the problem
#         is initialized with a uniform distribution over the parameter space around the
#         true value, with a scale of `diff` controlling the size of the uniform distribution
#         around the true value we search for, and hence the problem difficulty.
#         Solve inverse problem for `num_its` consuming `time_step` data at each iteration.
#         At each iteration, a set of possible sovle parameters will be searched for, using
#         varying number of PCA components, data points, and splits. The best solution will
#         be determined by the `best_method` argument.
#
#         """
#         logger.debug(f'Running online iterative solve over time window {time_windows}')
#
#         if seed is not None:
#             logger.info(f'Setting seed to {seed}')
#             set_seed(seed)
#
#         if len(time_windows) < 2:
#             raise ValueError("time_windows must be a list of at least length 2")
#         time_windows.sort()
#         if weights is not None and len(weights) != num_samples:
#             raise ValueError(f"weights must be None or of length {num_samples}")
#
#         logger.debug(f'Drawing {num_samples} samples from uniform +- {diff} around true value')
#         pi_in, samples = self.get_uniform_initial_samples(
#             num_samples=num_samples,
#             scale=diff)
#
#         weights = [] if weights is None else weights
#         best_flag = np.empty((num_samples, 1), dtype=bool)
#         t0 = time_windows[0]
#         probs = []
#         restart = False
#         sample_groups = []                # List of lists of data chunks groups used by common set of samples
#         sample_group = []                 # List for current iteration of data chunks used by common set of samples
#         skip_intervals = []               # List of intervals where no solution was found
#         for i, t in enumerate(time_windows[1:]):
#             sample_group += [i]
#             logger.debug(f"Getting measurements over time window {t0} to {t}")
#             self.get_data(t - t0, t0=t0)
#             measurements = get_df(self.data[-1].dropna(), 'q_lam_obs', self.n_sensors)
#
#             num_tries = 0
#             solution_found = False
#             prev_samples = samples
#             prev_pi_in = pi_in
#             while not solution_found and num_tries < 2:
#                 # Solve -> Saves states in state dictionary
#                 self.forward_solve(samples, restart=restart)
#
#                 prob = PCAMUDProblem(
#                     self.samples[-1],
#                     measurements,
#                     self.measurement_noise,
#                     pi_in=pi_in,
#                 )
#                 prob.set_weights(weights)
#
#                 try:
#                     prob.solve(pca_components=list(range(nc)))
#                 except ZeroDivisionError as z:
#                     # Zero division means predictabiltiy assumption violated
#                     # -> Param shift may have occured as predicted prob
#                     #    of a sample was set to zero where observed data was non-zero
#                     e_r_delta = -1.0
#                     logger.error(f"Failed: Ill-posed problem: {z}. Suspected param shift.")
#                 else:
#                     e_r = prob.result["e_r"].values[0]
#                     e_r_delta = np.abs(e_r - 1.0)
#                     logger.info(f"Succesfully solved problem - e_r_delta = {e_r_delta}, kl = {prob.divergence_kl()}")
#
#                 # If failed to solve problem because we have refined our weights to much
#                 # On the current set of samples, then resample from previous iterations updated distribution
#                 # To start with a fresh set of samples and no weights
#                 # This occurs when
#                 #   1. Weights vector is too refined, zero-ing out too many samples so we don't have enough variability
#                 #       in our samples to solve the problem usinG KDEs -> error through by prob.solve() which we catch by setting e_r_delta = 1.0
#                 #   2. The e_r_delta we get is above our resampling threshold, but not greater than the shift threshold where we may
#                 #       think that the true params have shifted and a violation of the predictabiltiy assumption is occuring instead
#                 #       of jus a resolution issue due to weighting of the curent samples.
#                 # over-ref
#                 if (e_r_delta > resample_thresh)  and (e_r_delta < shift_thresh):
#                     if i == 0:
#                         # Won't be able to sample from previous if this is the first iteration
#                         raise ValueError("Problem is ill-posed and cannot be solved from the first iteration.")
#                     logger.info(f"|E(r) - 1| = {e_r_delta} : < 0 or > {resample_thresh} -> Resampling from previous pi_up and retrying.")
#                     samples = probs[-1].sample_dist(num_samples, dist='pi_up')
#                     pi_in = probs[-1].dists['pi_up']
#                     logger.info(f"Zeroing out weights and retrying solve.")
#                     weights = []
#                     num_tries += 1
#                     restart = True
#                 elif e_r_delta > shift_thresh or e_r_delta < 0.0:
#                     logger.info(f"|E(r) - 1| = {e_r_delta} > {shift_thresh} --> Shift.")
#                     logger.info(f"Drawing {num_samples} samples from uniform +- {diff} around true value")
#                     pi_in, samples = self.get_uniform_initial_samples(num_samples=num_samples, scale=diff)
#                     weights = []
#                     num_tries += 1
#                     restart = True
#                 else:
#                     logger.info(f"|E(r) - 1| = {e_r_delta} < {resample_thresh} - Keeping solution.")
#                     logger.info(f"{prob.result}")
#                     probs.append(prob)
#
#                     best_flag = np.empty((num_samples, 1), dtype=bool)
#                     best_flag[:] = False
#                     best_flag[prob.mud_arg] = True
#                     self.samples[-1]['best_flag'] = best_flag
#
#                     solution_found = True
#                     # Determine if new set of weights is too refined -> Calculate effective sample size
#                     weights.append(prob.state["ratio"].values)
#                     net_weights = np.prod(np.array(weights).T, axis=1)
#                     eff_num_samples = len(np.where(net_weights > 1e-10)[0])
#                     logger.info(f"Effective sample size: {eff_num_samples}")
#                     if eff_num_samples/num_samples < min_eff_sample_size:
#                         logger.info(f"Getting new set of samples ({eff_num_samples} < {min_eff_sample_size}).")
#                         samples = prob.sample_dist(num_samples, dist='pi_up')
#                         pi_in = prob.dists['pi_up']
#                         weights = []
#                         restart = True                              # Whether to restart the forward solve simulations or continue from previous final state.
#                         sample_groups.append(sample_group)
#                         sample_group = []
#                     else:
#                         if num_tries > 0:
#                             # Got here after a retry -> Clear sample groups
#                             sample_groups.append(sample_group)
#                             sample_group = []
#                         logger.info(f"Keeping samples.")
#                         restart = False
#
#             if not solution_found:
#                 logger.info(f"No good solution found. Skipping to next time window.")
#                 pi_in = prev_pi_in
#                 samples = prev_samples
#                 restart = False
#                 skip_intervals.append(i)
#
#             logger.info(f'Sample groups {sample_group}')
#             t0 = t
#
#         return sample_groups, probs
#
#
#     def plot_iterations(self, base_size=5):
#         """
#         Plot states over time
#         """
#         grid_plot = closest_factors(self.n_params)
#         fig, ax = plt.subplots(
#             grid_plot[0],
#             grid_plot[1],
#             figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
#         )
#         for prob in len(self.probs):
#             for i, ax in enumerate(ax.flat):
#                 prob.plot_L(param_idx=i, ax=ax)
#
#     def plot_param_density(self, probs, param_idx=0, idxs=None, figsize=(5,5), lam_true=None, ax=None):
#
#         if ax is None:
#             fig, ax = plt.subplots(1, 1, figsize=figsize)
#
#         # Plot initial at first iteration
#         labels = []
#         idxs = np.arange(len(probs)) if idxs is None else idxs
#         _ = probs[idxs[0]].plot_L(ax=ax,
#                         param_idx=param_idx,
#                         initial_kwargs={"color": "black", "linestyle": ":", "fill": True},
#                         update_kwargs=None,
#                         plot_legend=False,
#                         mud_kwargs=None,
#                         lam_true=None)
#         labels += [f'$\pi^{{in}}$']
#         if len(idxs) > 2:
#             alphas = np.linspace(0.1,0.9,len(idxs))
#             for i, j in enumerate(idxs[1:-1]):
#                 if isinstance(probs[j], PCAMUDProblem):
#                     _, l = probs[j].plot_L(ax=ax,
#                                     param_idx=param_idx,
#                                     initial_kwargs=None,
#                                     update_kwargs={"color": "blue", "alpha": alphas[i], "linestyle": "--", "fill": False},
#                                     plot_legend=False,
#                                     mud_kwargs=None,
#                                     lam_true=None
#                     )
#                     labels += [f'$\pi^{{up}}_{{{j}}}$']
#         # plot update at final iteration
#         _, l = probs[idxs[-1]].plot_L(ax=ax,
#                         param_idx=param_idx,
#                         initial_kwargs=None,
#                         update_kwargs={"color": "blue", "linestyle": "-", "fill": True},
#                         plot_legend=False,
#                         mud_kwargs={'color': 'blue'},
#                         lam_true=None,
#         )
#         labels += [f'$\pi^{{up}}$', '$\lambda^{mud}$']
#         for l in lam_true:
#             colors = ["orange", "brown", "purple"]
#             if len(l) == 2:
#                 ax.axvline(
#                     x=l[1][param_idx],
#                     linewidth=3,
#                     color=colors[l[0]],
#                 )
#                 labels += [f'$\lambda^{{\dagger}}_{{{l[0]}}}$']
#             else:
#                 ax.axvline(
#                     x=l[param_idx],
#                     linewidth=3,
#                     color="orange",
#                 )
#                 labels += [f'$\lambda^{{\dagger}}$']
#         labels += l
#
#         ax.legend(labels)
#
#         return ax
#
#     def plot_param_densities(self, probs, idxs=None, figsize=None, base_size=5, lam_true=None):
#         """
#         TODO: FIx to general case when num_params != 4. Use grid_plot
#         """
#         fig, axs = plt.subplots(2, 2, figsize=figsize)
#
#         # idxs = np.arange(1, len(probs)-1, 1 if )
#         grid_plot = self._closest_factors(self.n_params)
#         fig, axs = plt.subplots(
#             grid_plot[0],
#             grid_plot[1],
#             figsize=figsize if figsize is None else
#             (grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
#         )
#         for i, ax in enumerate(axs.flat):
#             self.plot_param_density(probs, param_idx=i, idxs=idxs, ax=ax, lam_true=lam_true)
#
#         return axs
#     # plot_iterations(probs, idxs=np.arange(0, 10, 2), lam_true=[SEIRS_P2])
#
#     def e_r_plot(self, probs, e_r_thresh=None, x_vals=None, x_label='Iteration', ax=None):
#         """
#         Plot the expected ratio
#         """
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(12, 6))
#
#         e_r = [p.expected_ratio() for p in probs]
#         x_vals = np.arange(len(e_r)) if x_vals is None else x_vals
#
#         sns.lineplot(x=x_vals, y=e_r, ax=ax, label='Iterative Expected Ratio', marker="o")
#         xlims = ax.get_xlim()
#         if e_r_thresh is not None:
#             ax.hlines([1 + e_r_thresh, 1 - e_r_thresh], xmin=xlims[0], xmax=xlims[1], color='blue', linestyle=':', label='Threshold $|1 - \mathbb{E}(r)|$')
#         ax.hlines([1], xmin=xlims[0], xmax=xlims[1], color='black', linestyle=':', label='Predictability Assumption $\mathbb{E}(r)$ ≈ 1')
#         ax.set_xlabel(x_label)
#         ax.set_ylabel('$\mathbb{E}(r)$')
#
#     def kl_plot(self, probs, kl_thresh=None, x_vals=None, x_label='Iteration', ax=None):
#         """
#         Plot the expected ratio
#         """
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(12, 6))
#
#         d_kl = [p.divergence_kl() for p in probs]
#         x_vals = np.arange(len(d_kl)) if x_vals is None else x_vals
#
#         sns.lineplot(x=x_vals, y=d_kl, color='green', ax=ax, label='$\mathrm{KL}(\pi^{up}_i | \pi^{up}_{i-1})$', marker="o")
#         if kl_thresh is not None:
#             ax.hlines([kl_thresh], xmin=xlims[0], xmax=xlims[1], color='orange', linestyle=':', label='KL Threshold')
#
#         ax.set_xlabel(x_label)
#         ax.set_ylabel('$\mathrm{KL}()$')
#
#         return ax
#
#     def kl_delta_plot(self, probs, kl_thresh=None, x_vals=None, x_label='Iteration', ax=None):
#         """
#         Plot the expected ratio
#         """
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(12, 6))
#
#         d_kl = [p.divergence_kl() for p in probs]
#         kl_delta = np.abs(np.array(d_kl[1:]) - np.array(d_kl[:-1]))
#         x_vals = np.arange(1, len(kl_delta) + 1) if x_vals is None else x_vals
#
#         label = '$\Delta \mathrm{KL}(\pi^{up}_i | \pi^{up}_{i-1})$'
#         sns.lineplot(x=x_vals, y=kl_delta, color='purple', ax=ax, label=label, marker="o")
#
#         if kl_thresh is not None:
#             ax.hlines([kl_thresh], xmin=xlims[0], xmax=xlims[1], color='orange', linestyle=':', label='KL Threshold')
#
#         ax.set_xlabel('Iteration')
#         ax.set_ylabel('$\Delta \mathrm{KL}()$')
#
#         return ax
#
#     def joint_metrics_plot(self, probs, e_r_thresh=None, kl_thresh=None, y1='e_r', y2='kl', x_vals=None, x_label='Iteration', ax=None):
#         """
#         Plot the expected ratio and KL divergence metrics for a set of problems
#         """
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(12, 6))
#
#         # Check y1 and y1 are iwthin set ['e_r', 'kl', 'kl_delta']
#         if y1 not in ['e_r', 'kl', 'kl_delta'] or y2 not in ['e_r', 'kl', 'kl_delta']:
#             raise ValueError('y1 and y2 must be in set ["e_r", "kl", "kl_delta"]')
#         if y1 == y2:
#             raise ValueError('y1 and y2 must be different')
#
#         e_r = [p.expected_ratio() for p in probs]
#         d_kl = [p.divergence_kl() for p in probs]
#
#         axs = [ax]
#         for i, y in enumerate([y1, y2]):
#             if i > 0:
#                 ax = ax.twinx()
#                 axs.append(ax)
#             if y == 'e_r':
#                 e_r_plot(probs, e_r_thresh=e_r_thresh, x_vals=x_vals, x_label=x_label, ax=ax)
#             if y == 'kl':
#                 kl_plot(probs, kl_thresh=kl_thresh, x_vals=x_vals, x_label=x_label, ax=ax)
#             if y == 'kl_delta':
#                 kl_delta_plot(probs, kl_thresh=kl_thresh, x_vals=x_vals, x_label=x_label, ax=ax)
#
#         axs[0].legend(loc='upper left')
#         axs[1].legend(loc='upper right')
#
#         return axs
