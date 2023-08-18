"""
Dynamic Model Class

TODO: 
 - Document and add tests

"""
import pdb
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

        allowable_types = [int, float, str, bool, list, tuple, pd.DataFrame, np.ndarray]
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
                # pdb.set_trace()
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

        TODO: Store domain as class attribute
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

    def get_normal_initial_samples(self, num_samples=100, mean=0.0, std_dev=1.0):
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

