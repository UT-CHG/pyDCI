"""
Sequential MUD Estimation Algorithms

"""

import random
import pdb
import itertools
import concurrent.futures
import numpy as np
from mud.base import SpatioTemporalProblem as STP
import pandas as pd
from scipy.stats import gaussian_kde as gkde
from scipy.stats import uniform, entropy
from rich.console import Console
from alive_progress import alive_bar
from rich.table import Table
from loguru import logger
from rich.logging import RichHandler


def enable(file=None,
           level='INFO', fmt="{message}"):
    """
    Turn on logging for module with appropriate message format
    """
    if file is None:
        logger.configure(handlers=[
            {"sink": RichHandler(markup=True, rich_tracebacks=True),
             "level": level, "format": fmt}])
    else:
        logger.configure(handlers=[
            {"sink": file, "serialize": True,
             "level": level, "format": fmt, "rotation": "10 MB",
             "enqueue": True}])
    logger.enable('pydci')
    logger.info('Logger initialized')

    return logger


def disable():
    """
    Turn of logging
    """
    logger.disable('pydci')


def _try_mud(spt, nc=1, times_mask=None, weights=None):
    """
    Wrapper for trying a MUD problem and catching exceptions

    """
    try:
        mud_prob = spt.mud_problem(
            num_components=nc, times_mask=times_mask, sample_weights=weights
        )
    except Exception:
        logger.exception(f"\t{nc}: - Unable to generate mud_problem")
        return None

    try:
        mud_prob.estimate()
    except Exception:
        logger.exception(f"\t{nc}: - Unable to create mud estimate")
        return None

    return mud_prob


def get_uniform_box(center, factor=0.5, mins=None, maxs=None):
    """
    Generate a domain of [min, max] values
    """
    center = np.array(center)
    if np.sum(center) != 0.0:
        loc = center - np.abs(center * factor)
        scale = 2 * center * factor
    else:
        loc = center - factor
        scale = 2 * factor
    domain = np.array(list(zip(loc, np.array(loc) + np.array(scale))))
    if mins is not None:
        for i, d in enumerate(domain):
            if d[0] < mins[i]:
                d[0] = mins[i]
    if maxs is not None:
        for i, d in enumerate(domain):
            if d[1] > maxs[i]:
                d[1] = maxs[i]

    return domain


class Model():
    """
    Class defining a Dynamic Model to perform parameter estimation on.

    Attributes
    ----------
    forward_model : callable
        Function that runs the forward model. Should be callable using
    """
    def __init__(self,
                 forward_model,
                 x0,
                 true_param,
                 measurement_noise=0.05,
                 solve_ts=0.2,
                 sample_ts=1,
                 diff=0.5,
                 param_mins=None,
                 param_maxs=None,
                 param_shifts=None):

        self.forward_model = forward_model
        self.x0 = x0
        self.true_param = true_param
        self.measurement_noise = measurement_noise
        self.solve_ts = solve_ts
        self.sample_ts = sample_ts
        self.set_domain(diff=diff,
                        param_mins=param_mins,
                        param_maxs=param_maxs)

    @property
    def n_params(self) -> int:
        return len(self.true_param)

    @property
    def n_states(self) -> int:
        return len(self.x0)

    def set_domain(self,
                   domain=None,
                   diff=0.5,
                   param_mins=None,
                   param_maxs=None):
        """
        Set domain over parameter space to search over.
        """
        if domain is None:
            logger.info(f"Computing domain within {diff} of {self.true_param}")
            self.domain = get_uniform_box(
                self.true_param, factor=diff, mins=param_mins, maxs=param_maxs
            )
        else:
            self.domain = domain
        logger.info(f"Initialized uniform domain:\n{self.domain}")

    def get_initial_samples(self,
                            num_samples=1000):
        """
        Generate initial samples
        """
        loc = self.domain[:, 0]
        scale = self.domain[:, 1] - self.domain[:, 0]
        logger.info(f"Drawing {num_samples} from uniform at:\n" +
                    f"\tloc: {loc}\n\tscale: {scale}")
        samples = uniform.rvs(loc=loc, scale=scale,
                              size=(num_samples, self.n_params))
        return samples

    def _put_df(self, df, name, val, typ='param'):
        """
        """
        size = self.n_params if typ == 'param' else self.n_states
        for idx in range(size):
            df[f'{name}_{idx}'] = val[:, idx]
        return df

    def _get_df(self, df, name, typ='param'):
        """
        """
        size = self.n_params if typ == 'param' else self.n_states
        val = np.zeros((df.shape[0], size))
        for idx in range(size):
            val[:, idx] = df[f'{name}_{idx}'].values
        return val

    def get_param_intervals(self):
        """
        Determine timesteps and
        """
        t0 = self.time_windows[self.iteration]
        t1 = self.time_windows[self.iteration + 1]
        time_window = t1 - t0
        solve_step = int(time_window / self.solve_ts)
        ts = np.linspace(t0, t1, solve_step)
        shift_times = list(self.param_shifts.keys())
        shift_times.sort()
        shift_idx = np.zeros((len(ts)), dtype=int)
        param_vals = np.zeros((len(ts), self.n_params))
        for p_idx in range(self.n_params):
            param_vals[:, p_idx] = self.true_param[p_idx]
        for i, st in enumerate(shift_times):
            idxs = ts > st
            shift_idx[idxs] = i + 1
            for p_idx in range(self.n_params):
                param_vals[idxs, p_idx] = self.param_shifts[st][p_idx]

        return ts, shift_idx, param_vals

    def forward_solve(self):
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
        ts, shift_idx, param_vals = self.get_param_intervals()
        true_vals = np.zeros((len(ts), self.n_states))
        push_forwards = np.zeros((len(self.samples), len(ts), self.n_states))
        for i in range(shift_idx[-1] + 1):
            idxs = shift_idx == i
            times = ts[idxs]
            true_param = param_vals[idxs, :][0]
            true_vals[idxs] = self.forward_model(
              self.x0, times, tuple(true_param))
            for j, s in enumerate(self.samples):
                push_forwards[j, idxs, :] = self.forward_model(
                    self.x0, times, tuple(s))
            self.x0 = true_vals[idxs][-1]

        sample_step = int(self.sample_ts / self.solve_ts)
        sample_ts_flag = np.mod(np.arange(len(ts)), sample_step) == 0

        # Store everything in state DF
        state_df = pd.DataFrame(ts, columns=['ts'])
        state_df['shift_idx'] = shift_idx
        state_df['sample_flag'] = sample_ts_flag
        state_df = self._put_df(state_df, 'true_param', param_vals)
        state_df = self._put_df(state_df, 'true_vals', true_vals, typ='state')

        # TODO: Store some a sample set of push_forwards in state_df
        self.state_df = state_df

        self.stp = STP(
            df={
                "sample_dist": "u",
                "domain": self.domain,
                "sensors": np.zeros((self.n_states, 2)),
                "times": ts[sample_ts_flag],
                "lam_ref": self.true_param,
                "std_dev": self.measurement_noise,
                "true_vals": true_vals[sample_ts_flag],
                "lam": self.samples,
                "data": push_forwards[:, sample_ts_flag, :],
            }
        )
        self.stp.measurements_from_reference()

    def _get_ts_combinations(
        self, n_ts, max_tries=10, randomized=True
    ):
        """
        Utility function to determine sets of ts combinations to iterate through
        """
        # Divide the max#tries amongs the number of timesteps available
        if n_ts < max_tries:
            num_ts_list = range(1, n_ts)
            tries_per = int(max_tries/n_ts)
        else:
            num_ts_list = range(1, n_ts, int(n_ts/max_tries))
            tries_per = 1

        combs = []
        ts_choices = range(1, n_ts)
        for num_ts in num_ts_list:
            if randomized:
                possible = list([list(x) for x in itertools.combinations(
                    ts_choices, num_ts)])
                tries_per = tries_per if tries_per < len(possible) else len(possible)
                combs += random.sample(possible, tries_per)
            else:
                combs += [list(np.arange(num_ts))]

        return combs

    def find_best_mud_estimate(
        self,
        nc=1,
        max_tries=10,
        randomized=True,
    ):
        """
        Find best MUD Estimate

        Given a SpatioTemporalProblem stp, search for the best mud estimate using
        up to nc principal components, and from 1 to stp.num_ts data points.
        Best e_r is characterized by that which is closest to 1.
        """
        best = None
        best_e_r = None
        probs = []
        e_rs = []
        ts_combinations = []
        combs = self._get_ts_combinations(self.stp.n_ts,
                                          max_tries=max_tries,
                                          randomized=randomized)

        # Execute tries to mud algorithm in parallel
        mud_args = [[self.stp, nc, ts, self.weights] for ts in combs]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            future_mapping = {}
            for arg in mud_args:
                future = executor.submit(_try_mud,
                                         arg[0], arg[1], arg[2], arg[3])
                future_mapping[future] = arg
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                args = future_mapping[future]
                mud_prob = future.result()
                if mud_prob is None:
                    # TODO: Track errors?
                    continue
                exp_r = mud_prob.expected_ratio()
                e_rs.append(exp_r)
                probs.append(mud_prob)
                ts_combinations.append(args[2])
                if best is None or np.abs(1 - exp_r) < np.abs(1 - best_e_r):
                    best = mud_prob
                    best_ts_choice = args[2]
                    best_e_r = exp_r

        res = {
            "best": best,
            "best_ts_choice": best_ts_choice,
            "e_rs": np.array(e_rs),
            "probs": probs,
            "times": ts_combinations
        }

        return res

    def detect_shift(
        self,
        mean_exp_delta_thresh=0.5,
        kl_thresh_factor=3.0,
    ):
        """
        """
        if len(self.mud_res) == 1:
            # Shift not possible on first iteration
            return False

        # Mean condition - If significan shift in the mean exp_r value detected
        mean_e_r = self.mud_res[-1]['e_rs'].mean()
        prev_mean_e_r = self.mud_res[-2]['e_rs'].mean()
        if not np.abs(prev_mean_e_r - mean_e_r) > mean_exp_delta_thresh:
            return False

        # KL Div. Condition - If KL > 3.0 (3 std-dev since obs is N(0,1))
        kl = entropy(self.mud_res[-1]['best']._ob, self.mud_res[-1]['best']._pr)
        if kl < kl_thresh_factor:
            return False

        return True

    def iteration_update(
        self,
        exp_thresh=0.2,
        min_weight_thresh=1e-5,
        reweight=True,
    ):
        """
        """
        best = self.mud_res[-1]['best']
        best_e_r = best.expected_ratio()
        min_ratio = best._r.min()
        res = {'action': 'NONE',
               'weights': None}
        if self.detect_shift():
            # TODO: Add old logger call
            res['samples'] = self.get_initial_samples()
            res['action'] = 'RESET'
        if np.abs(1.0 - best_e_r) <= exp_thresh:
            if reweight and min_ratio > min_weight_thresh:
                logger.info(
                    "\tUpdating weights: e_r within threshold 1+-"
                    + f"{exp_thresh}, reweight set, and min weight "
                    + f"> {min_weight_thresh}"
                )
                self.weights = best._r
                res['action'] = 'RE-WEIGHT'
            else:
                logger.info(
                    f"\tDrawing new samples: e_r within 1+-{exp_thresh}")
                res['up_dist'] = gkde(self.samples.T, weights=best._r)
                res['samples'] = res['up_dist'].resample(
                    size=len(self.samples)).T
                res['action'] = 'UPDATE'

        self.best = best
        self.update_res = res

    def update_dfs(self):
        """
        Update DataFrame's tracking algorithms progression
        """
        # TODO: FINISH 

        # Update state df add:
        # - state for best(e_r)
        # - state for min/max(e_r)
        # - state for m number of other push-forwards from sample

        # Create E_r df (see function below)

        # Create Samples DF (see whiteboard)

        # Each of these should have an col they can be merged on with
        # the total df (if it exists yet). which is iteration = self.iteration
        pass

    def seq_solve_iteration(
        self,
    ):
        """
        One iteration of sequential solver, iterating from t0 to t1.

        Parmaters:
          NC - number of components to use in best mud search
          max_tries = 10/Randomized  - Randomize search/limit # tries
        """
        if self.iteration + 1 > len(self.time_windows):
            raise ValueError('Solver complete already')
        logger.info(f'Starting iteration {self.iteration}')
        self.forward_solve()
        self.mud_res.append(
            self.find_best_mud_estimate(
                nc=1,
                max_tries=10,
                randomized=True,
              )
            )
        self.iteration_update()

        # Store dataframes
        self.update_dfs()

        # Update for next iteration
        self.weights = self.update_res['weights']
        if 'up_dist' in self.update_res.keys():
            self.up_dist = self.update_res['up_dist']
        if 'samples' in self.update_res.keys():
            self.samples = self.update_res['samples']
        self.iteration = self.iteration + 1
        self.x0 = self.true_vals[-1]

    def seq_solve_init(
        self,
        time_windows,
        num_samples=1000,
        param_shifts=None,
        init_seed=None,
    ):
        """
        Initialize a sequential solve
        """
        # TODO: Detect if previous solve?
        np.random.seed(init_seed)  # Initial seed for sampling
        if 0 not in time_windows:
            time_windows.append(0)
        time_windows.sort()

        # Initialize Solver
        self.iteration = 0
        self.num_its = len(time_windows) - 1
        self.time_windows = time_windows
        self.samples = self.get_initial_samples(num_samples)
        self.weights = None
        self.mud_res = []
        self.up_dist = gkde(self.samples.T)
        self.param_shifts = {} if param_shifts is None else param_shifts

        logger.info(f'Solver init - {self.num_its} windows: {time_windows}')

    def seq_solve(
        self,
        time_windows,
        num_samples=1000,
        exp_thresh=0.1,
        nc=4,
        reweight=True,
        min_weight_thresh=1e-5,
        mean_exp_thresh=0.1,
        mean_exp_delta_thresh=0.5,
        param_shifts=None,
        random_search=True,
        init_seed=None,
        obs_seed=None,
    ):
        """
        Sequential estimation algorithm
        """
        self.seq_solve_init(time_windows, num_samples=num_samples,
                            param_shifts=param_shifts, init_seed=init_seed)
        with alive_bar(self.time_windows, dual_line=True,
                       title='MUD Iterations',
                       force_tty=True) as bar:
            for i in range(len(self.time_windows)):
                self.seq_solve_iteration()
                bar()

        # table = get_expected_ratio_summary(mud_probs)
        # console.log(table)

        # ret = {
        #     "domain": domain,
        #     "times": ts,
        #     "push_forwards": pfs,
        #     "true_values": tvs,
        #     "spatio_temporal_probs": stps,
        #     "mud_probs": mud_probs,
        #     "covariance": covariance,
        #     "correlation": correlation,
        # }

        # return ret


# def get_expected_ratio_summary(res):
#     """
#     Currently summarize an iteration with the following stats:
# 
#       (1) Best(E(r)) from current iteration
#       (2) Best mud estimate
#       (3) Mean(E(r)) from combinations of points searche
#       (4) StdDev(E(r)) from combinations searched
#       (4) Min(E(r)) from combinations searched
#       (4) Max(E(r)) from combinations searched
#     """
#     cols = ['Best E(r)', 'Best MUD', 'Mean(E(r))',
#               'StdDev(E(r))', 'Min(E(r)', 'Max(E(r))']
# 
#     table = Table(show_header=True, header_style="bold magenta")
#     for c in cols:
#         table.add_column(c)
# 
#     for r in res:
#         row = (str(r['best'].expected_ratio()),
#                str(r['best'].estimate()),
#                str(r['e_rs'].mean()),
#                str(r['e_rs'].std()),
#                str(r['e_rs'].min()),
#                str(r['e_rs'].max()))
#         table.add_row(*row)
# 
#     return table
# 
# 
# def print_summary(stp, weights, nc, res):
#     """
#     Print to stdout a summary of a mud point search.
#     """
#     weights = "" if weights is not None else "un-"
#     print(f"\t ... SEARCH SUMMARY ...\n\tNC = {nc}, {weights}weighted, ")
#     print(f"\tSearched {len(res['times'])} different combinations")
#     print(f"\tBest choice: {len(res['best_ts_choice'])}/" +
#           f"{stp.n_ts} points: {res['best_ts_choice']}")
#     print(f"\t\tBest(E(r)) = {res['best'].expected_ratio()}")
#     print(f"\t\tBest(Mud) = {res['best'].estimate()}")
#     print(
#         f"\t\tMean(E(r)) = {res['e_rs'].mean()}, "
#         + f"STD(E(r)) = {res['e_rs'].std()}"
#     )
#     print(
#         f"\t\tMIN(E(r)) = {res['e_rs'].min()}, "
#         + f"MAX(E(r))) = {res['e_rs'].max()}"
#     )
# 
# 
# def convert_push_forward(push_forward_array, interval):
#     number_runs, number_timesteps, number_states = push_forward_array.shape
#     header = [
#         [f"Push_Forward_{i}" for i in range(number_runs) for j in range(number_states)],
#         [f"X_{j}" for j in range(number_states)] * number_runs,
#     ]
#     reshaped_array = push_forward_array.reshape(
#         number_timesteps, number_runs, number_states
#     )
#     push_forward_df = pd.DataFrame(
#         reshaped_array.reshape(number_timesteps, -1), columns=header
#     )
# 
#     time_interval = pd.Categorical([f"Time_Interval_{interval}"] * len(push_forward_df))
#     push_forward_df["Interval"] = time_interval
# 
#     return push_forward_df
# 
# 
# def correlation_matrix(cov_matrix):
#     Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
#     return Dinv @ cov_matrix @ Dinv
# 

disable()

#v     def get_param_intervals(self, times):
#v         """
#v         Given an array of times, split up into appropriate
#v         """
#v         shift_times = list(self.param_shifts.keys())
#v         shift_times.sort()
#v 
#v         intervals = [(self.true_param, times)]
#v         for st in shift_times:
#v             if st > times[-1]:
#v                 break
#v             elif st > times[0] and st <= times[-1]:
#v                 first_int = times[times < st]
#v                 second_int = times[times >= st]
#v                 intervals[-1] = (intervals[-1][0], first_int)
#v                 intervals.append((self.param_shifts[st], second_int))
#v 
#v         return intervals

#     def find_best_mud_estimate(
#         self,
#         nc=2,
#         weights=None,
#         max_tries=10,
#     ):
#         """
#         Find best MUD Estimate
# 
#         Given a SpatioTemporalProblem SPT, search for the best mud estimate using
#         up to nc principal components, and from 1 to spt.num_ts data points.
#         Best e_r is characterized by that which is closest to 1.
#         """
# 
#         if max_tries is not None:
#             res = self.find_best_mud_estimate_random(
#                 nc=nc, weights=weights)
#         best = None
#         best_e_r = None
#         probs = []
#         e_rs = []
#         for num_ts in range(self.spt.n_ts)[1:]:
#             mud_prob = _try_mud(self.spt, nc=nc,
#                                 times_mask=range(num_ts), weights=weights)
#             if mud_prob is None:
#                 continue
#             exp_r = mud_prob.expected_ratio()
#             e_rs.append(exp_r)
#             probs.append(mud_prob)
#             if best is None or np.abs(1 - exp_r) < np.abs(1 - best_e_r):
#                 best = mud_prob
#                 best_num_ts = num_ts
#                 best_e_r = exp_r
#         e_rs = np.array(e_rs)
# 
#         res = {
#                 "best": best,
#                 "best_ts_choice": np.arange(best_num_ts),
#                 "e_rs": e_rs,
#                 "probs": probs,
#             }
# 
#         return res
