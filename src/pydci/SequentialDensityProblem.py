"""
Sequential MUD Estimation Algorithms

"""

import random
import itertools
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as gkde
from scipy.stats import uniform, entropy
from alive_progress import alive_bar
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text
from rich.console import Console
from loguru import logger

from pydci.SpatioTemporalAggregator import SpatioTemporalAggregator as STP


def log_table(rich_table):
    """Generate an ascii formatted presentation of a Rich table
    Eliminates any column styling
    """
    console = Console(width=70)
    with console.capture() as capture:
        console.print(rich_table)
    return Text.from_ansi(capture.get())


def enable_log(file=None, level='INFO', fmt=None, serialize=False):
    """
    Turn on logging for module with appropriate message format
    """
    if file is None:
        fmt = "{message}" if fmt is None else fmt
        logger.configure(handlers=[
            {"sink": RichHandler(markup=True, rich_tracebacks=True),
             "level": level, "format": fmt}])
    else:
        def_fmt = "{message}"
        if not serialize:
            def_fmt = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
        fmt = def_fmt if fmt is None else fmt
        logger.configure(handlers=[
            {"sink": file, "serialize": serialize,
             "level": level, "format": fmt}])
    logger.enable('pydci')
    logger.info('Logger initialized')

    return logger


def disable_log():
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


class SequentialDensityProblem():
    """
    Class defining a SequentialDensity Problem for parameter estimation on.

    To initialize the class, a forward model model, and parameters need to be
    sepcified. The main entrypoint for solving the estimation problem is the
    `seq_solve()` method, with the `search_params` class attribute contorlling
    how the the sequential algorithm behaves.

    Attributes
    ----------
    forward_model : callable
        Function that runs the forward model. Should be callable using
    x0 : ndarray
        Initial state of the system.
    true_param : ndarray
        True parameter value for creating the reference data using the passed
        in forward_model.
    """
    def __init__(self,
                 forward_model,
                 x0,
                 true_param,
                 measurement_noise=0.05,
                 solve_ts=0.2,
                 sample_ts=1,
                 diff=0.5,
                 hot_starts=True,
                 param_mins=None,
                 param_maxs=None,
                 param_shifts=None,
                 search_params={}
                 ):

        self.forward_model = forward_model
        self.x0 = x0
        self.true_param = true_param
        self.measurement_noise = measurement_noise
        self.solve_ts = solve_ts
        self.sample_ts = sample_ts
        self.hot_starts = hot_starts
        self.set_domain(diff=diff,
                        param_mins=param_mins,
                        param_maxs=param_maxs)
        self.param_shifts = {} if param_shifts is None else param_shifts

        self.search_params = {
            'nc': 1,
            'method': 'all',
            'best': 'closest',
            'max_tries': 10,
            'exp_thresh': 1e10,
            'mean_exp_delta_thresh': None,
            'kl_thresh_factor': None,
            'reweight': True,
            'resample': True,
            'min_weight_thresh': 0.0,
        }
        self.search_params.update(search_params)

    def _put_df(self, df, name, val, typ='param'):
        """
        Given an n-m dimensional `val`, stores into dataframe `df` with `n`
        rows by unpacking the `m` columns of val into separate columns with
        names `{name}_{j}` where j is the index of the column.
        """
        size = self.n_params if typ == 'param' else self.n_states
        for idx in range(size):
            df[f'{name}_{idx}'] = val[:, idx]
        return df

    def _get_df(self, df, name, typ='param'):
        """
        Gets an n-m dimensional `val` from `df` with `n` columns by retrieving
        the `m` columns of val into from columns of `df` with names `{name}_{j}`
        where j is the index of the column.
        """
        size = self.n_params if typ == 'param' else self.n_states
        val = np.zeros((df.shape[0], size))
        for idx in range(size):
            val[:, idx] = df[f'{name}_{idx}'].values
        return val

    @property
    def n_params(self) -> int:
        return len(self.true_param)

    @property
    def n_states(self) -> int:
        return len(self.x0)

    def _get_uniform_box(self, center, factor=0.5, mins=None, maxs=None):
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

    def set_domain(self,
                   domain=None,
                   diff=0.5,
                   param_mins=None,
                   param_maxs=None):
        """
        Set domain over parameter space to search over during the sequential
        estimation algorithm. A `domain` can be explicitly specified. Otherwise,
        a domain is inferred by taking a range within +- diff * self.true_param
        of the true parameter. If param_mins/maxs are specified, then the domain
        for each parameter is truncated as necessary.
        """
        if domain is None:
            logger.info(f"Computing domain within {diff} of {self.true_param}")
            self.domain = self._get_uniform_box(
                self.true_param, factor=diff, mins=param_mins, maxs=param_maxs
            )
        else:
            self.domain = domain
        logger.info(f"Initialized uniform domain:\n{self.domain}")

    def get_initial_samples(self,
                            num_samples=1000):
        """
        Generate initial samples from uniform distribution over domain set by
        `self.set_domain`.
        """
        loc = self.domain[:, 0]
        scale = self.domain[:, 1] - self.domain[:, 0]
        logger.info(f"Drawing {num_samples} from uniform at:\n" +
                    f"\tloc: {loc}\n\tscale: {scale}")
        samples = uniform.rvs(loc=loc, scale=scale,
                              size=(num_samples, self.n_params))
        return samples

    def get_param_intervals(self):
        """
        Given the algorithm's current iteration, determines the set of time
        steps to solve over, and the value of the true parameter at each
        time-step.
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
            shift_idx[idxs] = i
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
        x0 = self.x0
        for i in range(shift_idx[-1] + 1):
            idxs = shift_idx == i
            times = ts[idxs]
            true_param = param_vals[idxs, :][0]
            true_vals[idxs] = self.forward_model(
              x0, times, tuple(true_param))
            x0 = true_vals[idxs][-1]
        self.x0 = x0

        with alive_bar(len(self.samples),
                       title='Iteration {i} - Forward Solves:',
                       force_tty=True, receipt=False, length=20) as bar:
            for j, s in enumerate(self.samples):
                push_forwards[j, :, :] = self.forward_model(
                    self.samples_x0[j], ts, tuple(s))
                bar()

        if self.hot_starts:
            self.samples_x0[:] = x0
        else:
            self.samples_x0 = push_forwards[:, -1, :]

        sample_step = int(self.sample_ts / self.solve_ts)
        sample_ts_flag = np.mod(np.arange(len(ts)), sample_step) == 0

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

        measurements = np.empty((len(ts), self.n_states))
        measurements[:] = np.nan
        measurements[sample_ts_flag] = self.stp.measurements.reshape(
                self.stp.n_ts, self.n_states)

        # Store everything in state DF
        state_df = pd.DataFrame(ts, columns=['ts'])
        state_df['iteration'] = self.iteration
        state_df['shift_idx'] = shift_idx
        state_df['sample_flag'] = sample_ts_flag
        state_df = self._put_df(state_df, 'true_param', param_vals)
        state_df = self._put_df(state_df, 'true_vals', true_vals, typ='state')
        state_df = self._put_df(state_df, 'obs_vals', measurements, typ='state')

        self.push_forwards = push_forwards
        self.state_df = state_df

    def _get_ts_combinations(
        self
    ):
        """
        Utility function to determine sets of ts combinations to iterate through
        """
        n_ts = self.stp.n_ts
        method = self.search_params['method']

        if method == 'all':
            combs = [list(np.arange(n_ts))]
        elif method == 'linear':
            combs = [list(np.arange(i)) for i in range(1, n_ts + 1)]
        elif method == 'random':
            max_tries = self.search_params['max_tries']

            # Divide the max#tries amongs the number of timesteps available
            if n_ts < max_tries:
                num_ts_list = range(1, n_ts + 1)
                tries_per = int(max_tries/n_ts)
            else:
                num_ts_list = range(1, n_ts + 1, int(n_ts/max_tries))
                tries_per = 1

            combs = []
            ts_choices = range(0, n_ts)
            for num_ts in num_ts_list:
                possible = list([list(x) for x in itertools.combinations(
                    ts_choices, num_ts)])
                tries_per = tries_per if tries_per < len(possible) else len(possible)
                combs += random.sample(possible, tries_per)

        return combs

    def _search_mud_parallel(
        self,
    ):
        """
        Helper function for find_best_mud_estimate
        """
        nc = self.search_params['nc']

        # Get combinations to iterate through
        combs = self._get_ts_combinations()

        ts_combinations = np.zeros((self.stp.n_ts, 1 + len(combs)), dtype=int)
        ts_combinations[:, 0] = self.iteration
        results = np.zeros((len(combs), self.n_params + 4))
        results[:, 0] = self.iteration

        # Execute tries to mud algorithm in parallel
        i = 0
        probs = []
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
                    logger.info('MUD problem using {args[2]} idxs failed!')
                    continue
                probs.append(mud_prob)
                ts_combinations[:, i + 1] = np.array(
                    [0 if i not in args[2] else 1
                     for i in range(self.stp.n_ts)])
                results[i, 1:(self.n_params + 1)] = mud_prob.estimate()
                results[i, self.n_params + 1] = np.linalg.norm(
                    mud_prob.estimate() - self.true_param)
                results[i, self.n_params + 2] = mud_prob.expected_ratio()
                results[i, self.n_params + 3] = entropy(
                    mud_prob._ob, mud_prob._pr)
                i += 1

        return probs, ts_combinations, results

    def find_best_mud_estimate(
        self,
    ):
        """
        Find best MUD Estimate

        Given a SpatioTemporalProblem stp, search for the best mud estimate using
        up to nc principal components, and from 1 to stp.num_ts data points.
        Best e_r is characterized by that which is closest to 1.
        """
        best = self.search_params['best']
        exp_thresh = self.search_params['exp_thresh']

        probs, ts_combinations, results = self._search_mud_parallel()

        # Parse DataFrame with choices made in sampling of data
        ts_columns = ['iteration'] + [f'{i}' for i in range(len(probs))]
        ts_df = pd.DataFrame(ts_combinations, columns=ts_columns)

        # Parse DataFrame with results of mud estimations for each ts choice
        results_cols = ['iteration']
        results_cols += [f'lam_MUD_{i}' for i in range(self.n_params)]
        results_cols += ['l2_err', 'e_r', 'kl']
        res_df = pd.DataFrame(results, columns=results_cols)
        res_df['mean_e_r'] = res_df['e_r'].mean()
        res_df['e_r_std'] = res_df['e_r'].std()
        res_df['min_e_r'] = res_df['e_r'].min()
        res_df['max_e_r'] = res_df['e_r'].max()
        res_df['predict_delta'] = np.abs(res_df['e_r'] - 1.0)
        res_df['within_thresh'] = res_df['predict_delta'] <= exp_thresh
        # res_df['closest'] = res_df['predict_delta'] <= \
        #     res_df['predict_delta'].min()
        res_df['closest'] = np.logical_and(
            res_df['predict_delta'] <=
            res_df[res_df['within_thresh']]['predict_delta'].min(),
            res_df['within_thresh'])
        res_df['max_kl'] = np.logical_and(
            res_df['kl'] >= res_df[res_df['within_thresh']]['kl'].max(),
            res_df['within_thresh'])
        res_df['min_kl'] = np.logical_and(
            res_df['kl'] <= res_df[res_df['within_thresh']]['kl'].min(),
            res_df['within_thresh'])

        # Determine best MUD estimate
        am = ['closest', 'min_kl', 'max_kl']
        if best not in am:
            raise ValueError(f"Unrecognized best method: {best}. Allowed: {am}")

        best_idx = res_df[best].argmax()
        self.dfs['ts_choices'].append(ts_df)
        self.dfs['results'].append(res_df)

        res = {
            "best": probs[best_idx],
            "best_ts_choice": ts_df[f"{best_idx}"].values,
            "best_l2_err": res_df.loc[best_idx]['l2_err'],
            "best_kl": res_df.loc[best_idx]['kl'],
            "mean_e_r": res_df['mean_e_r'].values[0],
            "e_r_std": res_df['e_r_std'].values[0],
            "min_e_r": res_df['min_e_r'].values[0],
            "max_e_r": res_df['max_e_r'].values[0],
            "probs": probs,
        }

        return res

    def _detect_shift(
        self,
    ):
        """
        """
        mean_exp_delta_thresh = self.search_params['mean_exp_delta_thresh']
        kl_thresh_factor = self.search_params['kl_thresh_factor']

        if len(self.mud_res) < 2:
            # Need at least two results to detect shifts.
            return False

        shift = False
        # Mean condition - If significan shift in the mean exp_r value detected
        if mean_exp_delta_thresh is not None:
            mean_e_r = self.dfs['results'][-1]['e_r'].mean()
            prev_mean_e_r = self.dfs['results'][-2]['e_r'].mean()
            if np.abs(prev_mean_e_r - mean_e_r) > mean_exp_delta_thresh:
                shift = True

        # KL Div. Condition - If KL > 3.0 (3 std-dev since obs is N(0,1))
        if kl_thresh_factor is not None:
            if self.dfs['results'][-1]['kl'].values[0] >= kl_thresh_factor:
                shift = True

        return shift

    def _determine_action(
        self,
    ):
        """
        """
        # Get search parameters relevant for determining next step
        exp_thresh = self.search_params['exp_thresh']
        min_weight_thresh = self.search_params['min_weight_thresh']
        resample = self.search_params['resample']
        reweight = self.search_params['reweight']

        best = self.mud_res[-1]['best']
        best_e_r = best.expected_ratio()
        res = {'action': 'NONE',
               'weights': None}
        if self._detect_shift():
            # TODO: Add old logger call
            logger.info(f'Shift detected at {self.iteration}')
            if resample:
                res['samples'] = self.get_initial_samples()
            res['action'] = 'RESET'
        elif np.abs(1.0 - best_e_r) <= exp_thresh:
            if reweight:
                min_ratio = best._r.min()
                if min_ratio >= min_weight_thresh:
                    logger.info(
                        "\tUpdating weights: e_r within threshold 1+-"
                        + f"{exp_thresh:0.2f}, reweight set, and "
                        + f"{min_ratio:.2e} >= {min_weight_thresh:.2e}"
                    )
                    res['weights'] = best._r
                    res['action'] = 'RE-WEIGHT'
                else:
                    logger.info(
                        f"Reweight set but {min_ratio:.2e} < " +
                        f"{min_weight_thresh:.2e}")
            if resample and res['action'] != 'RE-WEIGHT':
                logger.info(
                    f"\tRe-Sampling: e_r within 1+-{exp_thresh:0.2f}")
                res['up_dist'] = gkde(self.samples.T, weights=best._r)
                res['samples'] = res['up_dist'].resample(
                    size=len(self.samples)).T
                res['action'] = 'UPDATE'
        else:
            logger.info('No action taken!')

        return res

    def iteration_update(self,
                         num_samples_to_save=10):
        """
        Update class attributes according to action determined per iteration.
        """
        # Determine next action to take
        action = self._determine_action()

        # Add action taken to results df
        self.dfs['results'][-1]['action'] = action['action']

        # Add to full state DF
        best_sample = np.argmax(self.mud_res[-1]['best']._up)
        worst_sample = np.argmin(self.mud_res[-1]['best']._up)
        if self.iteration == 0:
            rand_idxs = random.choices(range(len(self.samples)),
                                       k=num_samples_to_save)
            self.rand_idxs = rand_idxs
        else:
            rand_idxs = self.rand_idxs

        state_df = self._put_df(self.state_df, 'best',
                                self.push_forwards[best_sample],
                                typ='state')
        state_df = self._put_df(state_df, 'worst',
                                self.push_forwards[worst_sample],
                                typ='state')
        for idx in rand_idxs:
            state_df = self._put_df(state_df, f'random_{idx}',
                                    self.push_forwards[idx],
                                    typ='state')
        self.dfs['state'].append(state_df)

        # Samples DF - Store parameter samples, associated weights and ratios
        # This DF should be sufficient to plot init/update distributions by
        # doing KDEs on the parameter values using the weights/ratios.
        samples_df = pd.DataFrame(
                self.samples,
                columns=[f'lam_{i}' for i in range(self.n_params)])
        samples_df['ratio'] = self.mud_res[-1]['best']._r
        samples_df['weights'] = self.mud_res[-1]['best']._weights
        samples_df['up_weights'] = action['weights']
        samples_df['iteration'] = self.iteration
        self.dfs['samples'].append(samples_df)

        # Update for next iteration
        self.mud_res[-1]['action'] = action['action']
        self.weights = action['weights']
        if 'up_dist' in action.keys():
            self.up_dist = action['up_dist']
        if 'samples' in action.keys():
            self.samples = action['samples']
        self.iteration = self.iteration + 1

    def seq_solve(
        self,
        time_windows,
        num_samples=1000,
        init_seed=None,
        obs_seed=None,
    ):
        """
        Sequential estimation algorithm
        """
        # TODO: Detect if previous solve?
        np.random.seed(init_seed)  # Initial seed for sampling
        time_windows.sort()
        if 0 not in time_windows:
            time_windows.insert(0, 0)

        # Initialize Solver
        self.iteration = 0
        self.num_its = len(time_windows) - 1
        self.time_windows = time_windows
        self.samples = self.get_initial_samples(num_samples)
        self.samples_x0 = np.ones((len(self.samples), len(self.x0)))
        self.samples_x0[:] = self.x0
        self.weights = None
        self.mud_res = []
        self.up_dist = gkde(self.samples.T)
        self.dfs = {
                'state': [],
                'results': [],
                'samples': [],
                'ts_choices': [],
                }

        logger.info(f'Solver init - {self.num_its} windows: {time_windows}')
        for i in range(len(self.time_windows) - 1):
            logger.info(f'Iteration {self.iteration}.') 
            self.forward_solve()
            self.mud_res.append(self.find_best_mud_estimate())
            self.iteration_update()
            logger.info(
                f"Iteration {self.iteration}: " +
                f"[{time_windows[self.iteration-1]}, " +
                f"{time_windows[self.iteration]}] Summary:\n" +
                f"{log_table(self.get_summary_row())}")

    def get_summary_row(
            self,
    ):
        """
        """
        fields = ['Iteration', 'Action', 'L_2', 'KL', 'Best(E(r))',
                  'Mean(E(r))', 'std(E(r))']

        table = Table(show_header=True, header_style="bold magenta")
        cols = ['Key', 'Value']
        for c in cols:
            table.add_column(c)

        r = self.mud_res[-1]
        row = (str(len(self.mud_res)),
               f"{r['action']}",
               f"{r['best_l2_err']:0.3f}",
               f"{r['best_kl']:0.3f}",
               f"{r['best'].expected_ratio():0.3f}",
               f"{r['mean_e_r']:0.3f}",
               f"{r['e_r_std']:0.3f}")
        for i in range(len(fields)):
            table.add_row(fields[i], row[i])

        return table

    def get_full_df(
        self,
        df='state',
        iterations=None,
    ):
        """
        Concatenate stored df
        """

        if df not in self.dfs.keys():
            raise ValueError(f'{df} not one of {self.dfs.keys()}')

        dfs = self.dfs[df]
        if iterations is not None:
            dfs = [dfs[i] for i in range(len(dfs)) if i in iterations]

        return pd.concat(dfs, axis=0)


disable_log()
