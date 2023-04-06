"""
Dynamic Model Class

"""
import numpy as np
import pandas as pd
from alive_progress import alive_bar

from pydci.log import logger
from scipy.stats.distributions import uniform
from pydci.utils import get_uniform_box, add_noise, get_df, put_df


class DynamicModel():
    """
    Class defining a model for parameter estimation

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
                 t0=0.0,
                 measurement_noise=0.05,
                 solve_ts=0.2,
                 sample_ts=1,
                 diff=0.5,
                 hot_starts=True,
                 param_mins=None,
                 param_maxs=None,
                 param_shifts=None,
                 ):

        self.forward_model = forward_model
        self.x0 = x0
        self.t0 = t0
        self.samples = None
        self.samples_x0 = None
        self.true_param = true_param
        self.measurement_noise = measurement_noise
        self.solve_ts = solve_ts
        self.sample_ts = sample_ts
        self.hot_starts = hot_starts
        self.param_shifts = {} if param_shifts is None else param_shifts
        self.param_mins = param_mins
        self.param_maxs = param_maxs

    @property
    def n_params(self) -> int:
        return len(self.true_param)

    @property
    def n_states(self) -> int:
        return len(self.x0)

    @property
    def n_samples(self) -> int:
        if self.samples is not None:
            return len(self.samples)

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
            param_vals[:, p_idx] = self.true_param[p_idx]
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
            true_param = param_vals[idxs, :][0]
            true_vals[idxs] = self.forward_model(
              x0_temp, times, tuple(true_param))
            x0_temp = true_vals[idxs][-1]

        sample_step = int(self.sample_ts / self.solve_ts)
        sample_ts_flag = np.mod(np.arange(len(ts)), sample_step) == 0
        measurements = np.empty((len(ts), self.n_states))
        measurements[:] = np.nan
        measurements[sample_ts_flag] = np.reshape(
            add_noise(true_vals[sample_ts_flag].ravel(),
                      self.measurement_noise),
            true_vals[sample_ts_flag].shape,
        )

        x0_temp = self.x0
        self.t0 = ts[sample_ts_flag][-1]
        self.x0 = measurements[sample_ts_flag][-1]

        push_forwards = None
        if samples is not None:
            if samples_x0 is None:
                if self.samples_x0 is None:
                    self.samples_x0 = self.get_initial_condition(
                            x0_temp, len(samples))
                samples_x0 = self.samples_x0

            push_forwards = np.zeros((len(samples),
                                      len(ts), self.n_states))
            with alive_bar(len(samples),
                           title='Solving model sample set:',
                           force_tty=True, receipt=True, length=20) as bar:
                for j, s in enumerate(samples):
                    push_forwards[j, :, :] = self.forward_model(
                        samples_x0[j], ts, tuple(s))
                    bar()

            if self.hot_starts:
                self.samples_x0[:] = self.get_initial_condition(
                        self.x0, len(samples))
            else:
                self.samples_x0 = push_forwards[:, sample_ts_flag[-1], :]

        # Store everything in state DF
        state_df = pd.DataFrame(ts, columns=['ts'])
        state_df['shift_idx'] = shift_idx
        state_df['sample_flag'] = sample_ts_flag
        state_df = put_df(state_df, 'true_param', param_vals)
        state_df = put_df(state_df, 'true_vals', true_vals, size=self.n_states)
        state_df = put_df(state_df, 'obs_vals',
                          measurements, size=self.n_states)

        self.push_forwards = push_forwards
        self.state_df = state_df

    def mud_args(self,
                 t1,
                 samples=None,
                 num_samples=1000,
                 diff=0.5,
                 weights=None,
                 ):
        """
        """
        # samples: (1) user defined (2) previously initialized (3) uniform dist
        if samples is None:
            if self.samples is None:
                self.samples = self.get_uniform_initial_samples(
                        scale=diff,
                        num_samples=num_samples)
        else:
            self.samples = samples
        self.forward_solve(t1, samples=self.samples)

        # Build arguments for building MUD problem argument
        q_lam = np.array(self.push_forwards[:, np.where(
            self.state_df['sample_flag'].values), :]).reshape(
                    self.n_samples, -1)
        data = get_df(self.state_df.loc[self.state_df['sample_flag']],
                      'obs_vals', size=2)
        data = data.reshape(-1, 1)
        num_ts = self.state_df['sample_flag'].sum()
        args = {'samples': self.samples,
                'q_lam': q_lam,
                'data': data,
                'std_dev': self.measurement_noise,
                'max_nc': self.n_params if self.n_params <= num_ts else num_ts
                }

        return args

    def get_initial_condition(self,
                              x0,
                              num_samples):
        """
        Get Initial condition for a number of samples. Initial condition is
        given by populating x0 with measurement noise for each sample.
        """
        init_conds = np.empty((num_samples, self.n_states))
        init_conds[:] = x0
        init_conds = np.reshape(add_noise(init_conds.ravel(),
                                          self.measurement_noise),
                                (num_samples, self.n_states))
        return init_conds

    def get_uniform_initial_samples(
            self,
            scale=0.5,
            num_samples=1000):
        """
        Generate initial samples from uniform distribution over domain set by
        `self.set_domain`.
        """
        domain = get_uniform_box(
            self.true_param, factor=scale,
            mins=self.param_mins, maxs=self.param_maxs
        )
        loc = domain[:, 0]
        scale = domain[:, 1] - domain[:, 0]
        logger.info(f"Drawing {num_samples} from uniform at:\n" +
                    f"\tloc: {loc}\n\tscale: {scale}")
        samples = uniform.rvs(loc=loc, scale=scale,
                              size=(num_samples, self.n_params))
        return samples

