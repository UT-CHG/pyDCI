"""
Dynamic Model Class

"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as gkde
from scipy.stats import uniform, entropy
from alive_progress import alive_bar

from pydci.utils import get_uniform_box, add_noise


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
        self.true_param = true_param
        self.measurement_noise = measurement_noise
        self.solve_ts = solve_ts
        self.sample_ts = sample_ts
        self.hot_starts = hot_starts
        self.param_shifts = {} if param_shifts is None else param_shifts

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

    def forward_solve(self, t0, t1, x0=None, samples=None, samples_x0=None):
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
        ts, shift_idx, param_vals = self.get_param_intervals(t0, t1)
        true_vals = np.zeros((len(ts), self.n_states))
        x0_temp = self.x0 if x0 is None else x0

        for i in range(shift_idx[-1] + 1):
            idxs = shift_idx == i
            times = ts[idxs]
            true_param = param_vals[idxs, :][0]
            true_vals[idxs] = self.forward_model(
              x0_temp, times, tuple(true_param))
            x0_temp = true_vals[idxs][-1]

        push_forwards = None
        if samples is not None:
            if samples_x0 is None:
                samples_x0 = np.empty((len(samples), len(self.x0)))
                samples_x0[:] = self.x0 if x0 is None else x0

            push_forwards = np.zeros((len(samples),
                                      len(ts), self.n_states))
            with alive_bar(len(samples),
                           title='Iteration {i} - Forward Solves:',
                           force_tty=True, receipt=False, length=20) as bar:
                for j, s in enumerate(samples):
                    push_forwards[j, :, :] = self.forward_model(
                        samples_x0[j], ts, tuple(s))
                    bar()

        sample_step = int(self.sample_ts / self.solve_ts)
        sample_ts_flag = np.mod(np.arange(len(ts)), sample_step) == 0
        measurements = np.empty((len(ts), self.n_states))
        measurements[:] = np.nan
        measurements[sample_ts_flag] = np.reshape(
            add_noise(true_vals[sample_ts_flag].ravel(),
                      self.measurement_noise),
            true_vals[sample_ts_flag].shape,
        )

        # Store everything in state DF
        state_df = pd.DataFrame(ts, columns=['ts'])
        state_df['shift_idx'] = shift_idx
        state_df['sample_flag'] = sample_ts_flag
        state_df = self._put_df(state_df, 'true_param', param_vals)
        state_df = self._put_df(state_df, 'true_vals', true_vals, typ='state')
        state_df = self._put_df(state_df, 'obs_vals', measurements, typ='state')

        self.push_forwards = push_forwards
        self.state_df = state_df

        return state_df, push_forwards

