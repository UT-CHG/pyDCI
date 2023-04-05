from typing import Callable, List, Optional, Union
import pdb

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import distributions as dist  # type: ignore
from scipy.stats import gaussian_kde as gkde  # type: ignore
from scipy.stats import rv_continuous  # type: ignore

from pydci.utils import fit_domain, set_shape, put_df, get_df
import matplotlib.pyplot as plt
import seaborn as sns

from pydci.log import logger, enable_log, disable_log

class DCIProblem(object):
    """
    Sets up Data-Consistent Inverse Problem for parameter identification

    Data-Consistent inversion is a way to infer most likely model parameters
    using observed data and predicted data from the model.

    Attributes
    ----------
    samples : ArrayLike
        Array containing parameter samples from an initial distribution.
        Rows represent each sample while columns represent parameter values.
        If 1 dimensional input is passed, assumed that it represents repeated
        samples of a 1-dimensional parameter.
    model : Model
        Model to push forward samples through
    domain : ArrayLike
        Array containing ranges of each parameter value in the parameter
        space. Note that the number of rows must equal the number of
        parameters, and the number of columns must always be two, for min/max
        range. If non specified, will be inferred from the sampls array.
    weights : ArrayLike, optional
        Weights to apply to each parameter sample. Either a 1D array of the
        same length as number of samples or a 2D array if more than
        one set of weights is to be incorporated. If so the weights will be
        multiplied and normalized row-wise, so the number of columns must
        match the number of samples.
    """
    def __init__(
        self,
        lam,
        q_lam,
        obs_dist,
        init_dist = None,
        weights: ArrayLike = None,
        normalize: bool = False,
    ):
        self.lam = set_shape(np.array(lam), (1, -1))
        self.q_lam = set_shape(np.array(q_lam), (-1, 1))
        self.dists = {'initial': init_dist,
                      'predicted': None,
                      'observed': obs_dist,
                      'updated': None}
        self.state = pd.DataFrame(
                np.zeros((self.n_samples, self.n_params + self.n_states + 6)),
                columns=[f'lam_{i}' for i in range(self.n_params)] +
                [f'q_lam_{i}' for i in range(self.n_states)] +
                ['weight', 'pi_in', 'pi_pr', 'pi_obs', 'ratio', 'pi_up'])
        self.state = put_df(self.state, 'q_lam', self.q_lam, size=self.n_params)
        self.state = put_df(self.state, 'lam', self.lam, size=self.n_params)

        # Initialize weights
        self.set_weights(weights, normalize=normalize)

        logger.info('Initialized Data-Consistent Inversion Class')

    @property
    def n_params(self):
        return self.lam.shape[1]

    @property
    def n_states(self):
        return self.q_lam.shape[1]

    @property
    def n_samples(self):
        return self.lam.shape[0]

    def set_domain(self,
                   domain: ArrayLike = None
    ):
        if domain is not None:
            # Assert domain passed in is consistent with data array
            self.domain = set_shape(np.array(domain), (1, -1))
            assert (
                self.domain.shape[0] == self.n_params
            ), f"Size mismatch: domain: {self.domain.shape}, params: {self.n_params}"

        else:
            self.domain = fit_domain(self.lam)

    def set_weights(self, weights: ArrayLike = None, normalize: bool = False):
        """Set Sample Weights

        Sets the weights to use for each sample. Note weights can be one or two
        dimensional. If weights are two dimensional the weights are combined
        by multiplying them row wise and normalizing, to give one weight per
        sample. This combining of weights allows incorporating multiple sets
        of weights from different sources of prior belief.

        Parameters
        ----------
        weights : np.ndarray, List
            Numpy array or list of same length as the `n_samples` or if two
            dimensional, number of columns should match `n_samples`
        normalize : bool, default=False
            Whether to normalize the weights vector.

        Returns
        -------

        Warnings
        --------
        Resetting weights will delete the predicted and updated distribution
        values in the class, requiring a re-run of adequate `set_` methods
        and/or `fit()` to reproduce with new weights.
        """
        if weights is None:
            w = np.ones(self.n_samples)
        else:
            w = np.array(weights)

            # Reshape to 2D
            w = w.reshape(1, -1) if w.ndim == 1 else w

            # assert appropriate size
            assert self.n_samples == w.shape[1], f"`weights` must size {self.n_samples}"

            # Multiply weights column wise for stacked weights
            w = np.prod(w, axis=0)

            if normalize:
                w = np.divide(w, np.linalg.norm(w))

        self.state['weight'] = w

    def set_initial(self,
                    distribution: Optional[rv_continuous] = None,
                    bw_method: Union[str, Callable, np.generic] = None,
                    ):
        """
        Set initial probability distribution of model parameter values
        :math:`\\pi_{in}(\\lambda)`.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, optional
            scipy.stats continuous distribution object from where initial
            parameter samples were drawn from. If none provided, then a uniform
            distribution over domain of the density problem is assumed. If no
            domain is specified for density, then a standard normal
            distribution :math:`N(0,1)` is assumed.

        Warnings
        --------
        Setting initial distribution resets the predicted and updated
        distributions, so make sure to set the initial first.
        """
        if distribution is None:
            self.dists['initial'] = gkde(self.lam.T,
                                         bw_method=bw_method,
                                         weights=self.state['weight'])
        else:
            self.dists['initial'] = distribution

    def set_predicted(
        self,
        distribution: rv_continuous = None,
        bw_method: Union[str, Callable, np.generic] = None,
        weights: ArrayLike = None,
        **kwargs,
    ):
        """
        Set Predicted Distribution

        The predicted distribution over the observable space is equal to the
        push-forward of the initial through the model
        :math:`\\pi_{pr}(Q(\\lambda)`. If no distribution is passed,
        :class:`scipy.stats.gaussian_kde` is used over the predicted values
        :attr:`y` to estimate the predicted distribution.

        Parameters
        ----------
        distribution : :class:`scipy.stats.rv_continuous`, optional
            If specified, used as the predicted distribution instead of the
            default of using gaussian kernel density estimation on observed
            values y. This should be a frozen distribution if using
            `scipy`, and otherwise be a class containing a `pdf()` method
            return the probability density value for an array of values.
        bw_method : str, scalar, or `Callable`, optional
            Method to use to calculate estimator bandwidth. Only used if
            distribution is not specified, See documentation for
            :class:`scipy.stats.gaussian_kde` for more information.
        weights : ArrayLike, optional
            Weights to use on predicted samples. Note that if specified,
            :meth:`set_weights` will be run first to calculate new weights.
            Otherwise, whatever was previously set as the weights is used.
            Note this defaults to a weights vector of all 1s for every sample
            in the case that no weights were passed on upon initialization.
        **kwargs: dict, optional
            If specified, any extra keyword arguments will be passed along to
            the passed ``distribution.pdf()`` function for computing values of
            predicted samples.

        Note: `distribution` should be a frozen distribution if using `scipy`.

        Warnings
        --------
        If passing a `distribution` argument, make sure that the initial
        distribution has been set first, either by having run
        :meth:`set_initial` or :meth:`fit` first.
        """
        if weights is not None:
            self.set_weights(weights)

        if distribution is None:
            # Reweight kde of predicted by weights if present
            distribution = gkde(self.q_lam.T,
                                bw_method=bw_method,
                                weights=self.state['weight'])
        self.dists['predicted'] = distribution

    def solve(self, **kwargs):
        """
        Update Initial Distribution

        Constructs the updated distribution by fitting observed data to
        predicted data with:

        .. math::
            \\pi_{up}(\\lambda) = \\pi_{in}(\\lambda)
            \\frac{\\pi_{ob}(Q(\\lambda))}{\\pi_{pred}(Q(\\lambda))}
            :label: data_consistent_solution

        Note that if initial, predicted, and observed distributions have not
        been set before running this method, they will be run with default
        values. To set specific predicted, observed, or initial distributions
        use the ``set_`` methods.

        Parameters
        -----------
        **kwargs : dict, optional
            If specified, optional arguments are passed to the
            :meth:`set_predicted` call in the case that the predicted
            distribution has not been set yet.

        Returns
        -----------

        """
        if self.dists['initial'] is None:
            self.set_initial()
        if self.dists['predicted'] is None:
            self.set_predicted(**kwargs)

        self.state['pi_in'] = self.dists['initial'].pdf(
                self.lam.T).T
        self.state['pi_obs'] = self.dists['observed'].pdf(
                self.q_lam).prod(axis=1)
        self.state['pi_pr'] = self.dists['predicted'].pdf(
                self.q_lam.T).T.ravel()

        # Store ratio of observed/predicted
        # e.g. to comptue E(r) and to pass on to future iterations
        self.state['ratio'] = np.divide(self.state['pi_obs'],
                                        self.state['pi_pr'])

        # Multiply by initial to get updated pdf
        self.state['pi_up'] = np.multiply(
                self.state['pi_in'] * self.state['weight'],
                self.state['ratio'])

    def expected_ratio(self):
        """Expectation Value of R

        Returns the expectation value of the R, the ratio of the observed to
        the predicted density values.

        .. math::
            R = \\frac{\\pi_{ob}(\\lambda)}
                      {\\pi_{pred}(\\lambda)}
            :label: r_ratio

        If the predictability assumption for the data-consistent framework is
        satisfied, then :math:`E[R]\\approx 1`.

        Parameters
        ----------

        Returns
        -------
        expected_ratio : float
            Value of the E(r). Should be close to 1.0.
        """
        return np.average(self.state['ratio'], weights=self.state['weight'])

    def update(self):
        """Updated Distribution

        Returns the expectation value of the R, the ratio of the observed to
        the predicted density values.

        .. math::
            R = \\frac{\\pi_{ob}(\\lambda)}
                      {\\pi_{pred}(\\lambda)}
            :label: r_ratio

        If the predictability assumption for the data-consistent framework is
        satisfied, then :math:`E[R]\\approx 1`.

        Parameters
        ----------

        Returns
        -------
        expected_ratio : float
            Value of the E(r). Should be close to 1.0.
        """
        # Compute udpated density
        self.dists['updated'] = gkde(
                self.lam.T,
                weights=self.state['ratio'] * self.state['weight'])

        return self.dists['updated']

    def plot_param_state(
        self,
        true_vals=None,
        mud_val=None,
        ax=None,
        param_idxs=None,
        plot_initial=False,
        plot_legend=True,
    ):
        """
        Plotting functions for DCI Problem Class
        """
        sns.set_style("darkgrid")

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6,6))

        param_idxs = range(self.n_params) if param_idxs is None else param_idxs
        number_parameters = len(param_idxs)
        bright_colors = sns.color_palette("bright", n_colors=number_parameters)
        deep_colors = sns.color_palette("deep", n_colors=number_parameters)

        # Plot initial distributions for iteration
        lambda_labels = [f"$\pi^{{up}}_{{\lambda_{j}}}$"
                         for j in param_idxs]
        [
            sns.kdeplot(
                data=self.state,
                x=f'lam_{idx}',
                ax=ax,
                fill=True,
                color=bright_colors[j],
                label=lambda_labels[j],
                weights=self.state['weight'] * self.state['ratio'],
            )
            for j, idx in enumerate(param_idxs)
        ]
        if plot_initial:
            lambda_labels += [f"$\pi^{{in}}_{{\lambda_{j}}}$"
                              for j in param_idxs]
            [
                sns.kdeplot(
                    data=self.state,
                    x=f'lam_{idx}',
                    ax=ax,
                    fill=True,
                    color=bright_colors[j],
                    linestyle=':',
                    label=lambda_labels[len(param_idxs) + j],
                    weights='weight',
                )
                for j, idx in enumerate(param_idxs)
            ]

        # Generate vertical lines for true values
        true_labels = []
        if true_vals is not None:
            for p in param_idxs:
                true_labels += [f"$\lambda^{{\dagger}}_{p} = " +
                                f"{true_vals[0][p]:.4f}$"]
                ax.axvline(
                    x=true_vals[0][p],
                    linewidth=3, color="orange",
                    label=true_labels[-1]
                )

        # Add Shifts as vertical lines to the plot
        mud_labels = []
        if mud_val is not None:
            for p in param_idxs:
                mud_labels += [f"$\lambda^{{MUD}}_{p} = {mud_val[p]:.4f}$"]
                ax.axvline(
                    x=mud_val[p],
                    linewidth=3,
                    color="green",
                    linestyle='--',
                    label=mud_labels[-1],
                )

        # Set plot specifications
        ax.set_xlabel(r"$\Lambda$", fontsize=12)
        if plot_legend:
            ax.legend(
                labels=lambda_labels + true_labels + mud_labels,
                fontsize=12,
                title_fontsize=12,
            )
        plt.tight_layout()

        return ax

    def plot_obs_state(
        self,
        ax=None,
        state_idxs=None,
        plot_pf=False,
        plot_legend=True,
        obs_label='q_lam',
    ):
        """
        Plotting function for DCI Problem Class
        """
        sns.set_style("darkgrid")

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6,6))

        state_idxs = range(self.n_states) if state_idxs is None else state_idxs
        number_parameters = len(state_idxs)
        bright_colors = sns.color_palette("bright", n_colors=number_parameters)
        deep_colors = sns.color_palette("deep", n_colors=number_parameters)

        # Plot predicted distribution
        q_lambda_labels = [f"$\pi^{{pr}}_{{Q(\lambda)_{j}}}$"
                           for j in state_idxs]
        [
            sns.kdeplot(
                data=self.state,
                x=f'{obs_label}_{idx}',
                ax=ax,
                fill=True,
                color=bright_colors[j],
                label=q_lambda_labels[j],
                weights=self.state['weight'],
            )
            for j, idx in enumerate(state_idxs)
        ]

        if plot_pf:
            q_lambda_labels += [f"$\pi^{{pf}}_{{Q(\lambda)_{j}}}$"
                              for j in state_idxs]
            [
                sns.kdeplot(
                    data=self.state,
                    x=f'{obs_label}_{idx}',
                    ax=ax,
                    fill=True,
                    color=bright_colors[j],
                    linestyle=':',
                    label=q_lambda_labels[len(state_idxs) + j],
                    weights=self.state['weight'] * self.state['ratio'],
                )
                for j, idx in enumerate(state_idxs)
            ]

        # TODO: How to plot this using SNS? 
        obs_label = f"$\pi^{{obs}}_{{Q(\lambda)}}$"
        obs_domain = ax.get_xlim()
        obs_x = np.linspace(obs_domain[0], obs_domain[1], 100)
        obs = self.dists['observed'].pdf(obs_x)
        ax.plot(obs_x, obs, color='r', label=obs_label)
        q_lambda_labels += [obs_label] # + q_lambda_labels


        # Set plot specifications
        ax.set_xlabel(r"$\mathcal{D}$", fontsize=12)
        if plot_legend:
            ax.legend(
                labels=q_lambda_labels,
                fontsize=12,
                title_fontsize=12,
            )
        plt.tight_layout()

        return ax
