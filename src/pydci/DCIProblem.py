import pdb
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from scipy.stats import rv_continuous  # type: ignore
from scipy.stats import entropy
from scipy.stats import gaussian_kde as gkde  # type: ignore

from pydci.log import logger
from pydci.utils import fit_domain, put_df, set_shape


class DCIProblem(object):
    """
    Data Consistent Inversion Problem

    Solves a Data-Consistent Inversion Problem as formulated first in [1].

    Attributes
    ----------
    lam: ArrayLike
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

    References
    ----------
    [1] T. Butler, J. Jakeman, and T. Wildey, “Combining Push-Forward Measures
    and Bayes’ Rule to Construct Consistent Solutions to Stochastic Inverse
    Problems,” SIAM J. Sci. Comput., vol. 40, no. 2, pp. A984–A1011, Jan. 2018,
    doi: 10.1137/16M1087229.
    """

    def __init__(
        self,
        lam,
        q_lam,
        obs_dist,
        init_dist: rv_continuous = None,
        weights: ArrayLike = None,
        normalize: bool = False,
    ):
        self.init_state(lam, q_lam)
        self.dists = {
            "initial": init_dist,
            "predicted": None,
            "observed": obs_dist,
            "updated": None,
        }

        # Initialize weights
        self.set_weights(weights, normalize=normalize)

        self.result = None

    @property
    def n_params(self):
        return self.lam.shape[1]

    @property
    def n_states(self):
        return self.q_lam.shape[1]

    @property
    def n_samples(self):
        return self.lam.shape[0]

    @property
    def pi_up(self):
        """Updated Distribution"""
        # Compute udpated density
        if self.dists["updated"] is None:
            self.dists["updated"] = gkde(
                self.lam.T, weights=self.state["ratio"] * self.state["weight"]
            )

        return self.dists["updated"]

    def init_state(self, lam, q_lam):
        """
        Initialize state dataframe
        """
        self.lam = set_shape(np.array(lam), (1, -1))
        self.q_lam = set_shape(np.array(q_lam), (-1, 1))
        self.state = pd.DataFrame(
            np.zeros((self.n_samples, self.n_params + self.n_states + 6)),
            columns=["weight", "pi_in", "pi_pr", "pi_obs", "ratio", "pi_up"]
            + [f"lam_{i}" for i in range(self.n_params)]
            + [f"q_lam_{i}" for i in range(self.n_states)],
        )
        self.state = put_df(self.state, "q_lam", self.q_lam, size=self.n_states)
        self.state = put_df(self.state, "lam", self.lam, size=self.n_params)

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

        self.state["weight"] = w

    def set_initial(
        self,
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
            self.dists["initial"] = gkde(
                self.lam.T, bw_method=bw_method, weights=self.state["weight"]
            )
        else:
            self.dists["initial"] = distribution

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
            distribution = gkde(
                self.q_lam.T, bw_method=bw_method, weights=self.state["weight"]
            )
        self.dists["predicted"] = distribution

    def _update(self):
        """
        Update Initial Distribution

        Constructs the updated distribution by fitting observed data to
        predicted data with:

        .. math::
            \\pi_{up}(\\lambda) = \\pi_{in}(\\lambda)
            \\frac{\\pi_{ob}(Q(\\lambda))}{\\pi_{pred}(Q(\\lambda))}
            :label: data_consistent_solution

        Note that if initial, predicted, and observed distributions have not
        been seti before running this method, they will be run with default
        values. To set specific predicted, observed, or initial distributions
        use the ``set_`` methods.

        Parameters
        -----------

        Returns
        -----------
        """
        if self.dists["initial"] is None:
            self.set_initial()
        if self.dists["predicted"] is None:
            self.set_predicted()

        self.state["pi_in"] = self.dists["initial"].pdf(self.lam.T).T
        self.state["pi_obs"] = self.dists["observed"].pdf(self.q_lam).prod(axis=1)
        self.state["pi_pr"] = self.dists["predicted"].pdf(self.q_lam.T).T.ravel()

        # Store ratio of observed/predicted
        # e.g. to comptue E(r) and to pass on to future iterations
        self.state["ratio"] = np.divide(self.state["pi_obs"], self.state["pi_pr"])

        # Multiply by initial to get updated pdf
        self.state["pi_up"] = np.multiply(
            self.state["pi_in"] * self.state["weight"], self.state["ratio"]
        )

    def solve(self):
        """ """
        self._update()

        results_cols = ["e_r", "kl"]
        results = np.zeros((1, 2))
        results[0, 0] = self.expected_ratio()
        results[0, 1] = self.divergence_kl()

        res_df = pd.DataFrame(results, columns=results_cols)

        self.result = res_df

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
        return np.average(self.state["ratio"], weights=self.state["weight"])

    def divergence_kl(self):
        """KL-Divergence Between observed and predicted.

        Parameters
        ----------

        Returns
        -------
        kl: float
            Value of the kl divergence.
        """
        return entropy(self.state["pi_obs"], self.state["pi_pr"])

    def sample_update(self, num_samples):
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
        return self.pi_up.resample(size=num_samples).T

    def plot_param_state(
        self,
        ax=None,
        param_idx=0,
        ratio_col="ratio",
        plot_initial=False,
        plot_legend=True,
        figsize=(8, 8),
    ):
        """
        Plotting functions for DCI Problem Class
        """
        sns.set_style("darkgrid")

        labels = []
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        bright_colors = sns.color_palette("bright", n_colors=self.n_params)
        # deep_colors = sns.color_palette("deep", n_colors=self.n_params)

        pi_up_label = f"$\pi^{{up}}_{{\lambda_{param_idx}}}$"
        sns.kdeplot(
            data=self.state,
            x=f"lam_{param_idx}",
            ax=ax,
            fill=True,
            color=bright_colors[param_idx],
            label=pi_up_label,
            weights=self.state["weight"] * self.state[ratio_col],
        )
        labels.append(pi_up_label)
        if plot_initial:
            pi_in_label = f"$\pi^{{in}}_{{\lambda_{param_idx}}}$"
            sns.kdeplot(
                data=self.state,
                x=f"lam_{param_idx}",
                ax=ax,
                fill=True,
                color=bright_colors[param_idx],
                linestyle=":",
                label=pi_in_label,
                weights="weight",
            )
            labels.append(pi_in_label)

        # Set plot specifications
        ax.set_xlabel(r"$\Lambda$", fontsize=12)
        if plot_legend:
            ax.legend(
                labels=labels,
                fontsize=12,
                title_fontsize=12,
            )

        plt.tight_layout()

        return ax, labels

    def plot_obs_state(
        self,
        ax=None,
        state_idx=0,
        plot_pf=True,
        plot_obs=True,
        plot_legend=True,
        obs_col="q_lam",
        ratio_col="ratio",
        figsize=(6, 6),
    ):
        """
        Plotting function for DCI Problem Class
        """
        sns.set_style("darkgrid")

        labels = []
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        bright_colors = sns.color_palette("bright", n_colors=self.n_states)
        # deep_colors = sns.color_palette("deep", n_colors=number_parameters)

        # Plot predicted distribution
        pr_label = "$\pi^{{pr}}_{{Q(\lambda)_{state_idx}}}$"
        sns.kdeplot(
            data=self.state,
            x=f"{obs_col}_{state_idx}",
            ax=ax,
            fill=True,
            color=bright_colors[state_idx],
            label=pr_label,
            weights=self.state["weight"],
        )
        labels.append(pr_label)
        if plot_pf:
            pf_label = f"$\pi^{{pf}}_{{Q(\lambda)_{state_idx}}}$"
            sns.kdeplot(
                data=self.state,
                x=f"{obs_col}_{state_idx}",
                ax=ax,
                fill=True,
                color=bright_colors[state_idx],
                linestyle=":",
                label=pf_label,
                weights=self.state["weight"] * self.state[ratio_col],
            )
            labels.append(pf_label)

        # TODO: How to plot this using SNS?
        if plot_obs:
            obs_label = "$\pi^{{obs}}_{{Q(\lambda)}}$"
            obs_domain = ax.get_xlim()
            obs_x = np.linspace(obs_domain[0], obs_domain[1], 10000)
            obs_x_marginal = np.zeros((len(obs_x), self.n_states))
            obs_x_marginal[:, state_idx] = obs_x
            obs = self.dists["observed"].pdf(obs_x_marginal)[:, state_idx]
            ax.plot(obs_x, obs, color="r", label=obs_label)
            labels.append(obs_label)

        # Set plot specifications
        ax.set_xlabel(r"$\mathcal{D}$", fontsize=12)
        if plot_legend:
            ax.legend(
                labels=labels,
                fontsize=12,
                title_fontsize=12,
            )
        plt.tight_layout()

        return ax, labels

    def state_plot(
        self, state="q_lam_0", mask=None, x_col=None, ax=None, figsize=(8, 8), **kwargs
    ):
        """
        Plot the X and Y data on two subplots, and add a rectangle for
        each interval to each subplot.
        """
        # Set up the figure and axes
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.color_palette("bright")

        sns.scatterplot(
            x=self.state.index if x_col is None else x_col,
            y=state,
            ax=ax,
            color="blue",
            data=self.state,
            label="State",
        )

        return ax

    def density_plots(
        self,
        true_vals=None,
        figsize=(14, 6),
    ):
        """
        Plot param and observable space onto sampe plot
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        self.plot_param_state(true_vals=true_vals, ax=axs[0])
        self.plot_obs_state(ax=axs[1])
        fig.suptitle(self._parse_title())
        fig.tight_layout()

        return axs

    def _parse_title(
        self,
    ):
        """
        Parse Title
        """
        kl = self.result["kl"].values[0]
        e_r = self.result["e_r"].values[0]
        title = f"$\mathbb{{E}}(r)$= {e_r:.3f}, " + f"$\mathcal{{D}}_{{KL}}$= {kl:.3f}"

        return title
