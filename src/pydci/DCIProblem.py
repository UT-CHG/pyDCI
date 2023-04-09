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
    Consistent Bayesian Solution to the Data Consistent Inversion Problem

    Solves a Data-Consistent Inversion Problem using a density based
    solution formulated first in [1]. Given an observed distribtuion on data
    `obs_dist`, the goal is to determine the distribution of input parameters
    that, when pushed through the forward model Q, generates an observed
    distribution that is consistent with the distribution on the data. The
    class takes a set of initial input samples lambda `lam`, and their
    associated values `q_lam` from pushing each sample through the  forward
    model. Optionall as well, a set of weights can be passed to incorporate
    previous beliefs on each sample in the set into the solution.

    Attributes
    ----------
    lam: ArrayLike
        Array containing parameter samples from an initial distribution.
        Rows represent each sample while columns represent parameter values.
        If 1 dimensional input is passed, assumed that it represents repeated
        samples of a 1-dimensional parameter.
    q_lam: ArrayLike
        2D array of values of each `lam` sample pushed through the forward
        model. Each row represent the value for each parameter, with each
        column being the value of observed QoI.
    pi_obs: rv_continuous
        scipy.stats continuous distribution object describing distribution on
        observed data that the consistent bayesian solution is trying to match.
    pi_in: rv_continuous, optional
        scipy.stats continuous distribution object describing distribution on
        initial samples, if known. If not, a gaussain kernel density estimate
        is done on the initial set of samples, `lam`,
    pi_pr: rv_continuous, optional
        scipy.stats continuous distribution object describing distribution on
        push forward of initial samples, if known analytically. If not, a
        gaussain kernel density estimate is done on the push forward of the
        initial set of samples, `q_lam`.
    weights : ArrayLike, optional
        Weights to apply to each parameter sample. Either a 1D array of the
        same length as number of samples or a 2D array if more than
        one set of weights is to be incorporated. If so the weights will be
        multiplied, so the number of columns must match the number of samples.

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
        pi_obs,
        pi_in = None,
        pi_pr = None,
        weights=None,
    ):
        self.init_prob(lam, q_lam, pi_obs, pi_in=pi_in, pi_pr=pi_pr)

    @property
    def n_params(self):
        """
        Number of Parameters
        """
        return self.lam.shape[1]

    @property
    def n_states(self):
        """
        Number of States
        """
        return self.q_lam.shape[1]

    @property
    def n_samples(self):
        """
        Number of Samples
        """
        return self.lam.shape[0]

    def init_prob(self,
                  lam,
                  q_lam,
                  pi_obs,
                  pi_in=None,
                  pi_pr=None,
                  weights=None):
        """
        Initialize problem

        Initialize problem by setting the lambda samples, the values of the
        samples pushed through the forward map, and the observe distribution
        on the data. Can optionally as well set the initial and predicteed
        distributions explicitly, and pass in weights to incorporate prior
        beliefs on the `lam` sample sets.
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
        self.set_weights(weights)
        if pi_pr is None:
            logger.info('Calculating pi_pr by computing KDE on samples')
            pr = gkde(
                self.q_lam.T, weights=self.state["weight"]
            )
        self.dists = {
            "pi_in": pi_in,
            "pi_pr": pi_pr,
            "pi_obs": pi_obs,
            "pi_up": None,
        }
        self.result = None

    def pi_in(self,
              values=None):
        """
        Evaluate the initial distribution.

        Init distribion is either set explicitly in by a call to `init_prob`
        or calculated from a gaussain kernel density estimate (using scipy) on
        the initial samples, weighted by the sample weights.
        """
        if self.dists['pi_in'] is None:
            logger.info('Calculating pi_in by computing KDE on lam')
            self.dists['pi_in'] = gkde(
                self.lam.T, weights=self.state["weight"]
            )
        values = self.lam if values is None else values
        if isinstance(self.dists['pi_in'], gkde):
            return self.dists["pi_in"].pdf(values.T).T
        else:
            return self.dists["pi_in"].pdf(values)

    def pi_pr(self,
              values=None):
        """
        Evaluate the predicted distribution.

        Predicted distribion is either set explicitly in the call to
        `init_prob` or calculated from a gaussain kernel density estimate
        (using scipy) on the push forward of the initial samples, q_lam,
        weighted by the sample weights.
        """
        if self.dists['pi_pr'] is None:
            logger.info('Calculating pi_pr by computing KDE on q_lam')
            self.dists['pi_pr'] = gkde(
                self.q_lam.T, weights=self.state["weight"]
            )
        values = self.q_lam if values is None else values
        if isinstance(self.dists['pi_pr'], gkde):
            return self.dists["pi_pr"].pdf(values.T).T.ravel()
        else:
            return self.dists["pi_pr"].pdf(values).prod(axis=1)

    def pi_obs(self,
               values=None):
        """
        Evaluate the observed distribution.

        Observed distribion is set explicitly in the call to `init_prob`.
        """
        values = self.q_lam if values is None else values
        if isinstance(self.dists['pi_obs'], gkde):
            return self.dists["pi_obs"].pdf(values.T).T.ravel()
        else:
            return self.dists["pi_obs"].pdf(values).prod(axis=1)

    def pi_up(self,
              values=None):
        """
        Evaluate Updated Distribution

        Computed using scipy's gaussian kernel density estimation on the
        initial samples, but weighted by the ratio of the updated and predicted
        distributions (evaluated at each sample value). Note, if the initial
        samples were weighted, then the weights are applied as well.
        """
        # Compute udpated density
        if self.dists["pi_up"] is None:
            self.dists["pi_up"] = gkde(
                self.lam.T, weights=self.state["ratio"] * self.state["weight"]
            )
        values = self.lam if values is None else values
        return self.dists["pi_up"].pdf(values.T).T

    def pi_pf(self,
              values=None):
        """
        Evaluate Updated Distribution

        Computed using scipy's gaussian kernel density estimation on the
        initial samples, but weighted by the ratio of the updated and predicted
        distributions (evaluated at each sample value). Note, if the initial
        samples were weighted, then the weights are applied as well.
        """
        # Compute udpated density
        if self.dists["pi_pf"] is None:
            self.dists["pi_pf"] = gkde(
                self.q_lam.T, weights=self.state["ratio"] * self.state["weight"]
            )
        values = self.q_lam if values is None else values
        return self.dists["pi_pf"].pdf(values.T).T

    def sample_dist(self, num_samples=1, dist='pi_up'):
        """
        Sample Stored Distribution

        Samples from stored distribtuion. By default samples from updated
        distribution on parameter samples, but also can draw samples from any
        stored distribtuion: pi_in, pi_pr, pi_obs, and pi_up.

        Parameters
        ----------
        dist: optional, default='pi_up'
            Distribution to samples from. By default sample from the update
            distribution
        num_samples: optional, default=1
            Number of samples to draw from distribtuion

        Returns
        -------
        samples: ArrayLike
            Samples from the udpated distribution. Dimension of array is
            (num_samples * num_params)
        """
        if isinstance(self.dists[dist], gkde):
            return self.self.dists[dist].resample(size=num_samples).T
        else:
            dim = self.n_params if dist == 'pi_in' else self.n_states
            return self.self.dists[dist].rvs((num_samples, dim)).T

    def set_weights(self, weights: ArrayLike = None):
        """
        Set Sample Weights

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

        self.state["weight"] = w

    def solve(self):
        """
        Solve the data consistent inverse problem by computing:

        .. math::
            \\pi_{up}(\\lambda) = \\pi_{in}(\\lambda)
            \\frac{\\pi_{ob}(Q(\\lambda))}{\\pi_{pred}(Q(\\lambda))}
            :label: data_consistent_solution

        """
        self.state["pi_in"] = self.pi_in()
        self.state["pi_obs"] = self.pi_obs()
        self.state["pi_pr"] = self.pi_pr()
        self.state["ratio"] = np.divide(self.state["pi_obs"],
                                        self.state["pi_pr"])
        self.state["pi_up"] = np.multiply(
            self.state["pi_in"] * self.state["weight"], self.state["ratio"]
        )

        # Store result into result dataframe
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

    def plot_param_state(
        self,
        ax=None,
        param_idx=0,
        ratio_col="ratio",
        plot_initial=True,
        plot_legend=True,
        figsize=(6, 6),
    ):
        """
        Plot distributions over parameter space. This includes the initial and
        the updated distributinos.
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
        Plot distributions over observable space `q_lam`. This includes the
        observed distribution `pi_obs`, the predicted distribtuion `pi_pr`, and
        the push-forward of the updated distribution `pi_pf`.
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
            obs = self.pi_obs(values=obs_x_marginal)
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
        figsize=(14, 6),
    ):
        """
        Plot param and observable space onto sampe plot
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        self.plot_param_state(ax=axs[0])
        self.plot_obs_state(ax=axs[1])
        fig.suptitle(self._parse_title())
        fig.tight_layout()

        return axs

    def _parse_title(
        self,
    ):
        """
        Parse title for plots
        """
        kl = self.result["kl"].values[0]
        e_r = self.result["e_r"].values[0]
        title = f"$\mathbb{{E}}(r)$= {e_r:.3f}, " + f"$\mathcal{{D}}_{{KL}}$= {kl:.3f}"

        return title

