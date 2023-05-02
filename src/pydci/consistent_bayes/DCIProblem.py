"""
Consistent Bayesian Formulation for Data-Consistent Inversion

The classes in this module all derive off of the Consistent-Bayesian formulation
for solving Stochastic Inverse problems first proposed in [1]. The classes all
inherit from the base class, `DCIProblem` and all have the following structure
in general in terms of how they are to be used:

1. Initialization: Upon initailization the state of the system is set,
including parameter samples, their values evaluated through the forward model,
and data/assumptions on observed data. This is used to initailiaze a pandas
DataFrame `state`, that stores these values and is used for computing and
storing final solutions
2. Solving: solve() -> Main method called to solve the problem class. Specific
parameters controlling how the algorithm is solved can be set here. The results
of the solve are store in the `result` attribute of the class.
3. Visualizing: plot_L(), plot_D(), plot_dists() -> Plot resulting
distributions from solving the problem.

References
----------
[1] T. Butler, J. Jakeman, and T. Wildey, “Combining Push-Forward Measures
and Bayes’ Rule to Construct Consistent Solutions to Stochastic Inverse
Problems,” SIAM J. Sci. Comput., vol. 40, no. 2, pp. A984–A1011, Jan. 2018,
doi: 10.1137/16M1087229.

TODO List:

    - Mud point for inherited classes not being plotted correctly
    - Sequential (should be renamed split sequential?) - Plot distribution
    methods for plotting distributions per iteration, qoi combinations,
    or pca values
    Dynamic Problem -> Finish

"""
import itertools
import pdb
import random
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from rich.table import Table
from scipy.stats import distributions as dist  # type: ignore
from scipy.stats import rv_continuous  # type: ignore
from scipy.stats import entropy
from scipy.stats import gaussian_kde as gkde  # type: ignore
from scipy.stats.distributions import norm
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from pydci.log import disable_log, enable_log, log_table, logger
from pydci.utils import fit_domain, get_df, put_df, set_shape

sns.color_palette("bright")
sns.set_style("darkgrid")

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"


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
    samples: Union[Tuple[ArrayLike, ArrayLike], pandas.DataFrame]
        Either (1) Tuple of arrays `(lam, q_lam)`, with the `lam` array being
        of dimensions `(num_samples, num_params)` and the `q_lam` array being
        of dimensions `(num_samples, num_states)` or (2) a pandas DataFrame
        containg columns prefixed `lam_` for each parameter dimension and
        prefixed `q_lam_` for each osberved state dimension, and each row
        corresponds to a a parameter sample.
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
        samples,
        pi_obs,
        pi_in=None,
        pi_pr=None,
        weights=None,
    ):
        self.init_prob(samples, pi_obs, pi_in=pi_in, pi_pr=pi_pr)

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

    def init_prob(self, samples, pi_obs, pi_in=None, pi_pr=None, weights=None):
        """
        Initialize problem

        Initialize problem by setting the lambda samples, the values of the
        samples pushed through the forward map, and the observe distribution
        on the data. Can optionally as well set the initial and predicteed
        distributions explicitly, and pass in weights to incorporate prior
        beliefs on the `lam` sample sets.
        """
        if isinstance(samples, pd.DataFrame):
            cols = samples.columns
            n_params = len([x for x in cols if x.startswith("l")])
            n_states = len(cols) - n_params
            self.lam = get_df(samples, "lam", size=n_params)
            self.q_lam = get_df(samples, "q_lam", size=n_states)
        else:
            self.lam = set_shape(np.array(samples[0]), (1, -1))
            self.q_lam = set_shape(np.array(samples[1]), (-1, 1))
        self.state = pd.DataFrame(
            np.zeros((self.n_samples, 6)),
            columns=["weight", "pi_in", "pi_pr", "pi_obs", "ratio", "pi_up"],
        )
        self.state = put_df(self.state, "q_lam", self.q_lam, size=self.n_states)
        self.state = put_df(self.state, "lam", self.lam, size=self.n_params)
        self.dists = {
            "pi_in": pi_in,
            "pi_pr": pi_pr,
            "pi_obs": pi_obs,
            "pi_up": None,
        }
        self.set_weights(weights)
        self.result = None

    def pi_in(self, values=None):
        """
        Evaluate the initial distribution.

        Init distribion is either set explicitly in by a call to `init_prob`
        or calculated from a gaussain kernel density estimate (using scipy) on
        the initial samples, weighted by the sample weights.
        """
        if self.dists["pi_in"] is None:
            logger.info("Calculating pi_in by computing KDE on lam")
            self.dists["pi_in"] = gkde(self.lam.T, weights=self.state["weight"])
        values = self.lam if values is None else values
        if isinstance(self.dists["pi_in"], gkde):
            return self.dists["pi_in"].pdf(values.T).T
        else:
            return self.dists["pi_in"].pdf(values)

    def pi_pr(self, values=None):
        """
        Evaluate the predicted distribution.

        Predicted distribion is either set explicitly in the call to
        `init_prob` or calculated from a gaussain kernel density estimate
        (using scipy) on the push forward of the initial samples, q_lam,
        weighted by the sample weights.
        """
        if self.dists["pi_pr"] is None:
            logger.info("Calculating pi_pr by computing KDE on q_lam")
            self.dists["pi_pr"] = gkde(self.q_lam.T, weights=self.state["weight"])
        values = self.q_lam if values is None else values
        if isinstance(self.dists["pi_pr"], gkde):
            return self.dists["pi_pr"].pdf(values.T).T.ravel()
        else:
            return self.dists["pi_pr"].pdf(values).prod(axis=1)

    def pi_obs(self, values=None):
        """
        Evaluate the observed distribution.

        Observed distribion is set explicitly in the call to `init_prob`.
        """
        values = self.q_lam if values is None else values
        if isinstance(self.dists["pi_obs"], gkde):
            return self.dists["pi_obs"].pdf(values.T).T.ravel()
        else:
            return self.dists["pi_obs"].pdf(values).prod(axis=1)

    def pi_up(self, values=None):
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

    def pi_pf(self, values=None):
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

    def sample_dist(self, num_samples=1, dist="pi_up"):
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
            return self.dists[dist].resample(size=num_samples).T
        else:
            dim = self.n_params if dist in ["pi_in", "pi_up"] else self.n_states
            if self.dists[dist] is None:
                _ = getattr(self, dist, None)()
            if not isinstance(self.dists[dist], gkde):
                return self.dists[dist].rvs((num_samples, dim)).T
            else:
                return self.dists[dist].resample(num_samples).T

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
        if weights is None or len(weights) == 0:
            w = np.ones(self.n_samples)
        else:
            w = np.array(weights)

            # Reshape to 2D
            w = w.reshape(1, -1) if w.ndim == 1 else w

            # assert appropriate size
            assert self.n_samples == w.shape[1], f"`weights` must size {self.n_samples}"

            # Multiply weights column wise for stacked weights
            w = np.prod(w, axis=0)

        self.state['weight'] = w
        self.dists['pi_in'] = None

    def solve(self):
        """
        Solve the data consistent inverse problem by computing `pi_up` as the
        a multiplicative update to `pi_in`:

        `s['pi_up'] = s['weights'] * s['pi_in'] (s['pi_obs'] / s['pi_pr'])`

        Quantities for each density evaluated at each parameter sample/push
        forward value are stored in the `state` attribute DataFrame, along
        with the ratio of the observed (`pi_obs`) to predicted (`pi_pr`), for
        ease of access later.
        """
        self.state["pi_in"] = self.pi_in()
        self.state["pi_obs"] = self.pi_obs()
        self.state["pi_pr"] = self.pi_pr()
        self.state["ratio"] = np.divide(self.state["pi_obs"], self.state["pi_pr"])
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

        If the predictability assumption for the data-consistent framework is
        satisfied, then this values should be close to 1.0.

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

    def plot_L(
        self,
        df=None,
        param_idx=0,
        param_col="lam",
        ratio_col="ratio",
        weight_col="weight",
        plot_initial=True,
        plot_legend=True,
        ax=None,
        figsize=(6, 6),
    ):
        """
        Plot Lambda Space Distributions

        Plot distributions over parameter space. This includes the initial and
        the updated distributions.

        Parameters
        ----------
        df: pd.DataFrame, default=None
            Dataframe to use for accessing data. Defaults to the classes's
            `self.state` DataFrame. Can be used by sub-classes that store past
            states for plotting them.
        param_idx : int, default=0
            Index of parameter, `lam` to plot.
        param_col: str, default='lam'
            Column in DataFrame storing the parameter values to use.
        ratio_col : str, default='ratio'
            Column in DataFrame storing the `ratio` to use.
        plot_initial : bool, default=True
            Whether to include the initial distribution `pi_in` in the plot.
        plot_legend: bool, default=True
            Whether to include a labeled legend in the plot. Note, labels of
            what is plotted are returned along with the axis object for
            sub-classes to modify the legend as necessary.
        ax: matplotlib.pyplot.axes, default=None
           Axis to plot onto. If none provided, figure will be created.
        figsize: Tuple, default=(6, 6)
           If no axis to plot on is specified, then figure created will be of
           this size.

        Returns
        -------
        ax, labels : Tuple
            Tuple of (1) matplotlib axis object where distributions where
            plotted and (2) List of labels that were plotted, in order plotted.
        """
        df = self.state if df is None else df

        labels = []
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        bright_colors = sns.color_palette("bright", n_colors=self.n_params)
        # deep_colors = sns.color_palette("deep", n_colors=self.n_params)

        pi_up_label = f"$\pi^{{up}}_{{\lambda_{param_idx}}}$"
        sns.kdeplot(
            data=df,
            x=f"{param_col}_{param_idx}",
            ax=ax,
            fill=True,
            color=bright_colors[param_idx],
            label=pi_up_label,
            weights=df[weight_col] * df[ratio_col],
        )
        labels.append(pi_up_label)
        if plot_initial:
            pi_in_label = f"$\pi^{{in}}_{{\lambda_{param_idx}}}$"
            sns.kdeplot(
                data=df,
                x=f"{param_col}_{param_idx}",
                ax=ax,
                fill=True,
                color=bright_colors[param_idx],
                linestyle=":",
                label=pi_in_label,
                weights=weight_col,
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

    def plot_D(
        self,
        df=None,
        state_idx=0,
        state_col="q_lam",
        ratio_col="ratio",
        weight_col="weight",
        plot_obs=True,
        plot_pf=True,
        plot_legend=True,
        ax=None,
        figsize=(6, 6),
    ):
        """
        Plot Q(lambda) = D Space Distributions

        Plot distributions over observable space `q_lam`. This includes the
        observed distribution `pi_obs`, the predicted distribtuion `pi_pr`, and
        the push-forward of the updated distribution `pi_pf`.

        Parameters
        ----------
        df: pd.DataFrame, default=None
            Dataframe to use for accessing data. Defaults to the classes's
            `self.state` DataFrame. Can be used by sub-classes that store past
            states for plotting them.
        state_idx: int, default=0
            Index of state, `q_lam`, to plot.
        state_col: str, default='q_lam'
            Column in DataFrame storing the state values to use.
        ratio_col : str, default='ratio'
            Column in DataFrame storing the `ratio` to use in the state
            DataFrame.
        plot_obs: bool, default=True
            Whether to include the observed distribution `pi_obs` in the plot.
        plot_obs: bool, default=True
            Whether to include the push-forwardo of the updated distribution,
            `pi_pf`, in the plot.
        plot_legend: bool, default=True
            Whether to include a labeled legend in the plot. Note, labels of
            what is plotted are returned along with the axis object for
            sub-classes to modify the legend as necessary.
        ax: matplotlib.pyplot.axes, default=None
           Axis to plot onto. If none provided, figure will be created.
        figsize: Tuple, default=(6, 6)
           If no axis to plot on is specified, then figure created will be of
           this size.

        Returns
        -------
        ax, labels : Tuple
            Tuple of (1) matplotlib axis object where distributions where
            plotted and (2) List of labels that were plotted, in order plotted.
        """
        df = self.state if df is None else df

        labels = []
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        bright_colors = sns.color_palette("bright", n_colors=self.n_states)
        # deep_colors = sns.color_palette("deep", n_colors=number_parameters)

        # Plot predicted distribution
        pr_label = "$\pi^{{pr}}_{{Q(\lambda)_{state_idx}}}$"
        sns.kdeplot(
            data=self.state,
            x=f"{state_col}_{state_idx}",
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
                x=f"{state_col}_{state_idx}",
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

    def plot_sample(
        self,
        sample_idx=0,
        qoi_mask=None,
        ax=None,
        reshape=(-1, 1),
        figsize=(8, 8),
        plot_type="scatter",
        label=True,
        **kwargs,
    ):
        """
        Plot the X and Y data on two subplots, and add a rectangle for
        each interval to each subplot.
        """
        # Set up the figure and axes
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.color_palette("bright")

        cols = [f"q_lam_{i}" for i in range(self.n_states)]
        qoi_mask = np.arange(self.n_states) if qoi_mask is None else qoi_mask

        if isinstance(sample_idx, int):
            sample_idx = [sample_idx]
        elif not isinstance(sample_idx, list):
            raise ValueError("Sample idx must be an integer or integer list")
        for si in sample_idx:
            sample = np.array(self.state[cols].loc[si]).reshape(reshape)[:, qoi_mask]
            sample_df = pd.DataFrame(
                np.array([qoi_mask, sample[0]]).reshape(len(qoi_mask), 2),
                columns=["i", "q_lam_i"],
            )

            lab = None if not label else f"Sample {si} State"
            if plot_type == "scatter":
                sns.scatterplot(
                    x="i",
                    y="q_lam_i",
                    ax=ax,
                    color="blue",
                    data=sample_df,
                    label=lab,
                )
            else:
                sns.lineplot(
                    x="i",
                    y="q_lam_i",
                    ax=ax,
                    color="blue",
                    data=sample_df,
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
        self.plot_L(ax=axs[0])
        self.plot_D(ax=axs[1])
        fig.suptitle(self._parse_title())
        fig.tight_layout()

        return axs

    def param_density_plots(
        self,
        lam_true=None,
        base_size=4,
        max_np=9,
    ):
        base_size = 4
        n_params = self.n_params if self.n_params <= max_np else max_np
        grid_plot = self._closest_factors(n_params)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, ax in enumerate(ax.flat):
            self.plot_L(param_idx=i, lam_true=lam_true, ax=ax)

        fig.suptitle(self._parse_title(lam_true=lam_true))
        fig.tight_layout()

    def _parse_title(
        self,
        result=None,
    ):
        """
        Parse title for plots
        """
        result = self.result if result is None else result
        kl = result["kl"].values[0]
        e_r = result["e_r"].values[0]
        title = f"$\mathbb{{E}}(r)$= {e_r:.3f}, " + f"$\mathcal{{D}}_{{KL}}$= {kl:.3f}"

        return title

    def _closest_factors(self, n):
        for i in range(int(n**0.5), 0, -1):
            if n % i == 0:
                return (i, n // i)


class MUDProblem(DCIProblem):
    """
    Maximal Updated Density Problem

    Maxmal Update Density Inverse problem class for parameter identification.
    This extends the DCIProblem class by computing the Maximal Updated Density,
    or MUD, point, as the parameter sample that maximizes the updated
    distribution in order to solve a parmater estimation problem, as first
    proposed in [1].

    The key distinction is the assumptions being made in the data between a
    parameter estimation problem and a general Data Consistent Inversion
    problem. In a MUDProblem, we assume that the source of the error in the
    parameter samples that we want to quantify is epistemic in nature,
    and thus our data come from a true parameter value, that when pushed forward
    through our QoI map is populated with gaussian error. The goal is then to
    determine the true value of the parameter that produced the observed data.
    Note how this is different from the general DCIProblem, where instead of
    quantifying the probability distribution of the parameter itself, the
    solution is a point that maximizes the distribution, and not the
    distribution itself.

    This class extends the DCIProblem class in the following ways:

        1. Initialization - Instead of receiving an observed distribution on
        data as the input, the observed data itself should be passed, along
        with the standard deviation associated with the i.i.d. Gaussian noise
        the data is assumed to be populate with. The observe distribution is
        set by default to a Gaussian Distribution with mean around the man of
        the data and standard deviation equal to the passed in `std_dev`.
        2. `solve()` - Solve method is extended by computing the mud point and
        storing it in the attribute `mud_point`. Note this is calculated as the
        sample that has the maximum `pi_up` value in the classes's `state`
        DataFrame, NOT the maximum value as determined by the kde computed on
        these values.
        3. Plotting - Plots add vertical lines for MUD points on the parameter
        distribution plots, and options for plotting the true value if known.

    Note: this class does no data-aggregation using data-constructed QoI maps
    as proposed in [1] for parameter estimation. See sub-classes `WMEMUDProblem`
    and `PCAMUDProblem` for classes that use data-constructed QoI maps for
    parameter estimation.

    Attributes
    ----------
    data : ArrayLike
        Observed data. Should be of same dimension as the number of observed
        states for each passsed in sample, `q_lam`.
    std_dev : float
        Assumed measurement noise in collecting the data.

    References
    ----------
    [1] M. Pilosov, C. del-Castillo-Negrete, T. Y. Yen, T. Butler, and C.
    Dawson, “Parameter estimation with maximal updated densities,” Computer
    Methods in Applied Mechanics and Engineering, vol. 407, p. 115906, Mar.
    2023, doi: 10.1016/j.cma.2023.115906.
    """

    def __init__(
        self,
        samples,
        data,
        std_dev,
        pi_in=None,
        pi_pr=None,
        weights=None,
    ):
        self.init_prob(
            samples, data, std_dev, pi_in=pi_in, pi_pr=pi_pr, weights=weights
        )

    def init_prob(self, samples, data, std_dev, pi_in=None, pi_pr=None, weights=None):
        """
        Initialize problem

        Initialize problem by setting the lambda samples, the values of the
        samples pushed through the forward map, and the observe distribution
        on the data. Can optionally as well set the initial and predicteed
        distributions explicitly, and pass in weights to incorporate prior
        beliefs on the `lam` sample sets.
        """
        # Assume gaussian error around mean of data with assumed noise
        self.std_dev = std_dev
        self.data = set_shape(np.array(data), (-1, 1))
        pi_obs = dist.norm(loc=np.mean(data), scale=std_dev)
        super().init_prob(samples, pi_obs, pi_in=pi_in, pi_pr=pi_pr, weights=weights)
        self.mud_point = None

    def solve(self):
        """
        Solve MUD Parameter Estimation Problem

        Extends the parent method by computing the MUD point, the solution
        to the parameter estimation problem, as the samples that maximizes the
        `pi_up` column in the state DataFrame. This MUD Point is stored in the
        results DataFrame that is returned.
        """
        super().solve()
        m = np.argmax(self.state["pi_up"])
        mud_point = get_df(self.state.loc[[m]], "lam", size=self.n_params)
        self.result = put_df(self.result, "lam_MUD", mud_point, size=self.n_params)
        self.mud_point = mud_point[0]
        self.mud_arg = m

    def plot_L(
        self,
        lam_true=None,
        mud_point=None,
        df=None,
        param_idx=0,
        param_col="lam",
        ratio_col="ratio",
        weight_col="weight",
        plot_initial=True,
        plot_legend=True,
        ax=None,
        figsize=(6, 6),
    ):
        """
        Plot Lambda Space Distributions

        Plot distributions over parameter space. This includes the initial and
        the updated distributions. Extends `DCIProblem` methods by adding a
        vertical line for the MUD point and an optional line for the true
        solution if passed in. See documentation for `DCIProblem.plot_L` for
        more info on additional arguments

        Parameters
        ----------
        param_idx : int, default=0
            Index of parameter, `lam` to plot.
        lam_true: ArrayLike, default=None
            If specified, a vertical line for the true parameter solution will
            be added. Note this value must be the same dimension as the
            parameter space, even if it only the value at the `param_idx`
            specified is only used.
        plot_mud: bool, default=True
            Whether to add a vertical line for the computed MUD point solution.

        Returns
        -------
        ax, labels : Tuple
            Tuple of (1) matplotlib axis object where distributions where
            plotted and (2) List of labels that were plotted, in order plotted.
        """
        ax, labels = super().plot_L(
            df=df,
            param_idx=param_idx,
            param_col=param_col,
            ratio_col=ratio_col,
            weight_col=weight_col,
            plot_initial=plot_initial,
            plot_legend=plot_legend,
            ax=ax,
            figsize=figsize,
        )

        # Generate vertical lines for true values
        if lam_true is not None:
            lam_true_label = (
                f"$\lambda^{{\dagger}}_{param_idx} = "
                + f"{lam_true[0][param_idx]:.4f}$"
            )
            ax.axvline(
                x=lam_true[0][param_idx],
                linewidth=3,
                color="orange",
                label=lam_true_label,
            )
            labels.append(lam_true_label)

        mud_point = self.mud_point if mud_point is None else mud_point
        mud_label = f"$\lambda^{{MUD}}_{param_idx} = " + f"{mud_point[param_idx]:.4f}$"
        ax.axvline(
            x=mud_point[param_idx],
            linewidth=3,
            color="green",
            linestyle="--",
            label=mud_label,
        )
        labels.append(mud_label)

        if plot_legend:
            ax.legend(
                labels=labels,
                fontsize=12,
                title_fontsize=12,
            )

        return ax, labels

    def density_plots(
        self,
        lam_true=None,
        lam_kwargs=None,
        q_lam_kwargs=None,
        figsize=(14, 6),
    ):
        """
        Plot param and observable space onto sampe plot
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        lam_kwargs = {} if lam_kwargs is None else lam_kwargs
        q_lam_kwargs = {} if q_lam_kwargs is None else q_lam_kwargs
        lam_kwargs["ax"] = axs[0]
        q_lam_kwargs["ax"] = axs[1]
        self.plot_L(**lam_kwargs)
        self.plot_D(**q_lam_kwargs)
        lam_true = lam_kwargs.get("lam_true", None)
        fig.suptitle(self._parse_title(lam_true=lam_true))
        fig.tight_layout()

        return axs

    def _parse_title(
        self,
        result=None,
        lam_true=None,
    ):
        """
        Parse title for plots

        Extends DCIProblem _parse title by adding MUD point.
        Note sets result to parse title for to result set by class if none
        passed by call (for calls from sub-classes).
        """
        result = self.result if result is None else result
        title = super()._parse_title(result=result)
        if lam_true is not None:
            mud_point = get_df(result, "lam_MUD", size=self.n_params)[0]
            l2_err = np.linalg.norm(lam_true - self.mud_point)
            title = (
                "$||\lambda^{{\dagger}} - \lambda^{{MUD}}||_{{\ell_2}}$"
                + f" = {l2_err:.3f},  "
                + title
            )

        return title


class PCAMUDProblem(MUDProblem):
    """
    PCA MUD Problem

    Sets up a Maximal Updated Density (MUD) parameter estimation using the
    `q_pca` map to aggregate data as proposed in [1]. The `q_pca` map is a way
    of aggregating observed data with simulated data for parameter estimation
    problems using the Data Consistent Inversion (DCI) framework. By inverting
    on a map of aggregated data instead of the map that produced the data
    itself, the variance in the MUD parameter estimate can be reduced as more
    data is incorporated.

    This class extends the MUDProblem class by using the `q_pca()` function
    before solving the parameter estimation problem to aggregate data and invert
    on the data-constructed map instead.

    Attributes
    ----------
    pca_res : List[pd.DataFrame]

    Methods
    -------
    solve(pca_mask=None, max_nc=None, best_method="closest", exp_thresh=0.5)
        Solve the parameter estimation problem, with the parameters relevant
        to aggregating the data into the `q_pca()` map and determing how many
        principal components to use for optimal solution.

    """

    def __init__(
        self,
        samples,
        data,
        std_dev,
        pi_in=None,
        weights=None,
    ):
        self.init_prob(samples, data, std_dev, pi_in=pi_in, weights=weights)

    @property
    def n_qoi(self):
        return self.qoi.shape[1]

    def init_prob(self, samples, data, std_dev, pi_in=None, weights=None):
        """
        Initialize problem

        Initialize problem by setting the lambda samples, the values of the
        samples pushed through the forward map, and the observe distribution
        on the data. Can optionally as well set the initial and predicteed
        distributions explicitly, and pass in weights to incorporate prior
        beliefs on the `lam` sample sets.
        """
        # Assume gaussian error around mean of data with assumed noise
        super().init_prob(
            samples, data, std_dev, pi_in=pi_in, pi_pr=None, weights=weights
        )
        self.qoi = self.q_lam
        self.pca_states = None
        self.pca_results = None

    def q_pca(self, mask=None, max_nc=None):
        """
        Build QoI Map Using Data and Measurements

        Aggregate q_lam data with observed data for MUD convergence.
        """
        mask = np.arange(self.n_qoi) if mask is None else mask
        residuals = np.subtract(self.data[mask].T, self.qoi[:, mask]) / self.std_dev
        if max_nc is None:
            max_nc = self.n_params if self.n_params < len(mask) else len(mask)

        # Standarize and perform linear PCA
        sc = StandardScaler()
        pca = PCA(n_components=max_nc)
        X_train = pca.fit_transform(sc.fit_transform(residuals))
        self.pca = {
            "X_train": X_train,
            "vecs": pca.components_,
            "var": pca.explained_variance_,
        }

        # Compute Q_PCA
        self.q_lam = residuals @ pca.components_.T
        self.state = put_df(self.state, "q_pca", self.q_lam, size=max_nc)

    def solve(
        self,
        pca_mask: List[int] = None,
        max_nc: int = None,
        exp_thresh: float = 0.5,
        best_method: str = "closest",
    ):
        """
        Solve the parameter estimation problem

        This extends the `MUDProblem` solution class by using the `q_pca()` map
        to aggregate data between the observed and predicted values and
        determine the best MUD estimate that fits the data.

        Parameters
        ----------
        pca_mask: List[int], default=None
            Used control what subset of the observed data is used in the data
            constructed map `q_pca()`
        max_nc: int, default=None
            Specifies the max number of principal components to use when doing
            the PCA transformation on the residuals between the observed and
            simulated data. If not specified, defaults to the min of the number
            of states and the number of parameters.
        exp_thresh: float, default=0.5
            Threshold to accept a solution to the MUD problem as a valid
            solution. Any solution more than `exp_thresh` away from 1.0 will
            be deemed as violating the predictability assumption and not valid.
        best_method: str, default="closest"
            One of "closest", "min_k", or "max_kl", this specifies which
            solution should be deemed the "best". Closest is for the MUD
            estimate corresponding to the E(r) value closest to 1, while min/max
            kl specify the MUD estimate with the corresponding min/max KL
            divergence, indicating the least/most informative update. Note this
            is of estimates that are within `exp_thresh` of E(r). If none are,
            then no solution will be returned.
        """
        am = ["closest", "min_kl", "max_kl"]
        if best_method not in am:
            msg = f"Unrecognized best method {best_method}. Allowed: {am}"
            logger.error(msg)
            raise ValueError(msg)
        if exp_thresh <= 0:
            msg = f"Expected ratio thresh must be a float > 0: {exp_thresh}"
            logger.error(msg)
            raise ValueError(msg)
        self.exp_thresh = exp_thresh
        self.best_method = best_method

        self.q_pca(mask=pca_mask, max_nc=max_nc)
        all_qoi = self.q_lam
        results_cols = (
            ["nc"] + [f"lam_MUD_{i}" for i in range(self.n_params)] + ["e_r", "kl"]
        )
        results = np.zeros((len(self.pca["vecs"]), self.n_params + 3))
        results = []
        dists = []
        nc_list = range(1, len(self.pca["vecs"]) + 1)
        pca_states = []
        for nc in nc_list:
            logger.info(f"Solving using {nc} components")
            self.q_lam = all_qoi[:, 0:nc]
            self.dists["pi_obs"] = dist.norm(loc=nc * [0], scale=1)
            self.dists["pi_pr"] = None
            try:
                super().solve()
            except ValueError as v:
                if "array must not contain infs or NaNs" in str(v):
                    logger.error(f"Solve with {nc} components failed")
                    continue
                else:
                    raise v
            self.result["nc"] = nc
            results.append(self.result.set_index("nc"))
            pca_state = self.state[
                ["weight", "pi_obs", "pi_pr", "ratio", "pi_up"]
            ].copy()
            pca_state["nc"] = nc
            pca_states.append(
                pca_state[["nc", "weight", "pi_obs", "pi_pr", "ratio", "pi_up"]]
            )
            dists.append(self.dists)

        # Parse DataFrame with results of mud estimations for each ts choice
        res_df = pd.concat(results)  # , keys=nc_list, names=['nc'])
        res_df["predict_delta"] = np.abs(res_df["e_r"] - 1.0)
        res_df["within_thresh"] = res_df["predict_delta"] <= self.exp_thresh
        res_df["closest"] = np.logical_and(
            res_df["predict_delta"]
            <= res_df[res_df["within_thresh"]]["predict_delta"].min(),
            res_df["within_thresh"],
        )
        res_df["max_kl"] = np.logical_and(
            res_df["kl"] >= res_df[res_df["within_thresh"]]["kl"].max(),
            res_df["within_thresh"],
        )
        res_df["min_kl"] = np.logical_and(
            res_df["kl"] <= res_df[res_df["within_thresh"]]["kl"].min(),
            res_df["within_thresh"],
        )
        best_nc = res_df[self.best_method].idxmax()
        self.q_lam = all_qoi[:, 0:best_nc]
        self.dists["pi_obs"] = dist.norm(loc=best_nc * [0], scale=1)
        self.dists["pi_pr"] = None
        super().solve()
        # self.state = self.state.join(pca_states)
        self.pca_states = pd.concat(pca_states, axis=0)
        self.pca_results = res_df
        self.result = res_df.loc[[best_nc]]

    def plot_L(
        self,
        nc=None,
        lam_true=None,
        mud_point=None,
        df=None,
        param_idx=0,
        param_col="lam",
        ratio_col="ratio",
        weight_col="weight",
        plot_initial=True,
        plot_legend=True,
        ax=None,
        figsize=(6, 6),
    ):
        """
        Plot Lambda Space Distributions

        Plot distributions over parameter space. This includes the initial and
        the updated distributions. Extends `MUDProblem` methods by allowing
        an `nc` argument to plot the solutions that use only `nc` number of
        principal components in the `q_pca` map. If `nc` is not specified, the
        best choice of `nc` will be picked from those tried, with the best
        being chosen according to the `best_method` set in `solve()`.

        Parameters
        ----------
        param_idx : int, default=0
            Index of parameter, `lam` to plot.
        lam_true: ArrayLike, default=None
            If specified, a vertical line for the true parameter solution will
            be added. Note this value must be the same dimension as the
            parameter space, even if it only the value at the `param_idx`
            specified is only used.
        plot_mud: bool, default=True
            Whether to add a vertical line for the computed MUD point solution.

        Returns
        -------
        ax, labels : Tuple
            Tuple of (1) matplotlib axis object where distributions where
            plotted and (2) List of labels that were plotted, in order plotted.
        """
        if df is None:
            df_index = self.result.index.values[0]
            nc = df_index if nc is None else nc
            mud_point = get_df(self.pca_results.loc[[nc]], "lam_MUD", self.n_params)[0]
            df = self.state.join(
                self.pca_states[self.pca_states["nc"] == nc][["ratio"]].add_suffix(
                    f"_plot"
                )
            )
            ratio_col = "ratio_plot"

        ax, labels = super().plot_L(
            lam_true=lam_true,
            mud_point=mud_point,
            df=df,
            param_idx=param_idx,
            param_col=param_col,
            ratio_col=ratio_col,
            weight_col=weight_col,
            plot_initial=plot_initial,
            plot_legend=plot_legend,
            ax=ax,
            figsize=figsize,
        )

        return ax, labels

    def plot_D(
        self,
        nc=None,
        df=None,
        state_idx=0,
        state_col="q_pca",
        ratio_col="ratio",
        weight_col="weight",
        plot_obs=True,
        plot_pf=True,
        plot_legend=True,
        ax=None,
        figsize=(6, 6),
    ):
        """
        Plot Q(lambda) Space Distributions

        Plot distributions over observable space `q_lam`. This includes the
        observed distribution `pi_obs`, the predicted distribtuion `pi_pr`, and
        the push-forward of the updated distribution `pi_pf`. Extends
        `MUDProblem` methods by allowing an `nc` argument to plot the solutions
        that use only `nc` number of principal components in the `q_pca` map.
        If `nc` is not specified, the best choice of `nc` will be picked from
        those tried, with the best being chosen according to the `best_method`
        set in `solve()` (and corresponding to the solution stored in the
        `result` class attribute). See parent class for more info on relevant
        parameters.

        Parameters
        ----------
        nc: int, default=None
            Plot solution that only uses `nc` components. By default will plot
            the solution that corresponds to the best as determined in the
            `solve()` method and stored in the `result` attribute.

        Returns
        -------
        ax, labels : Tuple
            Tuple of (1) matplotlib axis object where distributions where
            plotted and (2) List of labels that were plotted, in order plotted.
        """
        if df is None:
            df = self.state
            if nc is not None:
                if nc > len(self.pca_results):
                    msg = f"{nc} greater than max number of components used"
                    logger.error(msg)
                    raise ValueError(msg)
                df = self.state.join(
                    self.pca_states[self.pca_states["nc"] == nc][["ratio"]].add_suffix(
                        f"_nc={nc}"
                    )
                )
                ratio_col = f"ratio_nc={nc}"
        ax, labels = super().plot_D(
            df=df,
            state_idx=state_idx,
            state_col=state_col,
            ratio_col=ratio_col,
            weight_col=weight_col,
            plot_obs=plot_obs,
            plot_pf=plot_pf,
            plot_legend=plot_legend,
            ax=ax,
            figsize=figsize,
        )

        return ax, labels

    def density_plots(
        self,
        idx=None,
        lam_true=None,
        lam_kwargs=None,
        q_lam_kwargs=None,
        figsize=(14, 6),
    ):
        """
        Plot param and observable space onto sampe plot
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        lam_kwargs = {} if lam_kwargs is None else lam_kwargs
        q_lam_kwargs = {} if q_lam_kwargs is None else q_lam_kwargs
        lam_kwargs["ax"] = axs[0]
        lam_kwargs["idx"] = nc
        q_lam_kwargs["ax"] = axs[1]
        q_lam_kwargs["idx"] = idx
        self.plot_L(**lam_kwargs)
        self.plot_D(**q_lam_kwargs)
        lam_true = lam_kwargs.get("lam_true", None)
        fig.suptitle(
            self._parse_title(
                result=self.result if nc is None else self.pca_results.loc[[nc]],
                lam_true=lam_true,
            )
        )
        fig.tight_layout()

        return axs

    def param_density_plots(
        self,
        nc=None,
        lam_true=None,
        base_size=4,
        max_np=9,
    ):
        base_size = 4
        n_params = self.n_params if self.n_params <= max_np else max_np
        grid_plot = self._closest_factors(n_params)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, ax in enumerate(ax.flat):
            self.plot_L(nc=nc, param_idx=i, lam_true=lam_true, ax=ax)
            ax.set_title(f"$\lambda_{i}$")

        fig.suptitle(
            self._parse_title(
                result=self.result if nc is None else self.pca_results.loc[[nc]],
                nc=nc,
                lam_true=lam_true,
            )
        )
        fig.tight_layout()

    def nc_param_density_plots(
        self,
        nc_mask=None,
        param_idx=0,
        lam_true=None,
        base_size=4,
        max_np=9,
    ):
        base_size = 4
        if nc_mask is None:
            nc_mask = np.arange(1, len(self.pca_results) + 1)
        nc = len(nc_mask)
        nc = nc if nc <= max_np else max_np
        grid_plot = self._closest_factors(nc)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, ax in enumerate(ax.flat):
            self.plot_L(nc=nc_mask[i], param_idx=param_idx, lam_true=lam_true, ax=ax)
            result = self.pca_results.loc[[nc_mask[i]]]
            ax.set_title(self._parse_title(result=result, lam_true=lam_true, nc=True))

        fig.suptitle("MUD Estimates by Number of PCA Components Used")
        fig.tight_layout()

    def learned_qoi_plot(self, nc_mask=None):
        """
        Scatter plots of learned `q_pca` components.
        """
        max_nc = len(self.pca["vecs"])
        nc_mask = np.arange(max_nc) if nc_mask is None else nc_mask
        max_nc = self.n_states if self.n_states <= max_nc else max_nc
        g = sns.PairGrid(self.state[[f"q_pca_{i}" for i in nc_mask]], diag_sharey=False)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot)

    def _parse_title(
        self,
        result=None,
        nc=True,
        lam_true=None,
    ):
        """
        Parse title for plots
        """
        result = self.result if result is None else result
        title = super()._parse_title(result=result, lam_true=lam_true)
        if nc:
            title = f"nc = {result.index[0]}: " + title

        return title


class SequentialProblem(PCAMUDProblem):
    """
    Class defining a SequentialDensity Problem for parameter estimation on.

    To initialize the class, a forward model model, and parameters need to be
    sepcified. The main entrypoint for solving the estimation problem is the
    `seq_solve()` method, with the `search_params` class attribute controlling
    how the the sequential algorithm behaves.

    Attributes
    ----------
    forward_model : callable
        Function that runs the forward model. Should be callable using
    x0 : ndarray
        Initial state of the system.
    """

    def __init__(
        self,
        samples,
        data,
        std_dev,
        pi_in=None,
        weights=None,
    ):
        self.init_prob(samples, data, std_dev, pi_in=pi_in, weights=weights)

    @property
    def it(self):
        return len(self.states["results"])

    @property
    def num_it(self):
        return len(self.states["data"])

    def init_prob(self, samples, data, std_dev, pi_in=None, weights=None):
        """
        Initialize problem

        Initialize problem by setting the lambda samples, the values of the
        samples pushed through the forward map, and the observe distribution
        on the data. Can optionally as well set the initial and predicteed
        distributions explicitly, and pass in weights to incorporate prior
        beliefs on the `lam` sample sets.
        """
        # Assume gaussian error around mean of data with assumed noise
        super().init_prob(samples, data, std_dev, pi_in=pi_in, weights=weights)
        self.states = {
            "qoi": np.array_split(self.qoi, 1, axis=1),
            "data": np.array_split(self.data, 1, axis=0),
            "results": [],
            "states": [],
        }

    def _create_binary_string(self, lst, max_int):
        binary_string = ""
        for i in range(max_int):
            if i in lst:
                binary_string += "1"
            else:
                binary_string += "0"
        return binary_string

    def _get_qoi_combinations(
        self,
        max_tries=10,
    ):
        """
        Utility function to determine sets of ts combinations to iterate through
        """
        min_num = self.n_params if self.n_params <= self.n_qoi else self.n_qoi
        if self.qoi_method == "all":
            combs = [list(np.arange(self.n_qoi))]
        elif self.qoi_method == "linear":
            combs = [list(np.arange(0, i)) for i in range(min_num, self.n_qoi)]
        elif self.qoi_method == "random":
            # Divide the max#tries amongs the number of timesteps available
            if self.n_qoi < max_tries:
                num_ts_list = range(min_num, self.n_qoi + 1)
                tries_per = int(max_tries / self.n_qoi)
            else:
                num_ts_list = range(
                    min_num, self.n_qoi + 1, int(self.n_qoi / max_tries)
                )
                tries_per = 1

            combs = []
            qoi_choices = range(0, self.n_qoi)
            for num_ts in num_ts_list:
                psble = list(
                    [list(x) for x in itertools.combinations(qoi_choices, num_ts)]
                )
                tries_per = tries_per if tries_per < len(psble) else len(psble)
                combs += random.sample(psble, tries_per)

        return combs

    def solve(
        self,
        num_splits: int = 1,
        qoi_method: str = "all",
        min_weight_thresh: float = 1e-4,
        exp_thresh: float = 0.5,
        best_method: str = "closest",
        max_nc: int = None,
    ):
        """
        Detect shift and determine next action.
        """
        self.qoi_method = qoi_method
        self.min_weight_thresh = min_weight_thresh

        if self.qoi_method not in ["all", "linear", "random"]:
            msg = f"Unrecognized qoi method: {qoi}. Allowed: {am}"
            logger.error(msg)
            raise ValueError(msg)

        self.states["qoi"] = np.array_split(self.qoi, num_splits, axis=1)
        self.states["data"] = np.array_split(self.data, num_splits, axis=0)
        pi_in = self.dists["pi_in"]
        weights = self.state["weight"]
        it_results = []
        it_states = []
        best_it_results = []
        logger.info(f"Starting sequential algorithm with {num_splits}")
        for it in range(self.num_it):
            logger.info(f"Re-initializing problem with split #{it}")
            # Use parent method here, don't want to erase states dictionary
            super().init_prob(
                (self.lam, self.states["qoi"][it]),
                self.states["data"][it],
                self.std_dev,
                pi_in=pi_in,
                weights=weights,
            )
            qoi_combs = self._get_qoi_combinations()

            results = []
            qc_strs = []
            pca_states = []
            logger.info(f"{qoi_method}: Trying {len(qoi_combs)} qoi combs.")
            for q_idx, qc in enumerate(qoi_combs):
                qc_strs.append(self._create_binary_string(qc, self.n_qoi))
                logger.info(f"Trying comb of size {len(qc)}: {qc_strs[-1]}")
                super().solve(
                    pca_mask=qc,
                    max_nc=max_nc,
                    exp_thresh=exp_thresh,
                    best_method=best_method,
                )
                res_df = self.pca_results
                res_df["qoi_comb"] = q_idx

                # actions = []
                # for nc, res in res_df.groupby("nc"):
                #     actions.append(self._get_action(res))
                # res_df["action"] = actions
                results.append(res_df.set_index("qoi_comb", append=True))
                temp = self.pca_states.copy()
                temp["qoi_comb"] = q_idx
                pca_states.append(temp)

            res_df = pd.concat(results)
            res_df["closest"] = np.logical_and(
                res_df["predict_delta"]
                <= res_df[res_df["within_thresh"]]["predict_delta"].min(),
                res_df["within_thresh"],
            )
            res_df["max_kl"] = np.logical_and(
                res_df["kl"] >= res_df[res_df["within_thresh"]]["kl"].max(),
                res_df["within_thresh"],
            )
            res_df["min_kl"] = np.logical_and(
                res_df["kl"] <= res_df[res_df["within_thresh"]]["kl"].min(),
                res_df["within_thresh"],
            )
            # TODO: replace this extra call with just setting appropriately
            # to what's saved in history to not reproduce work.
            idx_max = res_df[self.best_method].idxmax()
            super().solve(
                pca_mask=qoi_combs[idx_max[1]],
                max_nc=max_nc,
                exp_thresh=self.exp_thresh,
                best_method=self.best_method,
            )
            it_results.append(res_df.copy())
            best_it_results.append(res_df.loc[[idx_max]].copy())

            pca_states = pd.concat(pca_states, axis=0)
            pca_states["split"] = it
            it_states.append(
                pca_states[
                    [
                        "split",
                        "qoi_comb",
                        "nc",
                        "weight",
                        "pi_obs",
                        "pi_pr",
                        "ratio",
                        "pi_up",
                    ]
                ]
            )

            if it + 1 < self.num_it:
                # TODO: Implement weight inflation if weights < min thresh?
                logger.info("Update: setting pi_up -> pi_in, ratio -> weights")
                pi_in = self.dists["pi_up"]
                weights = self.state["ratio"]

        best_it_result = pd.concat(
            best_it_results, keys=np.arange(self.num_it), names=["split"]
        )
        self.split_results = pd.concat(
            it_results, keys=np.arange(self.num_it), names=["split"]
        )
        self.split_states = pd.concat(it_states, axis=0)
        self.result = best_it_result.iloc[[num_splits - 1]]

    def get_summary_table(
        self,
    ):
        """
        Get a summary table of results to print
        """
        # TODO: Implement
        fields = ["Iteration", "NC", "E(r)", "D_KL"]

        table = Table(show_header=True, header_style="bold magenta")
        cols = ["Key", "Value"]
        for c in cols:
            table.add_column(c)

        res_df = self.results[-1]
        best_idx = res_df[best].argmax()
        row = (
            str(len(self.mud_res)),
            f"{res_df.loc[best_idx]['action']}",
            f"{res_df.loc[best_idx]['nc']:1.0f}",
            f"{res_df.loc[best_idx]['e_r']:0.3f}",
            f"{res_df.loc[best_idx]['kl']:0.3f}",
        )
        for i in range(len(fields)):
            table.add_row(fields[i], row[i])

        return table

    def plot_L(
        self,
        idx=None,
        lam_true=None,
        mud_point=None,
        df=None,
        param_idx=0,
        param_col="lam",
        ratio_col="ratio",
        weight_col="ratio",
        plot_initial=True,
        plot_legend=True,
        ax=None,
        figsize=(6, 6),
    ):
        """
        Plot Lambda Space Distributions

        Plot distributions over parameter space. This includes the initial and
        the updated distributions. Extends `MUDProblem` methods by allowing
        an `nc` argument to plot the solutions that use only `nc` number of
        principal components in the `q_pca` map. If `nc` is not specified, the
        best choice of `nc` will be picked from those tried, with the best
        being chosen according to the `best_method` set in `solve()`.

        Parameters
        ----------
        param_idx : int, default=0
            Index of parameter, `lam` to plot.
        lam_true: ArrayLike, default=None
            If specified, a vertical line for the true parameter solution will
            be added. Note this value must be the same dimension as the
            parameter space, even if it only the value at the `param_idx`
            specified is only used.
        plot_mud: bool, default=True
            Whether to add a vertical line for the computed MUD point solution.

        Returns
        -------
        ax, labels : Tuple
            Tuple of (1) matplotlib axis object where distributions where
            plotted and (2) List of labels that were plotted, in order plotted.
        """
        if df is None:
            df = self.state
            res = self._get_plot_df(idx, cols=["ratio", "weight"])
            if res is not None:
                df = res[0]
                mud_point = res[1]
                ratio_col = res[2][0]
                weight_col = res[2][1]

        ax, labels = super().plot_L(
            lam_true=lam_true,
            mud_point=mud_point,
            df=df,
            param_idx=param_idx,
            param_col=param_col,
            ratio_col=ratio_col,
            weight_col=weight_col,
            plot_initial=plot_initial,
            plot_legend=plot_legend,
            ax=ax,
            figsize=figsize,
        )

        return ax, labels

    def splits_param_density_plots(
        self,
        split_mask=None,
        nc=None,
        qoi_comb=None,
        param_idx=0,
        lam_true=None,
        base_size=4,
        max_splits=9,
    ):
        base_size = 4
        if split_mask is None:
            split_mask = np.arange(self.num_it)
        ns = len(split_mask)
        ns = ns if ns <= max_splits else max_splits
        grid_plot = self._closest_factors(ns)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size),
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        for i, ax in enumerate(ax.flat):
            result = self.split_results.loc[pd.IndexSlice[i, :, :], :]
            best_result = result.iloc[[result[self.best_method].argmax()]]
            best_idx = best_result.index.values[0]
            self.plot_L(idx=best_idx, param_idx=param_idx, lam_true=lam_true, ax=ax)
            ax.set_title(self._parse_title(result=result, lam_true=lam_true, nc=True))

        fig.suptitle("Best MUD Estimates by Split For $\lambda_{param_idx}$")
        fig.tight_layout()

    def get_full_df(
        self,
        df="state",
        iterations=None,
    ):
        """
        Concatenate stored df
        """

        if df not in self.dfs.keys():
            raise ValueError(f"{df} not one of {self.dfs.keys()}")

        dfs = self.dfs[df]
        if iterations is not None:
            dfs = [dfs[i] for i in range(len(dfs)) if i in iterations]

        return pd.concat(dfs, axis=0)

    def _get_plot_df(self, idx=None, cols=["ratio"]):
        """
        Helper function to get sub df to plot

        We use seaborn's kde plot on the dataframe of lambda samples to plot
        initial and updated distributions, but just weighted appropriately with
        the ratio. If want to plot solution using a different number of
        components as the optimal one stored in the 'ratio' column of the
        samples dataframe, then we have to get it from the pca_states dataframe
        which stores the results from the `solve` routine. In this helper method
        we extract those columns if necessary and append them to the state
        dataframe for plotting.
        """
        state_cols = self.split_states[
            (self.split_states["nc"] == idx[1])
            & (self.split_states["split"] == idx[0])
            & (self.split_states["qoi_comb"] == idx[2])
        ][cols].add_suffix("_plot")
        df = self.state.join(state_cols)
        col_names = [f"{c}_plot" for c in cols]
        mud_point = get_df(self.split_results.loc[[idx]], "lam_MUD", self.n_params)[0]

        return df, mud_point, col_names

    def _parse_title(
        self,
        result=None,
        lam_true=None,
        nc=True,
        qoi_comb=False,
        split=True,
    ):
        """
        Parse title for plots
        """
        result = self.result if result is None else result
        title = super()._parse_title(result=result, lam_true=lam_true, nc=False)
        if nc:
            title = f"{result.index[0][1]} NC, " + title
        if qoi_comb:
            title = f"qoi_comb = {result.index[0][2]}, " + title
        if split:
            title = f"Split {result.index[0][0]}: " + title

        return title


class DynamicSequentialProblem(SequentialProblem):
    """
    Dynamic Seqential MUD Parameter Estimation Problem

    To initialize the class, a forward model model, and parameters need to be
    sepcified. The main entrypoint for solving the estimation problem is the
    `seq_solve()` method, with the `search_params` class attribute controlling
    how the the sequential algorithm behaves.

    Attributes
    ----------
    forward_model : callable
        Function that runs the forward model. Should be callable using
    x0 : ndarray
        Initial state of the system.
    """

    def __init__(
        self,
        model,
    ):
        self.model = model
        self.push_forwards = []

    def _detect_shift(
        self,
        res,
    ):
        """ """
        shift = False
        prev = self.get_prev_best()
        if prev is None:
            return False
        if prev["action"] == "RESET":
            return False

        # Mean condition - Shift in the mean exp_r value detected
        shift = True
        if self.e_r_delta is not None:
            condition = np.abs(prev["e_r"] - res["e_r"].values[0]) <= self.e_r_delta
            shift = shift if condition else False

        # KL Divergence Condition - If exceeds threshold then shift
        if self.kl_thresh is not None:
            condition = res["kl"].values[0] < self.kl_thresh
            shift = shift if condition else False

        return shift

    def _get_action(
        self,
        res,
    ):
        """ """
        action = None
        if np.abs(1.0 - res["e_r"].values[0]) <= self.exp_thresh:
            if self.min_weight_thresh is not None:
                r_min = self.state[f"ratio"].min()
                r_min = r_min[0] if not isinstance(r_min, np.float64) else r_min
                if r_min >= self.min_weight_thresh:
                    action = "RE-WEIGHT"
            if action != "RE-WEIGHT":
                action = "UPDATE"
        elif self._detect_shift(res):
            action = "RESET"

        return action

    def solve(
        self,
        time_windows,
        diff=0.5,
        num_samples=1000,
        seed=None,
        max_nc: int = None,
        splits_per: int = 1,
        qoi_method: str = "all",
        e_r_delta: float = 0.5,
        kl_thresh: float = 3.0,
        min_weight_thresh: float = 1e-4,
        exp_thresh: float = 0.5,
        best_method: str = "closest",
    ):
        """
        Iterative Solver

        Iterative between solving and pushing model forward using sequential
        MUD algorithm for parameter estimation.

        Parameters
        ----------

        Returns
        -------

        Note
        ----
        This will reset the state of the class and erase its previous dataframes.
        """
        bad = []
        if self.e_r_delta < 0.5:
            bad += [f"Shift detection delta(E(r)) must be >= 0.5: {self.e_r}"]
        if self.kl_thresh < 3.0:
            bad += [f"Shift detection D_KL_thresh(r) must be >= 3.0: {kl}"]
        if len(bad) > 0:
            msg = "Bad args:\n" + "\n".join(bad)
            logger.error(msg)
            raise ValueError(msg)

        self.diff = diff
        if self.samples is not None:
            yn = input("Previous run exists. Do you want to reset state? y/(n)")
            if yn == "n":
                return
            self.push_forwards = []
            self.states = []

        np.random.seed(seed)  # Initial seed for sampling
        self.samples = self.model.get_uniform_initial_samples(
            scale=diff, num_samples=num_samples
        )
        if len(time_windows) < 2:
            time_windows.insert(0, 0)
        time_windows.sort()
        self.model.t0 = time_windows[0]

        logger.info(f"Starting solve over time : {time_windows}")
        for it, tf in enumerate(time_windows[1:]):
            t0 = time_windows[it]
            logger.info(f"Iteration {it} [{t0}, {tf}]: ")
            args = self.forward_solve(tf, samples=self.samples)
            mud_args = self.get_mud_args()
            self.mud_prob.init_prob(
                *[mud_args[x] for x in ["lam", "q_lam", "data", "std_dev"]]
            )
            super().solve(
                num_splits=num_splits,
                qoi_method=qoi_method,
                min_weight_thresh=min_weight_thresh,
                exp_thresh=exp_thresh,
                best_method=best_method,
                max_nc=max_nc,
            )
            self.iteration_update()
            logger.info(f" Summary:\n{log_table(self.get_summary_row())}")

    def iteration_update(
        self,
    ):
        """
        Perform an update after a Sequential MUD estimation
        """
        action = self.mud_prob.result["action"].values[0]
        if action == "UPDATE":
            logger.info("Drawing from updated distribution")
            self.samples = self.mud_prob.sample_update(self.n_samples)
            self.sample_weights = None
        elif action == "RESET":
            logger.info("Reseting to initial distribution")
            self.samples = self.get_uniform_initial_samples(
                scale=self.diff, num_samples=self.n_samples
            )
        elif action == "RE-WEIGHT":
            logger.info("Re-weighting current samples")
            self.sample_weights = (
                self.mud_prob.state["weight"] * self.mud_prob.state["ratio"]
            )
        else:
            logger.info("No action taken, continuing with current samples")

    def get_summary_row(
        self,
    ):
        """ """
        best = self.search_params["best"]

        fields = ["Iteration", "Action", "NC", "E(r)", "D_KL"]

        table = Table(show_header=True, header_style="bold magenta")
        cols = ["Key", "Value"]
        for c in cols:
            table.add_column(c)

        res_df = self.results[-1]
        best_idx = res_df[best].argmax()
        row = (
            str(len(self.mud_res)),
            f"{res_df.loc[best_idx]['action']}",
            f"{res_df.loc[best_idx]['nc']:1.0f}",
            f"{res_df.loc[best_idx]['e_r']:0.3f}",
            f"{res_df.loc[best_idx]['kl']:0.3f}",
        )
        for i in range(len(fields)):
            table.add_row(fields[i], row[i])

        return table

    def get_full_df(
        self,
        df="state",
        iterations=None,
    ):
        """
        Concatenate stored df
        """

        if df not in self.dfs.keys():
            raise ValueError(f"{df} not one of {self.dfs.keys()}")

        dfs = self.dfs[df]
        if iterations is not None:
            dfs = [dfs[i] for i in range(len(dfs)) if i in iterations]

        return pd.concat(dfs, axis=0)
