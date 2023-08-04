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
from scipy.stats import rv_continuous  # type: ignore
from scipy.stats import entropy, gaussian_kde
from scipy.stats.distributions import norm
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from pydci.log import disable_log, enable_log, log_table, logger
from pydci.utils import KDEError, fit_domain, get_df, gkde, put_df, set_shape, closest_factors
from pydci.plotting import DEF_RC_PARAMS

sns.color_palette("bright")
sns.set_style("darkgrid")
plt.rcParams.update(DEF_RC_PARAMS)

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

    def init_prob(self, samples, pi_obs, pi_in=None, pi_pr=None):
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
        self.state["weight"] = 1.0
        self.state = put_df(self.state, "q_lam", self.q_lam, size=self.n_states)
        self.state = put_df(self.state, "lam", self.lam, size=self.n_params)
        self.dists = {
            "pi_in": pi_in,
            "pi_pr": pi_pr,
            "pi_obs": pi_obs,
            "pi_up": None,
            "pi_pf": None,
        }
        self.result = None

    def pi_in(self, values=None):
        """
        Evaluate the initial distribution.

        Init distribion is either set explicitly in by a call to `init_prob`
        or calculated from a gaussain kernel density estimate (using scipy) on
        the initial samples, weighted by the sample weights.
        """
        if self.dists["pi_in"] is None:
            logger.debug("Calculating pi_in by computing KDE on lam")
            try:
                self.dists["pi_in"] = gkde(
                    self.lam.T,
                    weights=self.state["weight"],
                    label="Initial Distribution",
                )
            except KDEError as k:
                k.msg = "KDE failed on initial samples"
                raise k
        values = self.lam if values is None else values
        if isinstance(self.dists["pi_in"], gaussian_kde):
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
            logger.debug("Calculating pi_pr by computing KDE on q_lam")
            try:
                self.dists["pi_pr"] = gkde(
                    self.q_lam.T,
                    weights=self.state["weight"],
                    label="Predicted Distribution",
                )
            except KDEError as k:
                k.msg = "KDE failed on observations"
                raise k
        values = self.q_lam if values is None else values
        if isinstance(self.dists["pi_pr"], gaussian_kde):
            return self.dists["pi_pr"].pdf(values.T).T.ravel()
        else:
            return self.dists["pi_pr"].pdf(values).prod(axis=1)

    def pi_obs(self, values=None):
        """
        Evaluate the observed distribution.

        Observed distribion is set explicitly in the call to `init_prob`.
        """
        values = self.q_lam if values is None else values
        if isinstance(self.dists["pi_obs"], gaussian_kde):
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
            try:
                self.dists["pi_up"] = gkde(
                    self.lam.T,
                    weights=self.state["ratio"] * self.state["weight"],
                    label="Updated Distribution",
                )
            except KDEError as k:
                k.msg = "KDE failed on updated samples"
                raise k
        values = np.array(values)
        values = self.lam if values is None else values
        return self.dists["pi_up"].pdf(np.array(values).T).T

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
            try:
                self.dists["pi_pf"] = gkde(
                    self.q_lam.T,
                    weights=self.state["ratio"] * self.state["weight"],
                    label="Push-Forward of Updated Distribution",
                )
            except KDEError as k:
                k.msg = "KDE failed on updated observations"
                raise k
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
        if isinstance(self.dists[dist], gaussian_kde):
            return self.dists[dist].resample(size=num_samples).T
        else:
            dim = self.n_params if dist in ["pi_in", "pi_up"] else self.n_states
            if self.dists[dist] is None:
                _ = getattr(self, dist, None)(np.zeros(dim))
            if not isinstance(self.dists[dist], gaussian_kde):
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

        Note: Setting weights resets the initial, predicted, updated, and
        push-forward of updated distributions, as they need to be recalculated
        using new set of sample weights. Only observed distribution is left
        untouched, since it is given by the user.

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

            # If non-zero weights set, whipe saved distributions
            self.dists["pi_in"] = None
            self.dists["pi_pr"] = None
            self.dists["pi_up"] = None
            self.dists["pi_pf"] = None

        self.state["weight"] = w

    def solve(self):
        """
        Solve the data consistent inverse problem by computing `pi_up` as the
        a multiplicative update to `pi_in`:

        `s['pi_up'] = s['weights'] * s['pi_in'] (s['pi_obs'] / s['pi_pr'])`

        Quantities for each density evaluated at each parameter sample/push
        forward value are stored in the `state` attribute DataFrame, along
        with the ratio of the observed (`pi_obs`) to predicted (`pi_pr`), for
        ease of access later.

        Raises
        ------
        ZeroDivisionError
            If the predictability assumption is violated for any sample. This
            mean's that the ratio of the observed to the predicted distributions
            is undefined or infinite, indicating our predictions aren't able
            to predict our observations.
        """
        self.state["pi_in"] = self.pi_in()
        self.state["pi_obs"] = self.pi_obs()
        pi_pr = self.pi_pr()
        self.state["pi_pr"] = pi_pr
        self.state["ratio"] = np.divide(self.state["pi_obs"], self.state["pi_pr"])
        update = np.multiply(self.state["ratio"], self.state["weight"])
        if len(bad := np.where(~np.isfinite(update))[0]) > 0:
            raise ZeroDivisionError(
                f"Predictability assumption violated for samples {len(bad)}/"
                + f"{self.n_samples} samples."
            )
        self.state["pi_up"] = np.multiply(self.state["pi_in"], update)

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
        initial_kwargs={},
        update_kwargs={},
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

        if initial_kwargs is not None:
            pi_in_label = f"$\pi^{{in}}_{{\lambda_{param_idx}}}$"
            init_args = dict(
                data=df,
                x=f"{param_col}_{param_idx}",
                ax=ax,
                fill=True,
                color=bright_colors[param_idx],
                linestyle=":",
                label=pi_in_label,
                weights=weight_col,
            )
            init_args.update(initial_kwargs)
            sns.kdeplot(**init_args)
            labels.append(init_args['label'])

        if update_kwargs is not None:
            pi_up_label = f"$\pi^{{up}}_{{\lambda_{param_idx}}}$"
            update_args = dict(
                data=df,
                x=f"{param_col}_{param_idx}",
                ax=ax,
                fill=True,
                color=bright_colors[param_idx],
                label=pi_up_label,
                weights=df[weight_col] * df[ratio_col],
            )
            update_args.update(update_kwargs)
            sns.kdeplot(
                **update_args
            )
            labels.append(update_args['label'])

        # Set plot specifications
        ax.set_xlabel(f"$\lambda_{param_idx}$")
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
        ax.set_xlabel(r"$\mathcal{D}$")
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
        max_np=8,
        figsize=(14, 6),
        lam_kwargs=None,
    ):
        # TODO: Add explicit figsize argument.
        base_size = 4
        n_params = self.n_params if self.n_params <= max_np else max_np
        grid_plot = closest_factors(n_params)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2),
                     grid_plot[0] * base_size) if figsize is None else figsize,
        )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        lam_kwargs = {} if lam_kwargs is None else lam_kwargs
        for i, ax in enumerate(ax.flat):
            plot_args = dict(param_idx=i, lam_true=lam_true, ax=ax)
            if i in lam_kwargs.keys():
                plot_args.update(lam_kwargs[i])
            self.plot_L(**plot_args)

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
