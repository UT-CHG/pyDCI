"""
Data Consistent Inversion Problem

The classes in this module all derive off of the Consistent-Bayesian formulation
for solving Stochastic Inverse problems first proposed in [1]. 
Inherited classes have the form:

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
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from scipy.stats import rv_continuous  # type: ignore
from scipy.stats import entropy, gaussian_kde

from pydci.plotting import DEF_RC_PARAMS
from pydci.utils import (
    KDEError,
    closest_factors,
    get_df,
    gkde,
    put_df,
    set_shape,
    print_rich_table,
    fmt_bytes,
)
import pydci.notation as dcin

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

    def __str__(self) -> str:
        info_df = self.get_info_table()
        return print_rich_table(
            info_df,
            columns=['num_samples', 'num_params', 'num_states',
                     'mem_usage', 'solved', 'e_r', 'kl', 'error'],
            vertical=True,
            title='DCI Problem',
        )
    
    def get_info_table(self) -> pd.DataFrame:
        """
        Get Info Table
        """
        info_df = self.result.copy()
        info_df['mem_usage'] = fmt_bytes(self.state.memory_usage(deep=True).sum() / 1024)
        info_df['num_samples'] = self.n_samples
        info_df['num_params'] = self.n_params
        info_df['num_states'] = self.n_states

        return info_df

    def init_prob(
        self,
        samples: Union[pd.DataFrame, List[np.ndarray]],
        pi_obs: Union[gaussian_kde, rv_continuous],
        pi_in: Optional[Union[gaussian_kde, rv_continuous]] = None,
        pi_pr: Optional[Union[gaussian_kde, rv_continuous]] = None,
        weights: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize the problem.

        Initialize the problem by setting the lambda samples, the values of the
        samples pushed through the forward map, and the observe distribution
        on the data. Optionally, set the initial and predicted distributions
        explicitly, and pass in weights to incorporate prior beliefs on the `lam`
        sample sets.

        Parameters
        ----------
        samples : Union[pd.DataFrame, List[np.ndarray]]
            Lambda samples. If a DataFrame, the columns with names starting with "l"
            are considered parameters, and the remaining columns are considered states.
            Otherwise, the samples are given as a list of two arrays, the first
            containing the lambda samples and the second the q_lambda samples.
        pi_obs : Union[gaussian_kde, rv_continuous]
            Observe distribution on the data.
        pi_in : Optional[Union[gaussian_kde, rv_continuous]], default=None
            Initial distribution. If None, will be calculated from initial samples.
        pi_pr : Optional[Union[gaussian_kde, rv_continuous]], default=None
            Predicted distribution. If None, will be calculated from inintial samples.
        weights : Optional[np.ndarray], default=None
            Weights to incorporate prior beliefs on the `lam` sample sets. If specified,
            will be used as weights fro the KDEs in pi_in, pi_pr, and the computation
            of the expected ratio.

        Returns
        -------
        None
            The function modifies the internal state of the object and doesn't return anything.
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
            "pi_pf": None,
        }
        self.result = pd.DataFrame(
            [[np.nan, np.nan, np.nan, np.nan, False, None]],
            columns=["e_r", "kl", "k_eff", "k_eff_up", "solved", "error"]
        )
        self.set_weights(weights)
        self.states = []
        self.results = []

    def pi_in(self, values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate the initial distribution.

        The initial distribution is either set explicitly by a call to `init_prob`
        or calculated from a Gaussian kernel density estimate (using scipy) on
        the initial samples, weighted by the sample weights.

        Parameters
        ----------
        values : Optional[np.ndarray], default=None
            Values at which to evaluate the distribution. If None, uses `self.lam`.

        Returns
        -------
        np.ndarray
            Evaluated initial distribution.

        Raises
        ------
        KDEError
            If the kernel density estimation fails on initial samples. note
            if initial weights specified, this can be because the weights are reducing
            the sample size significantly..
        """
        if self.dists["pi_in"] is None:
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

    def pi_pr(self, values: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """
        Evaluate the predicted distribution.

        The predicted distribution is either set explicitly in the call to `init_prob`
        or calculated from a Gaussian kernel density estimate (using scipy) on
        the push forward of the initial samples, q_lam, weighted by the sample weights.

        Parameters
        ----------
        values : Optional[np.ndarray], default=None
            Values at which to evaluate the distribution. If None, uses `self.q_lam`.

        Returns
        -------
        Union[np.ndarray, float]
            Evaluated predicted distribution.

        Raises
        ------
        KDEError
            If the kernel density estimation fails on predicted samples. note
            if initial weights specified, this can be because the weights are reducing
            the sample size significantly..
        """
        if self.dists["pi_pr"] is None:
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

    def pi_obs(self, values: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """
        Evaluate the observed distribution.

        The observed distribution is set explicitly in the call to `init_prob`.

        Parameters
        ----------
        values : Optional[np.ndarray], default=None
            Values at which to evaluate the distribution. If None, uses `self.q_lam`.

        Returns
        -------
        Union[np.ndarray, float]
            Evaluated observed distribution.
        """
        values = self.q_lam if values is None else values
        if isinstance(self.dists["pi_obs"], gaussian_kde):
            return self.dists["pi_obs"].pdf(values.T).T.ravel()
        else:
            return self.dists["pi_obs"].pdf(values).prod(axis=1)

    def pi_up(self, values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate the updated distribution.

        Computed using scipy's Gaussian kernel density estimation on the
        initial samples, but weighted by the ratio of the updated and predicted
        distributions (evaluated at each sample value). Note, if the initial
        samples were weighted, then the weights are applied as well.

        Parameters
        ----------
        values : Optional[np.ndarray], default=None
            Values at which to evaluate the distribution. If None, uses `self.lam`.

        Returns
        -------
        np.ndarray
            Evaluated updated distribution.

        Raises
        ------
        KDEError
            If the kernel density estimation fails on the updated samples. Note
            if update ratio is very small for too many samples, this can fail
            as the weights are reducing the sample size significantly.
        """
        if self.dists["pi_up"] is None:
            try:
                self.dists["pi_up"] = gkde(
                    self.lam.T,
                    weights=self.state["weighted_ratio"],
                    label="Updated Distribution",
                )
            except KDEError as k:
                k.msg = "KDE failed on updated samples"
                raise k

        values = self.lam if values is None else np.array(values)
        return self.dists["pi_up"].pdf(values.T).T

    def pi_pf(self, values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate the push-forward of the updated distribution.

        Computed using scipy's Gaussian kernel density estimation on the
        initial samples, but weighted by the ratio of the updated and predicted
        distributions (evaluated at each sample value). Note, if the initial
        samples were weighted, then the weights are applied as well.

        Parameters
        ----------
        values : Optional[np.ndarray], default=None
            Values at which to evaluate the distribution. If None, uses `self.q_lam`.

        Returns
        -------
        np.ndarray
            Evaluated push-forward of the updated distribution.

        Raises
        ------
        KDEError
            If the kernel density estimation fails on the updated predicted samples.
            Note if update ratio is very small for too many samples, this can fail
            as the weights are reducing the sample size significantly.
        """
        if self.dists["pi_pf"] is None:
            try:
                self.dists["pi_pf"] = gkde(
                    self.q_lam.T,
                    weights=self.state["weighted_ratio"],
                    label="Push-Forward of Updated Distribution",
                )
            except KDEError as k:
                k.msg = "KDE failed on updated observations"
                raise k

        values = self.q_lam if values is None else values
        return self.dists["pi_pf"].pdf(values.T).T

    def sample_dist(self, num_samples: int = 1, dist: str = "pi_up") -> np.ndarray:
        """
        Sample Stored Distribution.

        Samples from a stored distribution. By default samples from the updated
        distribution on parameter samples, but also can draw samples from any
        stored distribution: pi_in, pi_pr, pi_obs, pi_up, and pi_pf.

        Parameters
        ----------
        num_samples : int, optional, default=1
            Number of samples to draw from the distribution.
        dist : str, optional, default='pi_up'
            Distribution to sample from. By default, samples from the update
            distribution.

        Returns
        -------
        np.ndarray
            Samples from the updated distribution. Dimension of the array is
            (num_samples, num_params).

        Raises
        ------
        ValueError
            If the specified distribution is not recognized distribution.
        """
        if dist not in self.dists:
            raise ValueError(f"Unknown distribution: {dist}")

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
        KDEError
            If the KDE fails to fit the initial or predicted distributions from
            the data specified.
        """
        self.state["pi_in"] = self.pi_in()
        self.state["pi_obs"] = self.pi_obs()
        self.state["pi_pr"] = self.pi_pr()
        self.state["ratio"] = np.divide(self.state["pi_obs"], self.state["pi_pr"])
        self.state["weighted_ratio"] = np.multiply(
            self.state["ratio"], self.state["weight"])
        self.state["pi_up"] = np.multiply(
            self.state["pi_in"], self.state['weighted_ratio'])
        self.state["pred_assumption"] = self.state["weighted_ratio"].notna()
        self.result['e_r'] = self.expected_ratio()
        self.result['kl'] = self.divergence_kl()
        self.result['k_eff'] = len(np.where(
            self.state['weight'].values > 1e-10)[0]) / self.n_samples
        self.result['k_eff_up'] = len(np.where(
            self.state['weighted_ratio'].values > 1e-10)[0]) / self.n_samples

        num_violated = (~self.state["pred_assumption"]).sum()
        if num_violated > 0:
            msg = "Obsered/Predicted = Inf for " +\
                f"{num_violated}/{self.n_samples} samples"
            self.result['error'] = msg
            raise ZeroDivisionError(msg)
        else:
            self.result['solved'] = True

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

    def save_state(self, vals):
        """
        Save current state, adding columns with values in vals dictionary
        """
        keys = vals.keys()
        cols = [c for c in self.state.columns if c.startswith("lam_")]
        cols += [c for c in self.state.columns if c.startswith("q_pca_")]
        cols += ["weight", "pi_in", "pi_obs", "pi_pr", "ratio",
                 "weighted_ratio", "pi_up", "pred_assumption"]
        state = self.state[cols].copy()
        for key in keys:
            state[key] = vals[key]
        if len(self.states) == 0:
            self.states = state
        else:
            self.states = pd.concat([self.states, state], axis=0)

        self.results.append(self.result)

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
            # pi_in_label = dcin.pi('in', arg=dcin.param(idx=param_idx))
            # pi_in_label = rf"$\pi^{{in}}_{{\lambda_{param_idx}}}$"
            init_args = dict(
                data=df,
                x=f"{param_col}_{param_idx}",
                ax=ax,
                fill=True,
                color=bright_colors[param_idx],
                linestyle=":",
                label=dcin.pi('in', arg=dcin.lam(), idx=param_idx),
                weights=weight_col,
            )
            init_args.update(initial_kwargs)
            sns.kdeplot(**init_args)
            labels.append(init_args["label"])

        if update_kwargs is not None:
            # pi_up_label = dcin.pi('up', arg=dcin.param(idx=param_idx))
            # pi_up_label = rf"$\pi^{{up}}_{{\lambda_{param_idx}}}$"
            update_args = dict(
                data=df,
                x=f"{param_col}_{param_idx}",
                ax=ax,
                fill=True,
                color=bright_colors[param_idx],
                label=dcin.pi('up', arg=dcin.lam(), idx=param_idx),
                weights=df[weight_col] * df[ratio_col],
            )
            update_args.update(update_kwargs)
            sns.kdeplot(**update_args)
            labels.append(update_args["label"])

        # Set plot specifications
        ax.set_xlabel(dcin.lam(idx=param_idx))
        # ax.set_xlabel(rf"$\lambda_{param_idx}$")
        if plot_legend:
            ax.legend(
                labels=labels,
                fontsize=12,
                title_fontsize=12,
            )

        # fig.tight_layout()

        return ax, labels

    def plot_D(
        self,
        df=None,
        state_idx=0,
        state_col="q_lam",
        ratio_col="ratio",
        weight_col="weight",
        pr_kwargs={},
        obs_kwargs={},
        pf_kwargs={},
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
        if pr_kwargs is not None:
            pr_args = dict(
                data=self.state,
                x=f"{state_col}_{state_idx}",
                ax=ax,
                fill=True,
                color=bright_colors[state_idx],
                linestyle=":",
                label=dcin.pi('pr', arg=dcin.q_pca(idx=state_idx), idx=state_idx), # rf"$\pi^{{pr}}_{{Q(\lambda)_{state_idx}}}$",
                weights=self.state["weight"],
            )
            pr_args.update(pr_kwargs)
            sns.kdeplot(**pr_args)
            labels.append(pr_args['label'])
        
        if obs_kwargs is not None:
            # TODO: Check this
            obs_label = dcin.pi('ob', arg=dcin.q_pca(idx=state_idx), idx=state_idx)
            # obs_label = rf"$\pi^{{obs}}_{{\lambda_{state_idx}}}$"
            obs_args = dict(
                color="r",
                label=obs_label,
            )
            obs_domain = ax.get_xlim()
            obs_x = np.linspace(obs_domain[0], obs_domain[1], 10000)
            obs_x_marginal = np.zeros((len(obs_x), self.n_states))
            obs_x_marginal[:, state_idx] = obs_x
            obs = self.pi_obs(values=obs_x_marginal)
            obs_args.update(obs_kwargs)
            ax.plot(obs_x, obs, **obs_kwargs)
            sns.kdeplot(**obs_args)
            labels.append(obs_args["label"])

        if pf_kwargs is not None:
            pf_label = dcin.pi('pf', arg=dcin.q_pca(idx=state_idx), idx=state_idx)
            # pf_label = rf"$\pi^{{pf}}_{{\lambda_{state_idx}}}$"
            pf_args = dict(
                data=df,
                x=f"{state_col}_{state_idx}",
                ax=ax,
                fill=True,
                color=bright_colors[state_idx],
                label=pf_label,
                linestyle="-",
                weights=df[weight_col] * df[ratio_col],
            )
            pf_args.update(pf_kwargs)
            sns.kdeplot(**pf_args)
            labels.append(pf_args["label"])

        ax.set_xlabel(fr"${{\mathbf{{q}}_{state_idx}}}$")
        if plot_legend:
            ax.legend(
                labels=labels,
                fontsize=12,
                title_fontsize=12,
            )

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
        param_idx=0,
        state_idx=0,
        lam_true=None,
        lam_kwargs=None,
        q_lam_kwargs=None,
        axs=None,
        figsize=(14, 6),
    ):
        """
        Plot param and observable space onto sampe plot
        """
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        elif len(axs) != 2:
            len(axs) != self.n_params
        lam_kwargs = {} if lam_kwargs is None else lam_kwargs
        q_lam_kwargs = {} if q_lam_kwargs is None else q_lam_kwargs
        lam_kwargs["param_idx"] = param_idx
        lam_kwargs["ax"] = axs[0]
        q_lam_kwargs["ax"] = axs[1]
        q_lam_kwargs["state_idx"] = state_idx
        self.plot_L(**lam_kwargs)
        self.plot_D(**q_lam_kwargs)
        lam_true = lam_kwargs.get("lam_true", None)
        fig = axs[0].get_figure()
        fig.suptitle(
            self._parse_title(
                title=tilte,
                result=self.result,
            )
        )

        return axs

    def param_density_plots(
        self,
        base_size=4,
        max_np=8,
        figsize=(14, 6),
        title=None,
        lam_kwargs=None,
    ):
        # TODO: Add explicit figsize argument.
        base_size = 4
        n_params = self.n_params if self.n_params <= max_np else max_np
        grid_plot = closest_factors(n_params)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size)
            if figsize is None
            else figsize,
        )

        lam_kwargs = {} if lam_kwargs is None else lam_kwargs
        for i, ax in enumerate(ax.flat):
            plot_args = dict(param_idx=i, lam_true=lam_true, ax=ax)
            plot_args.update(lam_kwargs)
            self.plot_L(**plot_args)

        fig.suptitle(self._parse_title(title=title))

    def state_density_plots(
        self,
        base_size=4,
        max_ns=8,
        figsize=(14, 6),
        q_lam_kwargs=None,
        title=None,
    ):
        # TODO: Add explicit figsize argument.
        base_size = 4
        n_states = self.n_states if self.n_states <= max_ns else max_ns
        grid_plot = closest_factors(n_states)
        fig, ax = plt.subplots(
            grid_plot[0],
            grid_plot[1],
            figsize=(grid_plot[0] * (base_size + 2), grid_plot[0] * base_size)
            if figsize is None
            else figsize,
        )

        q_lam_kwargs = {} if q_lam_kwargs is None else q_lam_kwargs
        for i, ax in enumerate(ax.flat):
            plot_args = dict(state_idx=i, ax=ax)
            plot_args.update(q_lam_kwargs)
            self.plot_D(**plot_args)

        fig.suptitle(self._parse_title(title=title))

    def _parse_title(
        self,
        title=None,
        result=None,
    ):
        """
        Parse title for plots
        """
        result = self.result if result is None else result
        kl = result["kl"].values[0]
        e_r = result["e_r"].values[0]
        # title = f"$\mathbb{{E}}(r)$= {e_r:.3f}, " + f"$\mathcal{{KL}}_{{DCI}}$= {kl:.3f}"
        title = title if title is not None else ""
        title += f'{dcin.exp_ratio_str(e_r, format_spec=".3f")},{dcin.kl_str(kl, format_spec=".3f")}'
        
        return title
        
