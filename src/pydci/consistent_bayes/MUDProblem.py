"""
Maximal Updated Density (MUD) Problem Class

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
[1] M. Pilosov, C. del-Castillo-Negrete, T. Y. Yen, T. Butler, and C.
Dawson, “Parameter estimation with maximal updated densities,” Computer
Methods in Applied Mechanics and Engineering, vol. 407, p. 115906, Mar.
2023, doi: 10.1016/j.cma.2023.115906.

"""
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import distributions as dist  # type: ignore

from pydci import DCIProblem
from pydci.log import log_table, logger
from pydci.utils import get_df, put_df, set_shape, print_rich_table, closest_factors

sns.color_palette("bright")
sns.set_style("darkgrid")

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"


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
        self.init_prob(samples, data, std_dev, pi_in=pi_in, pi_pr=pi_pr, weights=weights)
    
    def __str__(self) -> str:
        info_df = self.get_info_table()
        return print_rich_table(
            info_df,
            columns=['num_samples', 'num_params', 'num_states',
                     'mem_usage', 'solved', 'lam_mud', 'mud_idx', 'e_r', 'kl', 'error'],
            vertical=True,
            title='MUDProblem',
        )
    
    def get_info_table(self) -> pd.DataFrame:
        """
        Get Info Table
        """
        info_df = super().get_info_table()
        info_df['lam_mud'] = str(self.mud_point)
        info_df['mud_idx'] = str(self.mud_arg)

        return info_df

    def init_prob(self, samples, data, std_dev, pi_in=None, pi_pr=None, weights=None):
        """
        Initialize problem

        Initialize problem by setting the lambda samples, the values of the
        samples pushed through the forward map, and the observe distribution
        on the data. Can optionally as well set the initial and predicted
        distributions explicitly, and pass in weights to incorporate prior
        beliefs on the `lam` sample sets.
        """
        # Assume gaussian error around mean of data with assumed noise
        self.std_dev = std_dev
        if isinstance(data, pd.DataFrame):
            # Compute n_states because self.n_states won't be set until super().init_prob called
            n_states = len([c for c in data.columns if c.startswith("q_lam_obs")])
            data = data.dropna()
            data = get_df(
                data,
                "q_lam_obs",
                size=n_states,
                dropna=True,
            )
            self.data = data.reshape(-1, 1)
        else:
            self.data = set_shape(np.array(data), (-1, 1))

        pi_obs = dist.norm(loc=np.mean(data), scale=std_dev)
        super().init_prob(samples, pi_obs, pi_in=pi_in, pi_pr=pi_pr, weights=weights)
        self.mud_point = None
        self.mud_arg = None

    def solve(self):
        """
        Solve MUD Parameter Estimation Problem

        Extends the parent method by computing the MUD point, the solution
        to the parameter estimation problem, as the samples that maximizes the
        `pi_up` column in the state DataFrame. This MUD Point is stored in the
        results DataFrame that is returned.
        """
        super().solve()
        m, mud_point = self.get_mud_point()
        self.result = put_df(self.result, "lam_MUD", mud_point, size=self.n_params)
        self.result["MUD_idx"] = m
        self.mud_point = mud_point[0]
        self.mud_arg = m

    def get_mud_point(self, state_df: pd.DataFrame = None) -> tuple:
        """
        Get MUD Point from DataFrame.

        Get MUD point from DataFrame. If DataFrame is not passed in, use the
        `result` attribute of the class.

        Parameters
        ----------
        state_df : DataFrame, optional
            The state DataFrame. If not provided, the method will use the `state`
            attribute of the class.

        Returns
        -------
        tuple
            A tuple containing the index of the maximum "pi_up" value in the
            DataFrame and the MUD point calculated using the `get_df` method
            with the "lam" column and `n_params` attribute of the class.

        """
        if state_df is None:
            state_df = self.state

        m = np.argmax(state_df["pi_up"])
        mud_point = get_df(state_df.iloc[[m]], "lam", size=self.n_params)

        return m, mud_point

    def compute_l2_errs(self, lam_true: np.ndarray = None, result_df: pd.DataFrame = None) -> None:
        """
        Compute L2 errors between the true lambda values and MUD point.

        If `lam_true` is not provided, the function will return without performing any computation.
        The function computes both absolute and relative L2 errors and updates the `result` attribute
        of the class with the calculated errors.

        Parameters
        ----------
        lam_true : ndarray, optional
            The true lambda values. If not provided, the function returns without performing
            any computations.

        Returns
        -------
        None
            The function modifies the `result` attribute of the class in-place and does not return
            a value.
        """
        res_df = self.result if result_df is None else result_df
        mud_point = self.get_mud_point()[1]

        if lam_true is not None:
            lam_true = np.reshape(lam_true, (1, -1))
            res_df['l2_err'] = np.linalg.norm(lam_true - mud_point)
            res_df['rel_l2_err'] = res_df['l2_err'] / np.linalg.norm(lam_true)
            for i in range(self.n_params):
                res_df[f'l2_err_{i}'] = np.linalg.norm(lam_true[0, i] - mud_point[0, i])
                res_df[f'rel_l2_err_{i}'] = res_df[f'l2_err_{i}'] / np.linalg.norm(lam_true[0, i])

        if result_df is not None:
            return res_df

        self.result = res_df

    def plot_L(
        self,
        lam_true=None,
        df=None,
        param_idx=0,
        param_col="lam",
        ratio_col="ratio",
        weight_col="weight",
        plot_legend=True,
        mud_kwargs={},
        initial_kwargs={},
        update_kwargs={},
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
            plot_legend=plot_legend,
            initial_kwargs=initial_kwargs,
            update_kwargs=update_kwargs,
            ax=ax,
            figsize=figsize,
        )

        if mud_kwargs is not None:
            mud_point = self.get_mud_point(df)[1][0]
            mud_label = (
                f"$\lambda^{{MUD}}_{param_idx+1} = " + f"{mud_point[param_idx]:.4f}$"
            )
            mud_args = dict(
                x=mud_point[param_idx],
                linewidth=3,
                color="green",
                linestyle="--",
                label=mud_label,
            )
            mud_args.update(mud_kwargs)
            ax.axvline(**mud_args)
            labels.append(mud_args["label"])

        # Generate vertical lines for true values
        if lam_true is not None:
            lam_true = np.reshape(lam_true, (1, -1))
            lam_true_label = (
                f"$\lambda^{{\dagger}}_{param_idx+1} = "
                + f"{lam_true[0][param_idx]:.4f}$"
            )
            ax.axvline(
                x=lam_true[0][param_idx],
                linewidth=3,
                color="orange",
                label=lam_true_label,
            )
            labels.append(lam_true_label)

        if plot_legend:
            ax.legend(
                labels=labels,
                fontsize=12,
                title_fontsize=12,
            )

        return ax, labels
    
    def param_density_plots(
        self,
        lam_true=None,
        base_size=4,
        max_np=8,
        figsize=(14, 6),
        lam_kwargs=None,
        title=None,
        axs=None,
    ):
        # TODO: Add explicit figsize argument.
        base_size = 4
        n_params = self.n_params if self.n_params <= max_np else max_np
        grid_plot = closest_factors(n_params)

        if axs is None:
            fig, axs = plt.subplots(
                grid_plot[0],
                grid_plot[1],
                figsize=(grid_plot[1] * (base_size + 2), grid_plot[0] * base_size)
                if figsize is None
                else figsize,
            )

        lam_true = set_shape(lam_true, (1, -1)) if lam_true is not None else lam_true
        lam_kwargs = {} if lam_kwargs is None else lam_kwargs
        for i, ax in enumerate(axs.flat):
            plot_args = dict(param_idx=i, lam_true=lam_true, ax=ax)
            plot_args.update(lam_kwargs)
            self.plot_L(**plot_args)

        if fig is None:
            fig = ax.get_figure()

        fig.suptitle(self._parse_title(lam_true=lam_true, title=title))

        return axs

    def _parse_title(
        self,
        result=None,
        title=None,
        lam_true=None,
    ):
        """
        Parse title for plots

        Extends DCIProblem _parse title by adding MUD point.
        Note sets result to parse title for to result set by class if none
        passed by call (for calls from sub-classes).
        """
        result = self.result if result is None else result
        if lam_true is not None:
            mud_point = get_df(result, "lam_MUD", size=self.n_params)[0]
            l2_err = np.linalg.norm(lam_true - mud_point)
            title = title if title is not None else ""
            title = (
                title + 
                "$||\lambda^{{\dagger}} - \lambda^{{MUD}}||_{{\ell_2}}$"
                + f" = {l2_err:.3f},  "
            )
        title = super()._parse_title(result=result, title=title)

        return title
