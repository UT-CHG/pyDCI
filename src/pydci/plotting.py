import pdb

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

sns.set_style("darkgrid")  # set the default seaborn style for our plots

# Add True State Data to the plot
bright_colors = [
    "blue",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
interval_colors = sns.color_palette("muted", n_colors=50)

# TODO: Matplotlib plotting options
# plt.backend = "Agg"
# plt.rcParams["mathtext.fontset"] = "stix"
# plt.rcParams["font.family"] = "STIXGeneral"
# plt.rcParams["figure.figsize"] = (10, 10)
# plt.rcParams["font.size"] = 16
# plt.rcParams["text.usetex"] = True
# plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
# plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
#
# params = {
#     "axes.labelsize": 6,
#     "axes.titlesize": 6,
#     "xtick.labelsize": 6,
#     "ytick.labelsize": 6,
#     "axes.titlepad": 1,
#     "axes.labelpad": 1,
#     "font.size": 12,
# }
# plt.rcParams.update(params)

def _parse_title(
    self,
    prob,
    iteration,
    shift="",
):
    """
    Parse Title
    """
    action = prob.dfs["results"][iteration]["action"].values[0]
    kl = prob.dfs["results"][iteration]["kl"].values[0]
    l2 = prob.dfs["results"][iteration]["l2_err"].values[0]
    e_r = prob.mud_res[iteration]["best"].expected_ratio()
    title = (
        f"{shift}{iteration}: ({action}) - "
        + f"$\mathbb{{E}}(r)$= {e_r:.3f}, "
        + f"$\mathcal{{D}}_{{KL}}$= {kl:.3f}, "
        + "$||\lambda^{{\dagger}} - \lambda^{{MUD}}||_{{\ell_2}}$"
        + f" = {l2:.3f}"
    )

    return title

def plot_state_data(
    dfs,
    plot_measurements=True,
    samples=None,
    n_samples=10,
    state_idx=0,
    time_col="ts",
    meas_col=None,
    window_type=None,
    plot_shifts=True,
    markersize=100,
    ax=None,
    figsize= (9, 8),
):
    """
    Takes a list of observed data dataframes and plots the state at a certain
    index over time. If pf_dfs passed as well, pf_dfs are plotted as well.
    """

    labels = []
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot each column (state) of data on a separate subplot
    for iteration, df in enumerate(dfs):
        n_ts = len(df)
        n_states = len([x for x in df.columns if x.startswith('q_lam_true')])

        sns.lineplot(
            x="ts",
            y=f"q_lam_true_{state_idx}",
            ax=ax,
            color="blue",
            data=df,
            linewidth=2,
            marker='o',
            label="True State",
        )
        # Add Measurement Data to the plot
        if plot_measurements:
            sns.scatterplot(
                x="ts",
                y=f"q_lam_obs_{state_idx}",
                ax=ax,
                color="black",
                data=df,
                s=markersize,
                marker="*",
                label="Measurements",
                zorder=10,
            )
        # Add Push Forward Data to the plot
        if samples is not None:
            cols = [f'q_lam_{i}' for i in range(n_states * n_ts)]
            max_samples = len(samples[iteration])
            to_plot = n_samples if n_samples < max_samples else n_samples
            rand_idxs = random.choices(range(max_samples), k=to_plot)
            for i, sample in enumerate(
                    samples[iteration][cols].loc[rand_idxs].iterrows()):
                sample_state_data = np.array([df['ts'].values, np.array(
                    sample[1]).reshape(2,2)[:, state_idx]]).T
                label = None if i != (to_plot - 1) else 'Samples'
                sns.lineplot(
                    x="ts",
                    y=f'q_lam_{state_idx}',
                    legend=False,
                    ax=ax,
                    color="purple",
                    data=pd.DataFrame(sample_state_data,
                                      columns=['ts', f'q_lam_{state_idx}']),
                    alpha=0.2,
                    marker='o',
                    label=label,
                )
            #TODO: implement best and worst plots if columns present
            # sns.lineplot(
            #     x="ts",
            #     y=f"best_{i}",
            #     ax=axes[i],
            #     data=state_df,
            #     color="purple",
            #     linestyle="--",
            #     alpha=0.5,
            #     label="Best Sample",
            # )
            # sns.lineplot(
            #     x="ts",
            #     y=f"worst_{i}",
            #     ax=axes[i],
            #     color="purple",
            #     data=state_df,
            #     linestyle=":",
            #     alpha=0.5,
            #     label="Worst Sample",
            # )

        if window_type == 'line':
            ax.axvline(
                df['ts'].min(),
                linestyle="--",
                color="green",
                alpha=1,
                label=None,
            )
        elif window_type == 'rectangle':
            xmin = df['ts'].min()
            xmax = df['ts'].max()
            ymin, ymax = ax.get_ylim()
            rect = Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, linewidth=0, alpha=0.3)
            rect.set_facecolor(interval_colors[i])
            ax.add_patch(rect)


        # Add Shifts as vertical lines to the plot
        if plot_shifts:
            max_si = df['shift_idx'].max()
            for si, sd in df[df['shift_idx'] > 0].groupby('shift_idx'):
                label = None if not ((si == max_si) and
                                     (iteration == len(dfs))) else 'Shift'
                ax.axvline(
                    x=sd['ts'].min(),
                    linewidth=3,
                    color="orange",
                    label=label
                )
    if window_type == 'line':
        ax.axvline(
            df['ts'].max(),
            linestyle="--",
            color="green",
            alpha=1,
            label="Time Interval",
        )
    ax.legend(fontsize=12)
    ax.set_title(f"State {state_idx} Temporal Evolution")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(f"State {state_idx}")

    plt.tight_layout()

    return ax


def plot_parameters(
    self,
    prob,
    iterations=None,
    param_idxs=None,
    nrows=2,
    ncols=4,
    plot_initial=False,
    plot_legend=True,
):
    sns.set_style("darkgrid")

    figsize = {"1": (20, 6), "2": (20, 10), "3": (20, 12), "4": (20, 14)}

    if iterations is None:
        iterations = np.arange(prob.iteration)[: nrows * ncols]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize[str(nrows)])

    if nrows == 1:
        axes = np.array([axes])

    param_idxs = range(prob.n_params) if param_idxs is None else param_idxs
    number_parameters = len(param_idxs)
    bright_colors = sns.color_palette("bright", n_colors=number_parameters)
    deep_colors = sns.color_palette("deep", n_colors=number_parameters)

    ylims = []
    for i, ax in enumerate(axes.flat):
        # Plot initial distributions for iteration
        lambda_labels = [f"$\pi^{{up}}_{{\lambda_{j}}}$" for j in param_idxs]
        [
            sns.kdeplot(
                data=prob.dfs["samples"][iterations[i]],
                x=f"lam_{idx}",
                ax=ax,
                fill=True,
                color=bright_colors[j],
                label=lambda_labels[j],
                weights="ratio",
            )
            for j, idx in enumerate(param_idxs)
        ]
        if plot_initial:
            lambda_labels += [f"$\pi^{{in}}_{{\lambda_{j}}}$" for j in param_idxs]
            [
                sns.kdeplot(
                    data=prob.dfs["samples"][iterations[i]],
                    x=f"lam_{idx}",
                    ax=ax,
                    fill=True,
                    color=bright_colors[j],
                    linestyle=":",
                    label=lambda_labels[len(param_idxs) + j],
                    weights="weights",
                )
                for j, idx in enumerate(param_idxs)
            ]

        # Generate vertical lines for true values
        unique_true = prob.dfs["state"][iterations[i]].drop_duplicates(
            subset="shift_idx"
        )
        true_vals = prob._get_df(unique_true, "true_param")
        true_labels = [
            f"$\lambda^{{\dagger}}_{j} = {true_vals[0][j]}$" for j in param_idxs
        ]
        [
            ax.axvline(
                x=true_vals[0][idx], linestyle="--", linewidth=1, color=deep_colors[j]
            )
            for j, idx in enumerate(param_idxs)
        ]
        # Plot or indiicate if shift occured in interval
        shift = "" if len(true_vals) == 1 else "SHIFT at "

        # Set plot specifications
        ax.set_xlabel(r"$\lambda$", fontsize=12)
        ax.set_title(self._parse_title(prob, iterations[i], shift=shift), fontsize=20)
        ylims.append(ax.get_ylim()[1])
        if plot_legend:
            ax.legend(
                title="MUDS",
                labels=lambda_labels + true_labels,
                fontsize=12,
                title_fontsize=12,
                loc="upper right",
            )

    ylim = 1.1 * np.mean(np.array(ylims))
    for i, ax in enumerate(axes.flat):
        ax.set_ylim([-0.1, 0.9 * ylim])

    plt.tight_layout()

    return axes


def plot_ratios(self, prob, iterations=[0]):
    """
    Plot Expected Ratio for Best MUD estimate
    """
    # TODO: Check this plot -> observed should be N(0,1) and doesn't look like it

    best_muds = [
        res["best"] for idx, res in enumerate(prob.mud_res) if idx in iterations
    ]
    axes = []
    for i, mud_prob in enumerate(best_muds):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.color_palette("bright")
        sns.kdeplot(data=mud_prob._ob, fill=True, ax=ax)
        sns.kdeplot(data=mud_prob._pr, fill=True, ax=ax)
        sns.kdeplot(data=mud_prob._r, fill=True, ax=ax)
        plt.legend(
            labels=[
                r"$\pi_{obs}(\lambda)$",
                r"$\pi_{predict}(\lambda)$",
                r"$r(\lambda)$",
            ]
        )
        true_vals = prob._get_df(
            prob.dfs["state"][iterations[i]].drop_duplicates(subset="shift_idx"),
            "true_param",
        )
        shift = "" if len(true_vals) == 1 else "SHIFT at "
        ax.set_title(self._parse_title(prob, iterations[i], shift=shift), fontsize=20)
        axes.append(ax)

    return axes


def plot_expected_ratios(self, prob):
    """
    Plot E(r) over each iteration
    """
    e_r_stats = pd.concat(prob.dfs["results"], axis=0)
    sns.color_palette("bright")
    plt.figure(figsize=(16, 6))
    ax = sns.lineplot(
        x="iteration",
        y="e_r",
        data=e_r_stats,
        color="red",
        marker="*",
        markersize=15,
    )
    plt.axhline(y=1, color="black", linestyle="--", label="Predict Assumption")
    plt.ylabel(r"$\mathbb{E}(r)$")
    plt.xlabel("Iteration")
    plt.title(r"$\mathbb{E}(r)$ per Iteration")
    plt.ylim([0.0, 1.5])
    plt.legend(loc="best")

    return ax


def plot_ratio_scatter(self, prob, iterations=None):
    """
    Visualize distirbution of ratios (obs/pr) over each iteration.
    """
    plt.figure(figsize=(20, 8))
    samples_df = prob.get_full_df("samples", iterations=iterations)
    number_intervals = len(samples_df["iteration"].unique())
    colors = sns.color_palette("bright", n_colors=number_intervals)
    ax = sns.scatterplot(
        x="iteration",
        y="ratio",
        hue="iteration",
        palette=colors,
        data=samples_df,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("Sample ratios ($\pi^{{obs}}/\pi^{{pr}}$) per Iteration")

    return ax


def plot_ratio_dist(self, prob, iterations=None, ncols=None, legend=False):
    """
    Visualie distribution of ratios (obs/pr) over each iteration.
    """
    samples_df = prob.get_full_df("samples", iterations=iterations)
    iterations = samples_df["iteration"].unique()
    number_intervals = len(iterations)

    figsize = (12, 10) if number_intervals <= 5 else (20, 12)
    # layout = True if number_intervals <= 5 else False
    font_size = 14 if number_intervals <= 5 else 18
    colors = sns.color_palette("bright", n_colors=number_intervals)

    plt.figure(figsize=figsize)

    ax = sns.FacetGrid(
        data=samples_df,
        col="iteration",
        hue="iteration",
        hue_order=iterations,
        palette=colors,
        col_wrap=ncols,
    )

    # Map the distplot function to the 'Residuals' column
    ax.map(sns.kdeplot, "ratio", fill=True, bw_adjust=1)
    ax.set(xlim=(-5, 10), ylim=(0, 1.5))
    ax.set_axis_labels(r"$r$", r"$\pi(r)$", fontsize=12)

    # display the plot
    ax.fig.subplots_adjust(top=0.9)
    ax.fig.suptitle("Ratio Distributions per interval", fontsize=font_size)

    if legend:
        ax.add_legend(title="Time Intervals", loc="lower right")

    _ = plt.tight_layout()

    return ax


def state_line_plot(
    df, state, mask=None, x_col=None, ax=None, figsize=(8, 8), **kwargs
):
    """
    Plot the X and Y data on two subplots, and add a rectangle for
    each interval to each subplot.
    """
    # Set up the figure and axes
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.color_palette("bright")

    sns.lineplot(
        x=df.index if x_col is None else x_col,
        y=state,
        ax=ax,
        color="blue",
        data=df,
        linewidth=2,
        label="State",
    )


def state_scatter_plot(
    df, state, mask=None, x_col=None, ax=None, figsize=(8, 8), **kwargs
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
        x=df.index if x_col is None else x_col,
        y=state,
        ax=ax,
        color="blue",
        data=df,
        label="State",
    )
