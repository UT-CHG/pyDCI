"""
pyDCI plotting functions

TODO:
 - Reconcile plotting functions here with ones in the classes
"""

# System imports
import random
from pathlib import Path
import importlib

# Math imports
import pandas as pd
import numpy as np
import random
from scipy.stats import uniform
from pandas import DataFrame

# Plotting imports
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Sequence, Iterable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
from matplotlib.legend import Legend
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib import ticker
from matplotlib.patches import Rectangle
from datetime import timedelta
from matplotlib.dates import HourLocator, DateFormatter

import matplotlib.axes._axes as axes
from enum import Enum, auto
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as Axes


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
DEF_RC_PARAMS = {
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "axes.labelsize": 16,
    "axes.titlesize": 22,
    "legend.fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlepad": 1,
    "axes.labelpad": 1,
    "font.size": 16,
    "savefig.facecolor": "white",
}
plt.rcParams.update(DEF_RC_PARAMS)


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
    figsize=(9, 8),
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
        n_states = len([x for x in df.columns if x.startswith("q_lam_true")])

        sns.lineplot(
            x="ts",
            y=f"q_lam_true_{state_idx}",
            ax=ax,
            color="blue",
            data=df,
            linewidth=2,
            marker="o",
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
            cols = [f"q_lam_{i}" for i in range(n_states * n_ts)]
            max_samples = len(samples[iteration])
            to_plot = n_samples if n_samples < max_samples else n_samples
            rand_idxs = random.choices(range(max_samples), k=to_plot)
            for i, sample in enumerate(
                samples[iteration][cols].loc[rand_idxs].iterrows()
            ):
                sample_state_data = np.array(
                    [df["ts"].values, np.array(sample[1]).reshape(2, 2)[:, state_idx]]
                ).T
                label = None if i != (to_plot - 1) else "Samples"
                sns.lineplot(
                    x="ts",
                    y=f"q_lam_{state_idx}",
                    legend=False,
                    ax=ax,
                    color="purple",
                    data=pd.DataFrame(
                        sample_state_data, columns=["ts", f"q_lam_{state_idx}"]
                    ),
                    alpha=0.2,
                    marker="o",
                    label=label,
                )
            # TODO: implement best and worst plots if columns present
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

        if window_type == "line":
            ax.axvline(
                df["ts"].min(),
                linestyle="--",
                color="green",
                alpha=1,
                label=None,
            )
        elif window_type == "rectangle":
            xmin = df["ts"].min()
            xmax = df["ts"].max()
            ymin, ymax = ax.get_ylim()
            rect = Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=0, alpha=0.3
            )
            rect.set_facecolor(interval_colors[i])
            ax.add_patch(rect)

        # Add Shifts as vertical lines to the plot
        if plot_shifts:
            max_si = df["shift_idx"].max()
            for si, sd in df[df["shift_idx"] > 0].groupby("shift_idx"):
                label = (
                    None
                    if not ((si == max_si) and (iteration == len(dfs)))
                    else "Shift"
                )
                ax.axvline(x=sd["ts"].min(), linewidth=3, color="orange", label=label)
    if window_type == "line":
        ax.axvline(
            df["ts"].max(),
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


def add_plot_settings(
    ax: Axes, 
    title: str, 
    label_pads: List[int],
    x_lim: List[int],
    y_lim: List[int], 
    label_sizes: List[int],
    loc: List[float], 
    axis_labels: List[str], 
     tick_label_size: int,
    title_pad: int, 
    title_font_size: int
) -> None:
    """
    Configures the plot settings for the provided Axes object, including the title,
    tick locator intervals, label padding, axis labels, and tick label size.

    Parameters
    ----------
    ax : Axes
        The Axes object to apply settings to.
    title : str
        The title of the plot.
    x_lim : List[int]
        A list containing the x-axis limits.
    y_lim : List[int] 
        A list containing the y-axis limits.
    label_pads : List[int]
        A list containing the padding for the x-axis and y-axis labels.
    label_sizes : List[int]
        A list containing the font sizes for the x-axis and y-axis labels.
    loc : List[float]
        A list containing the locator interval for the x-axis and y-axis.
    axis_labels : List[str]
        A list containing the labels for the x-axis and y-axis.
    tick_label_size : int
        The font size for the tick labels.
    title_pad : int
        The padding for the plot title.
    title_font_size : int

    Returns
    -------
    None

    """
        
    ax.set_xlim(*x_lim) if x_lim is not None else None
    ax.set_ylim(*y_lim) if y_lim is not None else None

        
    ax.xaxis.set_major_locator(mticker.MultipleLocator(loc[0])) if loc[0] is not None else None
    ax.yaxis.set_major_locator(mticker.MultipleLocator(loc[1])) if loc[1] is not None else None
    
    ax.tick_params(axis='x', labelsize=tick_label_size)
    ax.tick_params(axis='y', labelsize=tick_label_size)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.xaxis.labelpad = label_pads[0]
    ax.yaxis.labelpad = label_pads[1]
    
    ax.set_xlabel(axis_labels[0], fontsize=label_sizes[0], fontweight='bold')
    ax.set_ylabel(axis_labels[1], fontsize=label_sizes[1], fontweight='bold')
    
    if title is not None:
        ax.set_title(title, fontsize=title_font_size, fontweight='bold', pad=title_pad)


def plot_interval_lines(data: pd.DataFrame, 
                        plot_intervals: Optional[List[Tuple[str, Dict[str, Any], List[Tuple[int, int]]]]] = None, 
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plots vertical lines on an Axes at the intervals specified in the data.

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas DataFrame containing the data to be plotted. Must contain a time column.
    plot_intervals : Optional[List[Tuple[str, Dict[str, Any], List[Tuple[int, int]]]]], optional
        A list of tuples where each tuple contains a label string, a dictionary of arguments to 
        pass to axvline, and a list of tuples, each of which represents an interval with start and 
        end indexes in the time column. By default, it is None, which means no intervals are plotted.
    ax : Optional[matplotlib.pyplot.Axes], optional
        The matplotlib Axes object where the lines will be plotted. If None, it will be created.

    Returns
    -------
    matplotlib.pyplot.Axes
        The matplotlib Axes object with the interval lines plotted.

    """
    if ax is None:
        ax = plt.gca()

    time_col = 'ts' if 'ts' in data.columns else 'time'
    plot_intervals = plot_intervals or []

    for _, args, intervals in plot_intervals:
        for interval in intervals:
            args["label"] = None
            ax.axvline(data[time_col][interval[0]], linewidth=1, c="darkgoldenrod", zorder=1, **args)
        args["label"] = "Intervals"
        ax.axvline(data[time_col][intervals[-1][-1]], linewidth=1, c="darkgoldenrod", zorder=1, **args)
    
    return ax


def create_legend(ax: Axes,
                  colors: List[str],
                  styles: List[str],
                  labels: List[str],
                  location: str = 'upper left',
                  font_size: int = 30,
                  line_width: int = 8
                  ) -> None:
    """
    Creates a custom legend for the given Axes object with specified colors,
    line styles, and labels.

    Parameters
    ----------
    ax : maxes.Axes
        The Axes object to which the legend will be added.
    colors : List[str]
        A list of color specifications for the legend lines.
    styles : List[str]
        A list of line styles for the legend lines.
    labels : List[str]
        A list of label strings for the legend entries.
    location : str, optional
        The location of the legend on the Axes. Defaults to 'upper left'.
    font_size : int, optional
        The font size of the legend labels. Defaults to 30.
    line_width : int, optional
        The line width of the legend lines. Defaults to 8.
    Returns
    -------
    None

    """
    # Create Line2D objects for the legend.
    line_handles = [mlines.Line2D([], [], color=color, linestyle=style) 
                    for color, style in zip(colors, styles)]


    # Create the legend on the Axes.
    legend = ax.legend(handles=line_handles, labels=labels, handlelength=1.8,
                       loc=location, frameon=True)


    # Set line properties in the legend.
    for idx, (line,  text, color) in enumerate(zip(legend.get_lines(), legend.get_texts(),colors)):

        
        if line.get_linestyle() == "None":
            line.set_marker('.')
            line.set_markersize(25)
            line.set_linewidth(0)   
        else:
            line.set_linewidth(line_width)
            
        line.set_color(color)  # Set the color dynamically.
        text.set_fontsize(font_size)
        text.set_fontweight('bold')

    # Directly set the linestyle for the specific line if it's known.
    legend.get_lines()[-1].set_linestyle(styles[-1])


def create_colorbar(
    ax: Axes,
    data: pd.Series,
    label: str,
    pad: float = 0.025,
    label_size: int = 14,
    label_pad: int = 42,
    width: int = 0,
    length: int = 0,
    direction: str = "inout",
    y: float = 1.07,
    rotation: int = 0,
    font_size: int = 20,
) -> None:
    """
    Create a colorbar for a given axes object.

    Parameters
    ----------
    ax : Axes
        The axes object to which the colorbar will be added.
    data : pd.Series
        The data to be used for the colorbar.
    label : str
        The label for the colorbar.
    pad : float, optional
        The padding between the colorbar and the axes, by default 0.025
    label_size : int, optional
        The font size of the colorbar label, by default 14
    label_pad : int, optional
        The padding between the colorbar and its label, by default 42
    width : int, optional
        The width of the colorbar ticks, by default 0
    length : int, optional
        The length of the colorbar ticks, by default 0
    direction : str, optional
        The direction of the colorbar ticks, by default "inout"
    y : float, optional
        The y-coordinate of the colorbar, by default 1.07
    rotation : int, optional
        The rotation of the colorbar label, by default 0
    font_size : int, optional
        The font size of the colorbar ticks, by default 20

    Returns
    -------
    None
    """
    # Set colormap to viridis and normalize based on data
    norm = Normalize(data.min(), data.max())
    cmap = plt.cm.viridis
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=pad)
    cbar.ax.tick_params(
        labelsize=label_size, width=width, length=length, direction=direction
    )
    cbar.set_label(
        label=label, size=label_size, weight="bold", labelpad=label_pad, y=y, rotation=rotation
    )
    # cbar.ax.tick_params(labelsize=font_size)

def format_time_xlabel(ax, start_time, end_time):
    if (end_time - start_time) <= timedelta(days=1):
        ax.xaxis.set_major_locator(HourLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%d-%H:%M'))
        ax.set_xlabel(f'Time ({start_time.strftime("%b %d, %Y")})')
    elif start_time.month == end_time.month:
        ax.xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
        ax.set_xlabel(f'Time ({start_time.strftime("%b %Y")})')
    else:
        ax.set_xlabel('Time')
