import pdb
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("darkgrid")  # set the default seaborn style for our plots


class Plotter:
    def __init__(self) -> None:
        pass

    def _parse_title(
        self,
        prob,
        iteration,
        shift='',
      ):
        """
        Parse Title
        """
        action = prob.dfs['results'][iteration]['action'].values[0]
        kl = prob.dfs['results'][iteration]['kl'].values[0]
        l2 = prob.dfs['results'][iteration]['l2_err'].values[0]
        e_r = prob.mud_res[iteration]['best'].expected_ratio()
        title = (f"{shift}{iteration}: ({action}) - " +
                 f"$\mathbb{{E}}(r)$= {e_r:.3f}, " +
                 f"$\mathcal{{D}}_{{KL}}$= {kl:.3f}, " +
                 "$||\lambda^{{\dagger}} - \lambda^{{MUD}}||_{{\ell_2}}$" +
                 f" = {l2:.3f}")

        return title

    def plot_state_data(
        self,
        prob,
        plot_measurements=True,
        plot_push_forwards=True,
        window_type=None,
        plot_shifts=True,
        markersize=100,
        save=False,
        file_name=None,
    ):
        """
        Plot the X and Y data on two subplots, and add a rectangle for
        each interval to each subplot.
        """
        # Set up the figure and axes
        state_df = pd.concat(prob.dfs['state'], axis=0)
        number_states = prob.n_states
        figsize = (18, 8) if number_states <= 2 else (18, 12)

        fig, axes = plt.subplots(nrows=number_states, ncols=1, figsize=figsize)
        sns.despine(offset=5)

        # Plot each column (state) of data on a separate subplot
        for i in range(number_states):
            column_name = f"true_vals_{i}"
            obs_col_name = f"obs_vals_{i}"
            column_df = state_df[["ts", column_name,
                                  obs_col_name, "iteration"]].copy()

            # Add time intervals as vertical lines to the plot
            if window_type == "line":
                self.plot_vertical_lines(column_df, axes[i])

            # Add time intervals as rectangles to the plot
            if window_type == "rectangle":
                self.plot_rectangles(column_df, column_name, axes[i])

            # Add True State Data to the plot
            sns.color_palette("bright")
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

            sns.lineplot(
                x="ts",
                y=column_name,
                ax=axes[i],
                color=bright_colors[i],
                data=state_df,
                linewidth=2,
                label="True State",
            )

            # Add Measurement Data to the plot
            if plot_measurements:
                sns.scatterplot(
                    x="ts",
                    y=obs_col_name,
                    ax=axes[i],
                    color="black",
                    data=state_df,
                    s=markersize,
                    marker="*",
                    label="Measurements",
                    zorder=10,
                )

            # Add Push Forward Data to the plot
            if plot_push_forwards:
                sns.lineplot(
                    x="ts",
                    y=f"best_{i}",
                    ax=axes[i],
                    data=state_df,
                    color='purple',
                    linestyle='--',
                    alpha=0.5,
                    label="Best Sample",
                )
                sns.lineplot(
                    x="ts",
                    y=f"worst_{i}",
                    ax=axes[i],
                    color='purple',
                    data=state_df,
                    linestyle=':',
                    alpha=0.5,
                    label="Worst Sample",
                )
                rand_cols = [x for x in state_df.columns if
                             x.startswith('random_') and x.endswith(f'_{i}')]
                for rc in rand_cols:
                    sns.lineplot(
                        x="ts",
                        y=rc,
                        legend=False,
                        ax=axes[i],
                        color='purple',
                        data=state_df,
                        alpha=0.2
                    )

            # Add Shifts as vertical lines to the plot
            if plot_shifts:
                for j, shift in enumerate(prob.param_shifts.keys()):
                    axes[i].axvline(
                        x=shift,
                        linewidth=3,
                        color="orange",
                        label="Shift" if j == 0 else None,
                    )

            axes[i].legend(loc="upper right", fontsize=12)
            axes[i].set_title(f"State {i} Temporal Evolution")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel(f"State {i}")

        plt.tight_layout()

        return axes

    def plot_vertical_lines(self, df, ax):
        """
        Add vertical lines for each interval to the given axis.
        """

        # Create a DataFrame with the start and end times of each interval
        intervals = df["iteration"].unique()
        interval_starts = [df[df["iteration"] == i]["ts"].min() for i in intervals]
        interval_ends = [df[df["iteration"] == i]["ts"].max() for i in intervals]
        interval_df = pd.DataFrame(
            {"iteration": intervals, "Start": interval_starts, "End": interval_ends}
        )

        # Plot vertical lines at the start of each interval
        for i, row in interval_df.iterrows():
            if i == interval_df.shape[0] - 1:
                ax.axvline(
                    row["Start"],
                    linestyle="--",
                    color="green",
                    alpha=1,
                    label="Time Interval",
                )
            ax.axvline(row["Start"], linestyle="--", color="green", alpha=1)

    def plot_rectangles(self, df, column_name, ax):
        """
        Add a rectangle for each interval to the specified axis object.
        """
        num_intervals = len(df["iteration"].unique())
        colors = sns.color_palette("muted", n_colors=num_intervals)

        for i, (_, state_df_interval) in enumerate(df.groupby("iteration")):
            xmin, xmax = (
                state_df_interval["ts"].min(),
                state_df_interval["ts"].max(),
            )
            ymin, ymax = df[column_name].min() - 0.5, df[column_name].max() + 3.5
            rect = Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=0, alpha=0.3
            )
            rect.set_facecolor(colors[i])
            ax.add_patch(rect)

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
            iterations = np.arange(prob.iteration)[:nrows*ncols]

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
            lambda_labels = [f"$\pi^{{up}}_{{\lambda_{j}}}$"
                             for j in param_idxs]
            [
                sns.kdeplot(
                    data=prob.dfs['samples'][iterations[i]],
                    x=f'lam_{idx}',
                    ax=ax,
                    fill=True,
                    color=bright_colors[j],
                    label=lambda_labels[j],
                    weights='ratio',
                )
                for j, idx in enumerate(param_idxs)
            ]
            if plot_initial:
                lambda_labels += [f"$\pi^{{in}}_{{\lambda_{j}}}$"
                                  for j in param_idxs]
                [
                    sns.kdeplot(
                        data=prob.dfs['samples'][iterations[i]],
                        x=f'lam_{idx}',
                        ax=ax,
                        fill=True,
                        color=bright_colors[j],
                        linestyle=':',
                        label=lambda_labels[len(param_idxs) + j],
                        weights='weights',
                    )
                    for j, idx in enumerate(param_idxs)
                ]

            # Generate vertical lines for true values
            unique_true = prob.dfs['state'][iterations[i]].drop_duplicates(
                    subset='shift_idx')
            true_vals = prob._get_df(unique_true, 'true_param')
            true_labels = [f"$\lambda^{{\dagger}}_{j} = {true_vals[0][j]}$"
                           for j in param_idxs]
            [
                ax.axvline(
                    x=true_vals[0][idx], linestyle="--",
                    linewidth=1, color=deep_colors[j]
                )
                for j, idx in enumerate(param_idxs)
            ]
            # Plot or indiicate if shift occured in interval
            shift = '' if len(true_vals) == 1 else 'SHIFT at '

            # Set plot specifications
            ax.set_xlabel(r"$\lambda$", fontsize=12)
            ax.set_title(self._parse_title(prob, iterations[i], shift=shift),
                         fontsize=20)
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

    def plot_ratios(self,
                    prob,
                    iterations=[0]):
        """
        Plot Expected Ratio for Best MUD estimate
        """
        # TODO: Check this plot -> observed should be N(0,1) and doesn't look like it

        best_muds = [res["best"] for idx, res in enumerate(prob.mud_res)
                     if idx in iterations]
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
                    prob.dfs['state'][iterations[i]].drop_duplicates(
                        subset='shift_idx'), 'true_param')
            shift = '' if len(true_vals) == 1 else 'SHIFT at '
            ax.set_title(self._parse_title(prob, iterations[i], shift=shift),
                         fontsize=20)
            axes.append(ax)

        return axes

    def plot_expected_ratios(self, prob):
        """
        Plot E(r) over each iteration
        """
        e_r_stats = pd.concat(prob.dfs['results'], axis=0)
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

    def plot_ratio_scatter(
        self,
        prob,
        iterations=None
    ):
        """
        Visualize distirbution of ratios (obs/pr) over each iteration.
        """
        plt.figure(figsize=(20, 8))
        samples_df = prob.get_full_df('samples', iterations=iterations)
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

    def plot_ratio_dist(
        self,
        prob,
        iterations=None,
        ncols=None,
        legend=False
    ):
        """
        Visualie distribution of ratios (obs/pr) over each iteration.
        """
        samples_df = prob.get_full_df('samples', iterations=iterations)
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
            ax.add_legend(title="Time Intervals", loc='lower right')

        _ = plt.tight_layout()

        return ax
