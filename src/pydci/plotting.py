import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats  # for relative entropy (KL divergence)

sns.set_style("darkgrid")  # set the default seaborn style for our plots


class Plotter:
    def __init__(self) -> None:
        pass

    def plot_parameters(
        self,
        res,
        iterations,
        model,
        number_parameters=4,
        nrows=2,
        ncols=4,
        save=False,
        file_name=None,
    ):
        sns.set_style("darkgrid")

        figsize = {"1": (20, 6), "2": (20, 10), "3": (20, 12), "4": (20, 14)}

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize[str(nrows)])

        if nrows == 1:
            axes = np.array([axes])

        bright_colors = sns.color_palette("bright", n_colors=number_parameters)
        deep_colors = sns.color_palette("deep", n_colors=number_parameters)

        xlim = {"seir": [0, 0.7], "rlc": [50, 300], "lv": [0, 2]}
        ylim = {"seir": [0, 50], "rlc": [0, 0.3], "lv": [0, 14]}

        best_muds = [mud_problem["best"] for mud_problem in res["mud_probs"]]
        kls = [stats.entropy(mud_prob._ob, mud_prob._pr) for mud_prob in best_muds]

        for i, ax in enumerate(axes.flat):
            # Generate data
            data = res["spatio_temporal_probs"][iterations[i]].lam
            true = res["spatio_temporal_probs"][iterations[i]].lam_ref

            # Generate labels
            true_labels = [
                f"$\lambda^{{\dagger}}_{j} = {true[j]}$"
                for j in range(number_parameters)
            ]
            lambda_labels = [f"$\pi_{{\lambda_{x}}}$" for x in range(number_parameters)]

            data_df = pd.DataFrame(data, columns=lambda_labels)

            # Generate Plots
            [
                ax.axvline(
                    x=true_param, linestyle="--", linewidth=1, color=deep_colors[k]
                )
                for k, true_param in enumerate(true)
            ]
            [
                sns.kdeplot(
                    data=data_df[label],
                    ax=ax,
                    fill=True,
                    color=bright_colors[i],
                    label=label,
                )
                for i, label in enumerate(lambda_labels)
            ]

            # Set plot specifications
            ax.set_xlabel(r"$\lambda$", fontsize=12)
            ax.set_title(
                f"Time Interval: {iterations[i]} $\mathcal{{D}}_{{KL}}$= {kls[iterations[i]]:.4f}",
                fontsize=20,
            )
            ax.set_xlim(xlim[model])
            ax.set_ylim(ylim[model])
            ax.legend(
                title="MUDS",
                labels=true_labels + lambda_labels,
                fontsize=12,
                title_fontsize=12,
                loc="upper right",
            )

        plt.tight_layout()

        if save:
            plt.savefig(file_name)

        plt.show()

    # ! TODO: ADD PUSH FORWARD DATA TO STATE PLOTS
    def plot_state_data(
        self,
        state_df,
        measurement_df=None,
        push_forward_df=None,
        window_type=None,
        shifts=None,
        markersize=100,
        save=False,
        file_name=None,
    ):
        """
        Plot the X and Y data on two subplots, and add a rectangle for
        each interval to each subplot.
        """
        # Set up the figure and axes
        number_states = sum([col.startswith("X_") for col in state_df.columns])
        figsize = (18, 8) if number_states <= 2 else (18, 12)

        fig, axes = plt.subplots(nrows=number_states, ncols=1, figsize=figsize)
        sns.despine(offset=5)

        # Plot each column (state) of data on a separate subplot
        for i in range(number_states):
            column_name = f"X_{i}"
            column_df = state_df[["Times", column_name, "Interval"]].copy()

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
                x="Times",
                y=column_name,
                ax=axes[i],
                color=bright_colors[i],
                data=state_df,
                linewidth=2,
                label="True State",
            )

            # Add Measurement Data to the plot
            if measurement_df is not None:
                sns.scatterplot(
                    x="Times",
                    y=column_name,
                    ax=axes[i],
                    color="black",
                    data=measurement_df,
                    s=markersize,
                    marker="*",
                    label="Measurements",
                    zorder=10,
                )

            # Add Push Forward Data to the plot
            if push_forward_df is not None:
                sampled_forward_runs = push_forward_df[::100, :, i]
                number_runs = sampled_forward_runs.shape[0]
                sampled_forward_runs = pd.DataFrame(sampled_forward_runs)
                sns.lineplot(
                    data=sampled_forward_runs.T,
                    legend=False,
                    color="purple",
                    alpha=0.2,
                    ax=axes[i],
                )

            # Add Shifts as vertical lines to the plot
            if shifts:
                for j, shift in enumerate(shifts):
                    axes[i].axvline(
                        x=shift[0],
                        linewidth=3,
                        color="orange",
                        label=f"Shift" if j == 0 else None,
                    )

            # ! TODO: HACKY MUST BE REMOVED
            non_sir = (
                column_df[column_name].min() - 0.5,
                column_df[column_name].max() + 3.5,
            )
            sir = column_df[column_name].min() - 0.5, column_df[column_name].max() + 0.5
            ylim = non_sir if number_states <= 2 else sir

            axes[i].set_xlim(
                left=column_df["Times"].min(), right=column_df["Times"].max()
            )
            axes[i].set_ylim(bottom=ylim[0], top=ylim[1])
            axes[i].legend(loc="upper right", fontsize=12)
            axes[i].set_title(f"State {i} Temporal Evolution")

        plt.tight_layout()

        if save:
            plt.savefig(file_name)

        plt.show()

    def plot_vertical_lines(self, df, ax):
        """
        Add vertical lines for each interval to the given axis.
        """

        # Create a DataFrame with the start and end times of each interval
        intervals = df["Interval"].unique()
        interval_starts = [df[df["Interval"] == i]["Times"].min() for i in intervals]
        interval_ends = [df[df["Interval"] == i]["Times"].max() for i in intervals]
        interval_df = pd.DataFrame(
            {"Interval": intervals, "Start": interval_starts, "End": interval_ends}
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
        num_intervals = len(df["Interval"].unique())
        colors = sns.color_palette("muted", n_colors=num_intervals)

        for i, (_, state_df_interval) in enumerate(df.groupby("Interval")):
            xmin, xmax = (
                state_df_interval["Times"].min(),
                state_df_interval["Times"].max(),
            )
            ymin, ymax = df[column_name].min() - 0.5, df[column_name].max() + 3.5
            rect = Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=0, alpha=0.3
            )
            rect.set_facecolor(colors[i])
            ax.add_patch(rect)

    def plot_residual_scatter(self, residuals_df, save=False, file_name=None):
        plt.figure(figsize=(20, 8))
        number_intervals = len(residuals_df["Interval"].unique())
        colors = sns.color_palette("bright", n_colors=number_intervals)
        plot = sns.scatterplot(
            x="Interval",
            y="Residuals",
            hue="Interval",
            palette=colors,
            data=residuals_df,
        )
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
        plot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title("Residuals per Interval")

        if save:
            plt.savefig(file_name)

        plt.show()

    def plot_residual_dist(self, residuals_df, save=False, file_name=None):
        number_intervals = len(residuals_df["Interval"].unique())

        figsize = (12, 10) if number_intervals <= 5 else (20, 12)
        layout = True if number_intervals <= 5 else False
        font_size = 14 if number_intervals <= 5 else 18
        colors = sns.color_palette("bright", n_colors=number_intervals)

        plt.figure(figsize=figsize)

        g = sns.FacetGrid(
            data=residuals_df,
            col="Interval",
            hue="Interval",
            hue_order=residuals_df["Interval"].unique(),
            palette=colors,
            col_wrap=5,
        )

        # Map the distplot function to the 'Residuals' column
        g.map(sns.kdeplot, "Residuals", fill=True, bw_adjust=1)
        g.set(xlim=(-5, 10), ylim=(0, 1.5))
        g.set_axis_labels(r"$r$", r"$\pi(r)$", fontsize=12)

        # display the plot
        g.fig.subplots_adjust(top=0.9)
        g.add_legend(title="Time Intervals")
        g.fig.suptitle("Residual Distributions per Interval", fontsize=font_size)

        if layout:
            plt.tight_layout()

        # Show the plot
        if save:
            plt.savefig(file_name)

        plt.show()

    def plot_expected_ratios(self, df, save=False, file_name=None):
        df = df.copy()
        sns.color_palette("bright")
        plt.figure(figsize=(16, 6))
        plot = sns.lineplot(
            x="Interval",
            y="Residuals",
            data=df.Mean,
            color="red",
            marker="*",
            markersize=15,
        )
        plt.axhline(y=1, color="black", linestyle="--", label="Predict Assumption")
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
        plt.ylabel(r"$E(r)$")
        plt.title(r"$E(r)$ per Interval")
        plt.legend(loc="best")

        if save:
            plt.savefig(file_name)
        plt.show()

    # ! TODO: MAKE THIS A 2 COLUMN PLOT
    def plot_ratios(self, res):
        best_muds = [mud_problem["best"] for mud_problem in res["mud_probs"]]
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
            ax.set_title(f"Time Interval {i}")
            plt.show()
