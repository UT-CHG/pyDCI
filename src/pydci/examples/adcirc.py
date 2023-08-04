import pdb
import pickle
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import alive_bar

from pydci import PCAMUDProblem
from pydci.log import disable_log, enable_log, logger
from pydci.utils import add_noise, get_df, put_df


def load_full_ds(ds_path: str):
    """
    Merge pilosov2023parameter ADCIRC dataset into to numpy arrays.
    Last array in index is the "true" signal chosen in pilosov2023parameter paper.
    This allows for us to run different experiment picking different "true" lambda
    parameters to see how the algorithm performs.
    """
    with open(ds_path, "rb") as fp:
        full_ds = pickle.load(fp)
    all_lam = np.vstack([full_ds["lam"], full_ds["lam_ref"].reshape(1, -1)])
    all_data = np.vstack([full_ds["data"], full_ds["true_vals"].reshape(1, -1)])

    return all_lam, all_data, full_ds["times"]


def build_ds(
    data=None,
    ds_path="si-inlet-full-ds.pickle",
    lam_true_idx=-1,
    std_dev=0.05,
    num_samples=None,
    seed=None,
    outpath=None,
):
    """
    Convert ADCIRC MUD pickle format to pydci pandas DataFrame format.
    """
    if data is None:
        all_lam, all_data, times = stack_full_ds(ds_path)
    else:
        all_lam, all_data, times = data

    total_num_samples = all_data.shape[0] - 1
    q_lam_dim = all_data.shape[1]
    lam_dim = all_lam.shape[1]
    if num_samples is None:
        num_samples = total_num_samples - 1

    measurements = add_noise(all_data[[lam_true_idx]], std_dev, seed=seed).reshape(
        -1, 1
    )
    data_df = pd.DataFrame(times, columns=["ts"])
    data_df["shift_idx"] = 0
    data_df["sample_flag"] = True
    for i, val in enumerate(all_lam[lam_true_idx]):
        data_df[f"lam_true_{i}"] = val
    data_df = put_df(data_df, "q_lam_true", all_data[lam_true_idx].reshape(-1, 1))
    data_df = put_df(data_df, "q_lam_obs", measurements)

    sample_idxs = list(range(len(all_lam)))
    lam_true_idx = len(all_lam) - 1 if lam_true_idx == -1 else lam_true_idx
    sample_idxs.remove(lam_true_idx)
    sample_idxs = np.random.choice(sample_idxs, num_samples, replace=False)

    lam_df = pd.DataFrame(
        all_lam[sample_idxs],
        columns=[f"lam_{i}" for i in range(lam_dim)],
    )
    q_lam_df = pd.DataFrame(
        all_data[sample_idxs],
        columns=[f"q_lam_{i}" for i in range(q_lam_dim)],
    )
    samples_df = pd.concat([lam_df, q_lam_df], axis=1)

    if outpath is not None:
        outpath = Path(outpath)
        if not outpath.exists():
            outpath.mkdir(parents=True, exist_ok=True)
        else:
            if outpath.is_file():
                raise ValueError(f"Outpath {outpath} already exists as a file.")

        data_df.to_csv(outpath / "data.csv")
        samples_df.to_csv(outpath / "samples.csv")
        pd.DataFrame(measurements).to_csv(
            outpath / "measurements.csv", index=False, header=False
        )

    ret = {"data": data_df, "samples": samples_df, "measurements": measurements}

    return ret


def plot_state(data, samples=None, mask=None, plot_intervals=None):
    """
    Plots the true state, observed state, and samples of the state, with
    state being the observed water level at the recording station in the grid.

    Parameters
    ----------
    samples : pd.DataFrame
        Samples from the initial distribution of parameter samples, and their
        associated states, that is water levels.
    mask : list
        List of indices to plot from the data.
    plot_intervals : list
        List of tuples, where each tuple is a name, a dictionary of arguments
        to to pass to ax.axvline, and a list of intervals to plot, intervals
        being a tuploe of tuples of start and end indices.

    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    mask = data.index.values if mask is None else mask
    sns.lineplot(
        data.iloc[mask], x="ts", y="q_lam_true_0", label="True", linestyle="--", ax=ax
    )
    sns.scatterplot(
        data.iloc[mask],
        x="ts",
        y="q_lam_obs_0",
        marker="*",
        label="Observed",
        ax=ax,
    )

    if samples is not None:
        cols = [
            c
            for c in samples.columns
            if c.startswith("q_lam_") and int(c.split("_")[-1]) in mask
        ]
        num_plot = 10
        for idx, i in enumerate(
            np.random.choice(range(len(samples)), num_plot, replace=False)
        ):
            to_plot = pd.DataFrame(
                samples.iloc[[i]][cols].values.T, columns=[f"q_lam_s={i}"]
            )
            to_plot["ts"] = data.iloc[mask]["ts"].values
            label = None if idx != num_plot - 1 else "Predicted"
            sns.lineplot(
                to_plot,
                x="ts",
                y=f"q_lam_s={i}",
                ax=ax,
                color="r",
                alpha=0.1,
                label=label,
            )

    plot_intervals = [] if plot_intervals is None else plot_intervals
    for name, args, intervals in plot_intervals:
        for interval in intervals:
            ax.axvline(data["ts"][interval[0]], **args)
        args["label"] = name
        ax.axvline(data["ts"][intervals[-1][-1]], **args)

    # ax.set_title('Time Window 3')
    # ax.set_title(f'lam_true = {data["lam_true_0"].values[0]}, {data["lam_true_1"].values[0]}')
    ax.set_ylabel("Water Elevation (m)")
    ax.set_xlabel("Time")
    ax.legend()

    return ax


def plot_iterations(prob, max_plot=10, plot_idxs=None, lam_true=None):
    """
    Plot updated marginal distributions for each parameter for each iteration.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    if plot_idxs is None:
        num_its = prob.pca_states["iteration"].max()
        if num_its > max_plot:
            plot_idxs = list(range(0, num_its, int(num_its / max_plot)))
            if (num_its - 1) not in plot_idxs:
                plot_idxs += [num_its - 1]
        else:
            plot_idxs = range(0, num_its)

    label = f"$\pi^{{in}}_{0}$"
    sns.kdeplot(prob.state, x="lam_0", ax=ax[0], label=label)
    sns.kdeplot(prob.state, x="lam_1", ax=ax[1], label=label)
    for idx, state in prob.pca_states.groupby("iteration"):
        if idx in plot_idxs:
            label = f"$\pi^{{up}}_{{{idx}}}$"
            sns.kdeplot(
                state,
                x="lam_0",
                weights=state["weight"] * state["ratio"],
                ax=ax[0],
                label=label,
            )
            sns.kdeplot(
                state,
                x="lam_1",
                weights=state["weight"] * state["ratio"],
                ax=ax[1],
                label=label,
            )

    if lam_true is not None:
        ax[0].axvline(
            lam_true[0], color="k", linestyle="--", label=f"$\lambda^{{\dagger}}$"
        )
        ax[1].axvline(
            lam_true[1], color="k", linestyle="--", label=f"$\lambda^{{\dagger}}$"
        )

    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel("$\lambda_1$")
    ax[1].set_xlabel("$\lambda_2$")

    return ax


def process_result(prob, lam_true, times):
    """
    Further processes ther result of an iterative PCAMUDProblem,
    that is solved by `solve_it`, by doing the following:

    1. Calculating l2_err according to true value passed in
    2. Calculating state error, where state here is defined as
    the learned qoi from the PCA map, not the true time series values.
    3.
    """

    # Calculate L2 error and Covariance for each
    res = prob.it_results.copy()
    mud_points = get_df(prob.it_results, "lam_MUD", 2)
    l2_errs = np.linalg.norm(mud_points - lam_true, axis=1)
    l2_errs = np.linalg.norm(mud_points - lam_true, axis=1)
    covs = []
    res["l2_err"] = l2_errs

    mud_states = get_df(
        prob.state.iloc[prob.it_results["MUD_idx"].values], "q_lam", prob.n_qoi
    )
    res["state_err"] = np.linalg.norm((mud_states.T - prob.data).T, axis=1)

    res["ts"] = [times[int(x.split(",")[-1][1:-1])] for x in res["pca_mask"].values]

    res["i"] = res["i"] + 1
    first = res.iloc[[0]].copy()
    first["i"] = 0
    first["ts"] = [times[eval(res["pca_mask"].values[0])[0]]]
    res = pd.concat([first, res])

    res["max_it"] = res["i"].max()

    return res


def iterative_trials(
    ds: np.ndarray,
    lam_true_idx: int = -1,
    mask: Union[List[int], range, np.ndarray] = None,
    num_splits: int = 10,
    std_dev: float = 0.05,
    num_samples: int = 999,
    pca_components: List[List[int]] = [[0]],
    num_trials: int = 10,
) -> pd.DataFrame:
    """
    Perform iterative trials.

    Iterative trials are staged as the following:
    For each trial, there are the following parameters:
    - Number of samples
    - Number of splits

    Each trial is performed by first sampling the dataset, using the provided parameters.

    Parameters:
    -----------
    ds : numpy.ndarray
        Input full dataset, as loaded by `pydci.examples.adcirc.load_full_ds`.
    num_splits : int
        Number of splits to use or a list of numbers of splits for iterations.
        Default is 10.
    std_dev : float, optional
        Standard deviation value. Default is 0.05.
    num_samples : int
        List of numbers of samples. Default is [999].
    pca_components : list of list of int, optional
        List of PCA components. Default is [[0]].
    num_trials : int, optional
        Number of trials. Default is 10.

    Returns:
    --------
    pandas.DataFrame
        Results of the iterative trials.
    """

    all_results = []
    mask = range(len(ds[1])) if mask is None else mask
    with alive_bar(
        num_trials,
        title=f"Iterative (nc = {len(pca_components[0])})",
        force_tty=True,
        length=20,
    ) as bar:
        for n in range(num_trials):
            ret = build_ds(
                lam_true_idx=lam_true_idx,
                data=ds,
                std_dev=std_dev,
                num_samples=num_samples,
            )
            pca = PCAMUDProblem(ret["samples"], ret["data"], std_dev)
            pca.solve_it(
                pca_mask=mask,
                pca_splits=num_splits,
                pca_components=pca_components,
                exp_thresh=1e10,
            )
            res = process_result(pca, ds[0][lam_true_idx], ds[2])
            res["num_iters"] = num_splits
            res["trial"] = n
            res["type"] = f"iterative ($\ell$ = {len(pca_components[0])})"
            all_results.append(res)
            bar()

    all_results = pd.concat(all_results)

    return all_results


def fixed_trials(
    ds: np.ndarray,
    lam_true_idx: int = -1,
    mask: Union[List[int], range, np.ndarray] = None,
    num_splits: int = 10,
    std_dev: float = 0.05,
    num_samples: int = 999,
    exp_thresh: float = 1e10,
    pca_components: List[List[int]] = [[0]],
    num_trials: int = 10,
) -> pd.DataFrame:
    """
    Perform fixed trials.

    Fixed trials are performed by splitting the dataset into fixed intervals and solving
    the problem for each interval separately.

    Parameters:
    -----------
    ds : numpy.ndarray
        Input full dataset, as loaded by `pydci.examples.adcirc.load_full_ds`.
    num_splits : int
        Number of splits to use or a list of numbers of splits for iterations.
        Default is 10.
    std_dev : float, optional
        Standard deviation value. Default is 0.05.
    num_samples : int, optional
        Number of samples. Default is 999.
    exp_thresh : float, optional
        Threshold for exponential value. Default is 1e10.
    pca_components : list of list of int, optional
        List of PCA components. Default is [[0]].
    num_trials : int, optional
        Number of trials. Default is 10.

    Returns:
    --------
    pandas.DataFrame
        Results of the fixed trials.
    """
    all_results = []
    mask = range(len(ds[1])) if mask is None else mask
    with alive_bar(
        num_trials * num_splits,
        title=f"Full (nc = {len(pca_components[0])})",
        force_tty=True,
        length=20,
    ) as bar:
        for n in range(num_trials):
            ret = build_ds(
                lam_true_idx=lam_true_idx,
                data=ds,
                std_dev=std_dev,
                num_samples=num_samples,
            )
            intervals = [
                range(mask[0], x[-1]) for x in np.array_split(mask, num_splits)
            ]
            for interval in intervals:
                pca = PCAMUDProblem(ret["samples"], ret["data"], std_dev)
                pca.solve_it(
                    pca_mask=interval,
                    pca_splits=1,
                    pca_components=pca_components,
                    exp_thresh=1e10,
                )
                all_results.append(process_result(pca, ds[0][lam_true_idx], ds[2]))
                all_results[-1]["trial"] = n
                all_results[-1]["num_iters"] = num_splits
                all_results[-1]["type"] = f"full ($\ell$ = {len(pca_components[0])})"
                bar()

    all_results = pd.concat(all_results)

    return all_results


def plot_metric(results, metric="e_r", figsize=(12, 5), ax=None, lineplot_kwargs=None):
    """
    Plot metric for scan results over time.
    """
    results = pd.concat(results) if isinstance(results, list) else results
    metric = [metric] if isinstance(metric, str) else metric
    if ax is None:
        fig, ax = plt.subplots(len(metric), 1, figsize=figsize)
    else:
        ax = [ax] if isinstance(ax, plt.Axes) else ax
        if len(ax) != len(metric):
            raise ValueError("Number of axes must match number of metrics")

    metric_labels = {
        "e_r": "$\mathbb{E}(r)$",
        "kl": "$\mathcal{D}_{KL}$",
        "l2_err": "$||\lambda^{MUD} - \lambda^{\dagger}||_{\ell_2}$",
        "state_err": "$||Q(\lambda^{MUD}) - Q(\lambda^{\dagger})||_{\ell_2}$",
    }
    metric_ttiles = {
        "e_r": "Predictability Assumption",
        "kl": "$\mathcal{D}_{KL}$",
        "l2_err": "$||\lambda^{MUD} - \lambda^{\dagger}||_{\ell_2}$",
        "state_err": "$||Q(\lambda^{MUD}) - Q(\lambda^{\dagger})||_{\ell_2}$",
    }
    results = results[
        [c for c in results.columns if c in ["i", "ts", "n_iters", "type"] + metric]
    ].copy()
    def_kwargs = {"x": "ts"}
    def_kwargs.update(lineplot_kwargs or {})
    for i, mc in enumerate(metric):
        def_kwargs["y"] = mc
        def_kwargs["ax"] = ax[i]
        sns.lineplot(results.dropna(), **def_kwargs)
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel(metric_labels[mc])

        if mc == "e_r":
            ax[i].axhline(y=1, color="black", linestyle="--")
            # ax[i].hlines(1, *ax[i].get_xlim(), color='red', linestyle='--')

    return ax
