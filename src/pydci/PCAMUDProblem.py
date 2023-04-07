import itertools
import pdb
import random
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import distributions as dist  # type: ignore
from scipy.stats import rv_continuous  # type: ignore

from pydci.log import logger
from pydci.MUDProblem import MUDProblem
from pydci.pca import pca
from pydci.utils import get_df, put_df


class PCAMUDProblem(MUDProblem):
    """
    Sets up Maxmal Update Density Inverse wroblem for parameter identification

    """

    def __init__(
        self,
        lam,
        q_lam,
        data,
        std_dev,
        max_nc: int = None,
        best_method: str = "closest",
        exp_thresh: float = 0.5,
        init_dist: rv_continuous = None,
        weights: ArrayLike = None,
        normalize: bool = False,
    ):
        # Since we aggregate -> Observed distribution fixed
        super().__init__(
            lam,
            q_lam,
            data,
            std_dev,
            init_dist=init_dist,
            weights=weights,
            normalize=normalize,
        )
        # Stash q_lam into qoi since q_lam = q_pca now
        self.qoi = self.q_lam
        max_nc = self.n_params if max_nc is None else max_nc
        self.max_nc = self.n_states if max_nc > self.n_states else max_nc
        self.exp_thresh = exp_thresh
        self.best_method = best_method
        self.pca_mask = None

    @property
    def n_qoi(self):
        return self.qoi.shape[1]

    def _validate_params(self, search_params):
        """
        Validate method parameters
        """
        am = ["closest", "min_kl", "max_kl"]
        if best := self.best_method not in am:
            raise ValueError(f"Unrecognized best method: {best}. Allowed: {am}")
        if ex := self.exp_thresh <= 0:
            raise ValueError(f"exp_thresh must be > 0: {ex}")

    def q_pca(self, mask=None):
        """
        Build QoI Map Using Data and Measurements

        Aggregate q_lam data with observed data for MUD convergence.
        """
        mask = np.arange(self.n_qoi) if self.pca_mask is None else self.pca_mask
        residuals = np.subtract(self.data[mask].T, self.qoi[:, mask]) / self.std_dev

        # Learn qoi to use using PCA
        pca_res, X_train = pca(residuals, n_components=self.max_nc)
        self.pca = {
            "X_train": X_train,
            "vecs": pca_res.components_,
            "var": pca_res.explained_variance_,
        }
        qoi = residuals @ pca_res.components_.T

        self.q_lam = qoi
        self.state = put_df(self.state, "q_pca", qoi, size=self.max_nc)

    def solve(self):
        """
        Solve problem
        """
        # Determine best MUD estimate
        self.q_pca()
        all_qoi = self.q_lam
        results_cols = (
            ["nc"] + [f"lam_MUD_{i}" for i in range(self.n_params)] + ["e_r", "kl"]
        )
        results = np.zeros((self.max_nc, self.n_params + 3))
        pred_dists = []
        nc_data_cols = {}
        for nc in range(1, self.max_nc + 1):
            self.q_lam = all_qoi[:, 0:nc]
            self.dists["observed"] = dist.norm(loc=nc * [0], scale=1)
            self.dists["predicted"] = None
            try:
                super().solve()
            except ValueError as v:
                if "array must not contain infs or NaNs" in str(v):
                    logger.error(f"Solve with {nc} components failed")
                    continue
                else:
                    raise v
            results[nc - 1, 0] = nc
            results[nc - 1, 1 : (self.n_params + 1)] = get_df(
                self.result, "lam_MUD", size=self.n_params
            )
            results[nc - 1, self.n_params + 1] = self.result["e_r"]
            results[nc - 1, self.n_params + 2] = self.result["kl"]

            cols_to_store = ["pi_obs", "pi_pr", "ratio", "pi_up"]
            for col in cols_to_store:
                nc_data_cols[f"{col}_nc={nc}"] = self.state[f"{col}"]
            pred_dists.append(self.dists["predicted"])

        self.pca_states = pd.concat(nc_data_cols, axis=1)
        # self.state = pd.concat([self.state, pca_states], axis=1)

        # Parse DataFrame with results of mud estimations for each ts choice
        res_df = pd.DataFrame(results, columns=results_cols)

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

        best_nc = int(res_df.loc[res_df[self.best_method].argmax()]["nc"])
        self.q_lam = all_qoi[:, 0:best_nc]
        self.dists["observed"] = dist.norm(loc=best_nc * [0], scale=1)
        self.dists["predicted"] = None
        super().solve()

        self.pca_result = res_df

    def plot_param_state(
        self,
        true_vals=None,
        ax=None,
        nc=None,
        param_idx=0,
        plot_mud=True,
        plot_initial=True,
        plot_legend=True,
    ):
        """
        Plotting functions for DCI Problem Class
        """
        ratio_col = "ratio" if nc is None else f"ratio_nc={nc}"
        ax, labels = super().plot_param_state(
            true_vals=true_vals,
            ax=ax,
            param_idx=param_idx,
            plot_mud=True,
            plot_initial=True,
            plot_legend=True,
            figsize=(8, 8),
            ratio_col=ratio_col,
        )

        return ax, labels

    def plot_obs_state(
        self,
        ax=None,
        nc=None,
        state_idx=0,
        plot_pf=True,
        plot_obs=True,
        plot_legend=True,
        figsize=(8, 8),
    ):
        """
        Plotting function for DCI Problem Class
        """
        ratio_col = "ratio" if nc is None else f"ratio_nc={nc}"
        ax, labels = super().plot_obs_state(
            ax=ax,
            state_idx=state_idx,
            plot_pf=plot_pf,
            plot_obs=plot_obs,
            plot_legend=plot_legend,
            obs_col="q_pca",
            ratio_col=ratio_col,
            figsize=figsize,
        )

        return ax, labels
