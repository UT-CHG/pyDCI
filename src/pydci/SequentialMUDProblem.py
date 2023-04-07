"""
Sequential MUD Estimation Algorithms

"""
import pdb
import random

import numpy as np
import pandas as pd
from rich.table import Table
from scipy.stats.distributions import norm

from pydci.log import log_table, logger
from pydci.PCAMUDProblem import PCAMUDProblem
from pydci.utils import get_df, put_df


class SequentialMUDProblem(PCAMUDProblem):
    """
    Class defining a SequentialDensity Problem for parameter estimation on.

    To initialize the class, a forward model model, and parameters need to be
    sepcified. The main entrypoint for solving the estimation problem is the
    `seq_solve()` method, with the `search_params` class attribute contorlling
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
        *args,
        qoi_method: str = "all",
        e_r_delta: float = 0.5,
        kl_thresh: float = 3.0,
        min_weight_thresh: float = 1e-4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.states = []
        self.results = []
        self.hist = {"results": [], "states": [], "data": [], "std_dev": []}
        self.e_r_delta = e_r_delta
        self.kl_thresh = kl_thresh
        self.min_weight_thresh = min_weight_thresh
        self.qoi_method = qoi_method
        self._validate_params()

    def _validate_params(self):
        """
        Validate Search params
        """
        if erd := self.e_r_delta < 0.5:
            raise ValueError(f"Shift detection delta(E(r)) must be >= 0.5: {erd}")
        if kl := self.kl_thresh < 3.0:
            raise ValueError(f"Shift detection D_KL_thresh(r) must be >= 3.0: {kl}")
        am = ["all", "linear", "random"]
        if qoi := self.qoi_method not in am:
            raise ValueError(f"Unrecognized qoi method: {qoi}. Allowed: {am}")

    def _get_qoi_combinations(
        self,
        max_tries=10,
    ):
        """
        Utility function to determine sets of ts combinations to iterate through
        """
        if self.qoi_method == "all":
            combs = [list(np.arange(self.n_qoi))]
        elif self.qoi_method == "linear":
            combs = [list(np.arange(0, i)) for i in range(self.max_nc, self.n_qoi)]
        elif self.qoi_method == "random":
            # Divide the max#tries amongs the number of timesteps available
            if self.n_qoi < max_tries:
                num_ts_list = range(self.max_nc, self.n_qoi + 1)
                tries_per = int(max_tries / self.n_qoi)
            else:
                num_ts_list = range(
                    self.max_nc, self.n_qoi + 1, int(self.n_qoi / max_tries)
                )
                tries_per = 1

            combs = []
            qoi_choices = range(0, self.n_qoi)
            for num_ts in num_ts_list:
                possible = list(
                    [list(x) for x in itertools.combinations(qoi_choices, num_ts)]
                )
                tries_per = tries_per if tries_per < len(possible) else len(possible)
                combs += random.sample(possible, tries_per)

        return combs

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

    def get_action(
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
    ):
        """
        Detect shift and determine next action.
        """
        qoi_combs = self._get_qoi_combinations()

        results = []
        state_cols = {}
        for q_idx, qc in enumerate(qoi_combs):
            self.pca_mask = qc
            super().solve()

            res_df = self.pca_result

            actions = []
            for nc, res in res_df.groupby("nc"):
                actions.append(self.get_action(res))
            res_df["action"] = actions

            cs = ["pi_obs", "pi_pr", "ratio", "pi_up"]
            for nc in range(self.max_nc):
                for col in cs:
                    col_name = f"{col}_{q_idx}_nc={nc+1}"
                    state_cols[col_name] = self.pca_states[f"{col}_nc={nc+1}"]

            results.append(res_df)

        self.search_states = pd.concat(state_cols, axis=1)

        res_df = pd.concat(results, keys=np.arange(len(qoi_combs)))
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

        # Resolve problem using optimal mask for pca data
        self.results = res_df
        best_idx = res_df[self.best_method].argmax()
        best_qoi_comb_idx = res_df.iloc[best_idx].name[0]
        self.pca_mask = qoi_combs[best_qoi_comb_idx]
        super().solve()
        self.result = res_df.iloc[[best_idx]]

    def update_iteration(
        self,
        lam,
        q_lam,
        data,
        std_dev,
        weights=None,
    ):
        """ """
        self.hist["results"].append(self.results)
        self.hist["states"].append(self.state)
        self.hist["data"].append(self.data)
        self.hist["std_dev"].append(self.std_dev)
        self.init_state(lam, q_lam)
        self.data = data
        self.std_dev = std_dev
        self.max_nc = self.n_states if self.n_params > self.n_states else self.n_params
        self.dists = {
            "initial": None,
            "predicted": None,
            "observed": norm(loc=self.max_nc * [0], scale=1),
            "updated": None,
        }
        self.set_weights(weights)

        if weights is not None:
            self.state["weight"] = weights

    def get_prev_best(
        self,
    ):
        """
        Get previous best from last iteration
        """
        if len(self.hist["results"]) < 1:
            return None
        prev = self.hist["results"][-1]
        prev_best_idx = prev[self.best_method].argmax()
        return prev.iloc[prev_best_idx]

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
