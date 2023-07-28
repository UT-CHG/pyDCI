import importlib
import pdb

import matplotlib.pyplot as plt
import numpy as np

from pydci import Model


class Monomial1D(Model.DynamicModel):
    def __init__(
        self,
        p,
        x0=[0],  # Note in the constant monomial case, initial state won't matter
        lam_true=[0.25**0.2],
        solve_ts=1.0,
        sample_ts=1.1,
        measurement_noise=0.1,
        **kwargs,
    ):
        self.p = p
        super().__init__(
            x0,
            lam_true,
            solve_ts=solve_ts,
            sample_ts=sample_ts,
            measurement_noise=measurement_noise,
            **kwargs,
        )

    def forward_model(
        self,
        x0,
        times,
        lam,
    ):
        """
        Monomial Forward Model
        """
        # return np.array([[lam[0] ** self.p]])
        res = np.repeat(
            np.array([[lam[0] ** self.p]]), len(times), axis=0
        )
        return res


class Monomial2D(Model.DynamicModel):
    def __init__(
        self,
        p,
        x0=[0, 0],  # Note in the constant monomial case, initial state won't matter
        lam_true=[0.3, 0.8],
        solve_ts=1.0,
        sample_ts=1.0,
        measurement_noise=0.05,
        **kwargs,
    ):
        self.p = p
        super().__init__(
            x0,
            lam_true,
            solve_ts=solve_ts,
            sample_ts=sample_ts,
            measurement_noise=measurement_noise,
            **kwargs,
        )

    def forward_model(
        self,
        x0,
        times,
        lam,
    ):
        """
        Monomial Forward Model

        Static in time (tim array ignored)
        """
        res = np.repeat(
            np.array([[lam[0] ** self.p, lam[1] ** self.p]]), len(times), axis=0
        )
        return res

    def plot_states(
        self,
    ):
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        for i, ax in enumerate(ax):
            self.plot_state(state_idx=i, ax=ax)
            ax.set_title(f"{i}: Temporal Evolution")
