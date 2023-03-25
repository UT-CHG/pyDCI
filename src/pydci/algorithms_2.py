"""
Sequential MUD Estimation Algorithms

"""

import random
import pdb
import numpy as np

import pandas as pd
from scipy.integrate import odeint
from scipy.stats import beta
from scipy.stats import gaussian_kde as gkde
from scipy.stats import uniform

import sys

sys.path.insert(0, "/Users/rylanspence/Desktop/Git/DC/mud")
from mud.base import SpatioTemporalProblem as STP


def _try_mud(spt, nc=1, times_mask=None, weights=None):
    """
    Wrapper for trying a MUD problem and catching exceptions

    """
    try:
        # mud_prob = spt.mud_problem(
        #     num_components=nc, times_mask=times_mask, sample_weights=weights
        # )

        # RYLAN ADDED METHOD
        mud_prob = spt.mud_problem(
            method="pca",
            num_components=nc,
            times_mask=times_mask,
            sample_weights=weights,
        )
    except Exception as e:
        print(f"\t{nc}: - Unable to generate mud_problem: {e}")
        return None

    try:
        mud_prob.estimate()
        # print(f"TRYING MUD ESTIMATE {mud_prob.estimate()}")
    except Exception as e:
        print(f"\t{nc}: - Unable to create mud estimate: {e}")
        return None

    return mud_prob


def find_best_mud_estimate(
    spt, max_nc=2, exp_thresh=0.2, weights=None, print_summary=True
):
    """
    Find best MUD Estimate

    Given a SpatioTemporalProblem SPT, search for the best mud estimate using
    up to max_nc principal components, and from 1 to spt.num_ts data points.
    Best e_r is characterized by that which is closest to 1.

    TODO: implement random sampling and max number of tries.
    """
    res_by_nc = []
    for nc in range(1, max_nc + 1):
        best = None
        best_e_r = None
        probs = []
        e_rs = []
        best_num_ts = None
        for num_ts in range(spt.n_ts)[1:]:
            mud_prob = _try_mud(spt, nc=nc, times_mask=range(num_ts), weights=weights)
            if mud_prob is None:
                continue
            exp_r = mud_prob.expected_ratio()
            e_rs.append(exp_r)
            probs.append(mud_prob)
            if best is None or np.abs(1 - exp_r) < np.abs(1 - best_e_r):
                best = mud_prob
                best_num_ts = num_ts
                best_e_r = exp_r
        e_rs = np.array(e_rs)
        res_by_nc.append(
            {
                "nc": nc,
                "best": best,
                "probs": probs,
                "best_num_ts": best_num_ts,
                "e_rs": e_rs,
            }
        )

    best_idx = np.argmin([np.abs(1 - x["best"].expected_ratio()) for x in res_by_nc])
    best = res_by_nc[best_idx]["best"]
    best_num_ts = res_by_nc[best_idx]["best_num_ts"]
    best_e_r = best.expected_ratio()
    e_rs = res_by_nc[best_idx]["e_rs"]
    if print_summary:
        print("\t ... SEARCH SUMMARY ...")
        for r in res_by_nc:
            print(f"\t{r['nc']}: {r['best_num_ts']}/" + f"{spt.n_ts} points:")
            print(f"\t\tBest(E(r)) = {r['best'].expected_ratio()}")
            print(f"\t\tBest(Mud) = {r['best'].estimate()}")
            print(
                f"\t\tMean(E(r)) = {r['e_rs'].mean()}, "
                + f"STD(E(r)) = {r['e_rs'].std()}"
            )
            print(
                f"\t\tMIN(E(r)) = {r['e_rs'].min()}, "
                + f"MAX(E(r))) = {r['e_rs'].max()}"
            )

    res = {
        "best": {
            "prob": best,
            "nc": res_by_nc[best_idx]["nc"],
            "num_ts": best_num_ts,
            "e_r": best_e_r,
            "e_rs": e_rs,
        },
        "all": res_by_nc,
    }

    return res


def get_uniform_box(center, factor=0.5, mins=None, maxs=None):
    """
    Generate a domain of [min, max] values
    """
    center = np.array(center)
    if np.sum(center) != 0.0:
        loc = center - np.abs(center * factor)
        scale = 2 * center * factor
    else:
        loc = center - factor
        scale = 2 * factor
    domain = np.array(list(zip(loc, np.array(loc) + np.array(scale))))
    if mins is not None:
        for i, d in enumerate(domain):
            if d[0] < mins[i]:
                d[0] = mins[i]
    if maxs is not None:
        for i, d in enumerate(domain):
            if d[1] > maxs[i]:
                d[1] = maxs[i]

    loc = domain[:, 0]
    scale = domain[:, 1] - domain[:, 0]

    return loc, scale, domain


def compute_push_forward(run_model, samples, xi, times):
    push_forward = [run_model(xi, times, tuple(s)) for s in samples]
    push_forward_array = np.array(push_forward)

    return push_forward_array


def convert_push_forward(push_forward_array, interval):
    number_runs, number_timesteps, number_states = push_forward_array.shape
    header = [
        [f"Push_Forward_{i}" for i in range(number_runs) for j in range(number_states)],
        [f"X_{j}" for j in range(number_states)] * number_runs,
    ]
    reshaped_array = push_forward_array.reshape(
        number_timesteps, number_runs, number_states
    )
    push_forward_df = pd.DataFrame(
        reshaped_array.reshape(number_timesteps, -1), columns=header
    )

    time_interval = pd.Categorical([f"Time_Interval_{interval}"] * len(push_forward_df))
    push_forward_df["Interval"] = time_interval

    return push_forward_df


def correlation_matrix(cov_matrix):
    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    return Dinv @ cov_matrix @ Dinv


def sequential_resampling(
    run_model,
    x0,
    true_param,
    num_samples=1000,
    measurement_noise=0.05,
    diff=0.5,
    solve_ts=0.2,
    sample_ts=1,
    time_window=10,
    end_time=200,
    exp_thresh=0.1,
    param_shifts=[100, [0.5, 0.5, 0.8, 0.5]],
    max_nc=4,
    reweight=True,
    min_weight_thresh=1e-5,
    param_mins=None,
    param_maxs=None,
    init_seed=None,
    obs_seed=None,
):
    """ """

    iteration = 0
    sample_step = int(sample_ts / solve_ts)
    solve_step = int(time_window / solve_ts)
    times = np.linspace(0.0, time_window, solve_step)

    loc, scale, domain = get_uniform_box(
        true_param, factor=diff, mins=param_mins, maxs=param_maxs
    )
    print(f"Initi: Uniform over {domain}")

    np.random.seed(init_seed)  # Initial seed for sampling

    samples = uniform.rvs(loc=loc, scale=scale, size=(num_samples, len(true_param)))
    weights = None

    # Initial state of system
    xi = x0

    ts = []
    pfs = []
    tvs = []
    stps = []
    mud_probs = []
    sample_decision = []
    covariance = []
    correlation = []

    up_dist = gkde(samples.T)

    while times[-1] < end_time:
        print("")
        print(f"====== iteration {iteration} ======")
        if len(param_shifts) > 0:
            if times[0] >= param_shifts[-1][0]:
                p_shift = param_shifts.pop()
                print(f"!!!! SHIFTING PARAM TO {p_shift} !!!!")
                true_param = p_shift[1]
        print(f"True Param: {true_param}")
        true_vals = run_model(xi, times, tuple(true_param))

        print(f"Pushing {num_samples} samples forward through model")

        # push_forward = []
        # for s in samples:
        #     push_forward.append(run_model(xi, times, tuple(s)))
        # push_forward = np.array(push_forward)

        push_forward = compute_push_forward(run_model, samples, xi, times)

        # Aggregate data using Q_PCA map and generate MUD estimate
        stp = STP(
            df={
                "sample_dist": "u",
                "domain": domain,
                "sensors": np.zeros((x0.shape[0], 2)),
                "times": times[0:-1:sample_step],
                "lam_ref": true_param,
                "std_dev": measurement_noise,
                "true_vals": true_vals[0:-1:sample_step],
                "lam": samples,
                "data": push_forward[:, 0:-1:sample_step, :],
            }
        )
        stp.measurements_from_reference(seed=obs_seed)

        res = find_best_mud_estimate(
            stp, max_nc=max_nc, weights=weights, print_summary=True
        )

        best = res["best"]["prob"]
        print(
            f"\n\tBest mud estimate nc = {res['best']['nc']}, num_ts = "
            + f"{res['best']['num_ts']}:"
        )
        print(f"\tmud point = {res['best']['prob'].estimate()}")
        print(f"\tE(r) = {res['best']['e_r']}")
        print(f"\tmin/max = {(best._r.min(), best._r.max())}")

        # Generate new set of samples for next iteration
        if (
            res["best"]["e_r"] < 1.0 + exp_thresh
            and res["best"]["e_r"] > 1.0 - exp_thresh
        ):
            if reweight and best._r.min() > min_weight_thresh:
                print(
                    "\tUpdating weights: e_r within threshold 1+-"
                    + f"{exp_thresh}, reweight set, and min weight "
                    + f"> {min_weight_thresh}"
                )
                weights = best._r
                sample_decision.append("reweight")
            else:
                print(f"\tDrawing new samples: e_r within threshold {exp_thresh}")
                up_dist = gkde(samples.T, weights=best._r)
                samples = up_dist.resample(size=num_samples).T
                sample_decision.append("resample_updated")
        elif np.mean(res["best"]["e_rs"]) < 1e-1:
            print("\t!!! Expected ratios for all attemps were < 1e-1 !!!")
            print("\t!!! Parameter shift suspected, inflating search domain !!!")
            ### How do we resample when a shift is expected??
            # loc, scale, domain = get_uniform_box(
            #         res['best']['prob'].estimate(),
            #         factor=diff,
            #         mins=param_mins,
            #         maxs=param_maxs)
            print(f"\t!!! Resetting to: Uniform over {domain}")
            samples = uniform.rvs(
                loc=loc, scale=scale, size=(num_samples, len(true_param))
            )
            sample_decision.append("resample_initial")
            weights = None
        else:
            print("\t Resampling from previous updated distribution")
            samples = up_dist.resample(size=num_samples).T
            sample_decision.append("resample_previous_updated")

        # covariance.append(np.cov(samples.T, aweights=best._r, ddof=0))
        # correlation.append(correlation_matrix(covariance[-1]))

        ts.append(times)
        mud_probs.append(res)
        pfs.append(convert_push_forward(push_forward, iteration))
        tvs.append(true_vals)
        stps.append(stp)

        iteration = iteration + 1
        xi = true_vals[-1]
        times = np.linspace(times[-1], times[-1] + time_window, solve_step)

    ret = {
        "domain": domain,
        "times": ts,
        "push_forwards": pfs,
        "true_values": tvs,
        "spatio_temporal_probs": stps,
        "mud_probs": mud_probs,
        "covariance": covariance,
        "correlation": correlation,
    }

    return ret
