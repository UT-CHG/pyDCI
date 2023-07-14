"""
pyDCI Utilities

"""
import pdb
from typing import List, Tuple, Union, Dict, List, Any

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde
from itertools import product


class KDEError(Exception):
    def __init__(
        self,
        X,
        name=None,
        weights=None,
        msg="Failed to compute gaussian_kde using scipy",
    ):
        self.name = name
        self.X = X
        self.weights = weights
        self.msg = msg

    def __str__(self):
        if self.name is not None:
            msg = f"{self.name}:{self.msg}"
        else:
            msg = f"{self.msg}"
        msg += f":\nshape: {self.X.shape}"
        if self.weights is not None:
            msg += f", sum(weights): {np.sum(self.weights)}"
        return msg


def gkde(X, weights=None, label=None):
    """
    Try to compute gkde using scipy, catching common errors
    """
    try:
        res = gaussian_kde(X, weights=weights)
    except (LinAlgError, ValueError) as e:
        if "array must not contain infs or NaNs" in str(e):
            msg = "scipy gaussian KDE failed because weights too small"
            raise KDEError(X, weights=weights, name=label, msg=msg)
        elif "data appears to lie in a lower-dimensional" in str(e):
            msg = "scipy gaussian KDE failed - Degenerative data covariance."
            msg += "Can mean weights are too small or too few samples."
            raise KDEError(X, weights=weights, name=label, msg=msg)
        else:
            raise e
    else:
        return res


def add_noise(signal: ArrayLike, sd: float = 0.05, seed: int = None):
    """
    Add Noise

    Add noise to synthetic signal to model a real measurement device. Noise is
    assumed to be from a standard normal distribution std deviation `sd`:

    $\\mathcal{N}(0,\\sigma)$

    Paramaters
    ---------
    signal : numpy.typing.ArrayLike
      Signal to add noise to.
    sd : float, default = 0.05
      Standard deviation of error to add.
    seed : int, optional
      Seed to use for numpy random number generator.

    Returns
    -------
    noisy_signal: numpy.typing.ArrayLike
      Signal with noise added to it.

    Example Usage
    -------------
    Generate test signal, add noise, check average distance
    >>> seed = 21
    >>> test_signal = np.ones(5)
    >>> noisy_signal = add_noise(test_signal, sd=0.05, seed=21)
    >>> np.round(1000*np.mean(noisy_signal-test_signal))
    4.0
    """
    signal = np.array(signal)

    if seed is not None:
        np.random.seed(seed)

    # Populate qoi_true with noise
    noise = np.random.randn(signal.size) * sd

    return signal + noise


def fit_domain(
    x: np.ndarray = None, min_max_bounds: np.ndarray = None, pad_ratio: float = 0.1
) -> np.ndarray:
    """
    Fit domain bounding box to array x

    Parameters
    ----------
    x : ArrayLike
        2D array to calculate min, max values along columns.
    pad_ratio : float, default=0.1
        What ratio of total range=max-min to subtract/add to min/max values to
        construct final domain range. Padding is done per x column dimension.

    Returns
    -------
    min_max_bounds : ArrayLike
        Domain fitted to values found in 2D array x, with padding added.

    Examples
    --------
    Input must be 2D. Set pad_ratio = 0 to get explicit min/max bounds
    >>> fit_domain(np.array([[1, 10], [0, -10]]), pad_ratio=0.0)
    array([[  0,   1],
           [-10,  10]])

    Can extend domain around the array values using the pad_ratio argument.

    >>> fit_domain(np.array([[1, 10], [0, -10]]), pad_ratio=1)
    array([[ -1,   2],
           [-30,  30]])
    """
    if min_max_bounds is None:
        if x is None:
            raise ValueError("Both x and min_max_bounds can't be None")
        min_max_bounds = np.array([x.min(axis=0), x.max(axis=0)]).T
    pad = pad_ratio * (min_max_bounds[:, 1] - min_max_bounds[:, 0])
    min_max_bounds[:, 0] = min_max_bounds[:, 0] - pad
    min_max_bounds[:, 1] = min_max_bounds[:, 1] + pad
    return min_max_bounds


def set_shape(array: np.ndarray, shape: Union[List, Tuple] = (1, -1)) -> np.ndarray:
    """Resizes inputs if they are one-dimensional."""
    return array.reshape(shape) if array.ndim < 2 else array


def get_uniform_box(center, factor=0.5, mins=None, maxs=None):
    """
    Generate a domain of [min, max] values around a center value.
    """
    center = np.array(center)
    if np.sum(center) != 0.0:
        loc = center - np.abs(center * factor)
        scale = np.abs(2 * center * factor)
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

    return domain


def put_df(df, name, val, size=1, mask=None):
    """
    Given an n-m dimensional `val`, stores into dataframe `df` with `n`
    rows by unpacking the `m` columns of val into separate columns with
    names `{name}_{j}` where j is the index of the column.
    """
    idxs = range(size)
    if mask is not None:
        pdb.set_trace()
        idxs = [i for i in idxs if mask[i]]
    if len([x for x in df.columns if x.startswith(f"{name}_")]) > 0:
        for i, idx in enumerate(idxs):
            df[f"{name}_{i}"] = val[:, idx]
    else:
        concat_cols = {}
        for i, idx in enumerate(idxs):
            concat_cols[f"{name}_{i}"] = val[:, idx]
        df = pd.concat([df, pd.DataFrame(concat_cols)], axis=1)

    return df


def get_df(df, name, size=1):
    """
    Gets an n-m dimensional `val` from `df` with `n` columns by retrieving
    the `m` columns of val into from columns of `df` with names `{name}_{j}`
    where j is the index of the column.
    """
    val = np.zeros((df.shape[0], size))
    for idx in range(size):
        val[:, idx] = df[f"{name}_{idx}"].values
    return val


def closest_factors(n: int) -> Tuple[int, int]:
    """
    Returns the two closest factors of an integer.

    Parameters
    ----------
    n : int
        The integer to find the closest factors of.

    Returns
    -------
    Tuple[int, int]
        A tuple of two integers that are the closest factors of `n`.
    """
    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return (i, n // i)

def generate_combinations(args_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate a list of dictionaries with every combination of possible arguments for each key in the input dictionary.

    Parameters
    ----------
    args_dict : dict
        A dictionary where each key is a parameter name and each value is a list of possible arguments for that parameter.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary contains one combination of possible arguments for each key in the input dictionary.
    """
    keys = args_dict.keys()
    values = args_dict.values()
    combinations = list(product(*values))
    return [dict(zip(keys, combination)) for combination in combinations]
